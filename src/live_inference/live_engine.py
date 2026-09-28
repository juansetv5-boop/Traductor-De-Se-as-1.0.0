"""
live_engine.py — Motor de Inferencia en Tiempo Real para el Traductor Alpha LSC Dinámico.
Orquesta: Captura OpenCV -> MediaPipe Holistic -> Buffer Circular (T=40) -> Inferencia GPU (TCN)
-> Detección Cinemática de Apex -> Disparo Asíncrono de TTS con Cero Desfase Perceptual.
"""

from __future__ import annotations

import collections
import os
import sys
import time
from typing import Deque, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import cv2
import numpy as np
import torch

from src.audio.async_tts import AsyncTTSEngine
from src.config_dynamic import (
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CONFIDENCE_THRESHOLD,
    INPUT_FEATURES_PER_FRAME,
    MODEL_WEIGHTS_PATH,
    NUM_CLASSES,
    PILOT_VOCABULARY,
    RAW_SPATIAL_FEATURES,
    TARGET_FPS,
    TEMPORAL_WINDOW_SIZE,
    TRACED_MODEL_PATH,
)
from src.data_collection.feature_extractor import HolisticFeatureExtractor
from src.live_inference.apex_detector import ApexKinematicDetector
from src.models.tcn_model import LSCDynamicTCNClassifier, build_tcn_model


class LSCLiveEngine:
    """
    Motor orquestador de inferencia continua y síntesis de voz en tiempo real.
    """

    def __init__(
        self,
        model_path: str = MODEL_WEIGHTS_PATH,
        traced_path: str = TRACED_MODEL_PATH,
        device_str: Optional[str] = None,
    ) -> None:
        if device_str:
            self.device = torch.device(device_str)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[MOTOR LSC] Inicializando en dispositivo: {self.device}")

        # 1. Cargar Modelo (TorchScript si está disponible para máxima velocidad, o PyTorch estándar)
        self.model = self._load_model(model_path, traced_path)

        # 2. Inicializar componentes auxiliares
        self.extractor = HolisticFeatureExtractor()
        self.apex_detector = ApexKinematicDetector()
        self.tts = AsyncTTSEngine()

        # 3. Buffer circular temporal (FIFO de T=40 frames espaciales)
        self.spatial_buffer: Deque[np.ndarray] = collections.deque(maxlen=TEMPORAL_WINDOW_SIZE)

        # 4. Estado de inferencia
        self.last_prediction: str = "Neutro"
        self.last_confidence: float = 0.0
        self.last_apex_trigger: str = ""
        self.trigger_display_timer: float = 0.0
        self.fps: float = 0.0
        self.current_camera_index: int = CAMERA_INDEX

    def _load_model(self, weights_path: str, traced_path: str) -> torch.nn.Module:
        """Carga el modelo compilado TorchScript o los pesos de PyTorch."""
        if os.path.exists(traced_path):
            try:
                print(f"[MODELO] Cargando modelo optimizado TorchScript desde: {traced_path}")
                model = torch.jit.load(traced_path, map_location=self.device)
                model.eval()
                return model
            except Exception as e:
                print(f"[AVISO] Falló carga de TorchScript ({e}), recurriendo a pesos estándar...")

        model = build_tcn_model(self.device)
        if os.path.exists(weights_path):
            print(f"[MODELO] Cargando pesos desde: {weights_path}")
            model.load_state_dict(torch.load(weights_path, map_location=self.device))
        else:
            print(f"[AVISO] No se encontraron pesos entrenados en '{weights_path}'. Usando modelo base.")
        model.eval()
        return model

    @torch.no_grad()
    def _run_gpu_inference(self) -> Tuple[int, float, np.ndarray]:
        """
        Ejecuta la inferencia en GPU sobre el buffer temporal actual de 40 frames.
        Retorna (predicted_class_idx, confidence, probabilities_array).
        """
        # Convertir buffer deque a array (40, 147)
        spatial_seq = np.array(self.spatial_buffer, dtype=np.float32)

        # Calcular derivadas de velocidad y aceleración: (40, 441)
        kinematics = HolisticFeatureExtractor.compute_temporal_kinematics(spatial_seq)

        # Reorganizar a formato Conv1d: (1, Channels=441, Time=40)
        tensor_in = (
            torch.from_numpy(np.transpose(kinematics, (1, 0)))
            .unsqueeze(0)
            .to(self.device, non_blocking=True)
        )

        logits = self.model(tensor_in)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        return pred_idx, confidence, probs

    def run(self) -> None:
        """Inicia el ciclo principal de captura, inferencia y rendering."""
        cap = cv2.VideoCapture(self.current_camera_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

        if not cap.isOpened():
            print(f"[ERROR] No se pudo acceder a la cámara (índice {self.current_camera_index}, backend CAP_DSHOW)")
            return

        # Validación inmediata del primer frame antes de entrar al bucle
        first_ret, first_frame = cap.read()
        if not first_ret or first_frame is None:
            cap.release()
            print(f"[ERROR] No se pudo leer el primer frame de la cámara (índice {self.current_camera_index}, backend CAP_DSHOW).")
            print("[DIAGNÓSTICO] Verifica que la cámara no esté en uso por otra app o prueba con CAMERA_INDEX = 1.")
            return

        print("\n" + "=" * 70)
        print("  TRADUCTOR ALPHA LSC — MOTOR DINÁMICO EN VIVO (GPU + APEX TTS)")
        print("=" * 70)
        print("Instrucciones: Realice una seña continua. Presione [Q] para salir.\n")

        prev_time = time.time()
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                loop_start = time.time()

                # ─── 1. Extracción de Keypoints ──────────────────────────────
                results = self.extractor.process_frame(frame)
                spatial_vec, wrist_positions = self.extractor.extract_spatial_vector(results)

                # Agregar al buffer circular
                self.spatial_buffer.append(spatial_vec)

                # Determinar muñeca activa (mano derecha preferida o izquierda)
                active_wrist = wrist_positions["right_wrist"]
                if active_wrist is None:
                    active_wrist = wrist_positions["left_wrist"]

                # ─── 2. Detección Cinemática de Apex ─────────────────────────
                is_apex, current_vel, peak_vel = self.apex_detector.update(active_wrist)

                # ─── 3. Inferencia GPU (Cuando el buffer está completo) ───────
                all_probs = np.zeros(NUM_CLASSES, dtype=np.float32)

                if len(self.spatial_buffer) == TEMPORAL_WINDOW_SIZE:
                    pred_idx, conf, all_probs = self._run_gpu_inference()
                    self.last_prediction = PILOT_VOCABULARY[pred_idx]
                    self.last_confidence = conf

                    # ─── 4. Trigger de TTS Inteligente ───────────────────────
                    # Se dispara si:
                    #  a) Se detecta el Apex cinemático (fin del stroke)
                    #  b) La confianza supera el umbral (0.85)
                    #  c) No es la clase 'Neutro'
                    if (
                        is_apex
                        and self.last_confidence >= CONFIDENCE_THRESHOLD
                        and self.last_prediction != "Neutro"
                    ):
                        spoken = self.tts.speak(self.last_prediction)
                        if spoken:
                            self.last_apex_trigger = self.last_prediction
                            self.trigger_display_timer = time.time()
                            print(f"[TRIGGER APEX] ¡Seña detectada y sintetizada!: {self.last_prediction} ({self.last_confidence*100:.1f}%)")

                # ─── 5. Renderizado del HUD y Visualización ──────────────────
                annotated = self.extractor.draw_landmarks(frame, results)
                self._draw_hud(annotated, all_probs, current_vel, is_apex)

                # Cálculo de FPS
                frame_count += 1
                curr_time = time.time()
                if curr_time - prev_time >= 1.0:
                    self.fps = frame_count / (curr_time - prev_time)
                    frame_count = 0
                    prev_time = curr_time

                cv2.imshow("Traductor Alpha LSC - Dynamic Engine", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("}"):
                    print("[HOT-SWAP] Cambiando de cámara...")
                    cap.release()
                    self.current_camera_index += 1
                    cap = cv2.VideoCapture(self.current_camera_index, cv2.CAP_DSHOW)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
                    test_ret, _ = cap.read()
                    if not test_ret:
                        print(f"[HOT-SWAP] Cámara {self.current_camera_index} no disponible. Volviendo a índice 0.")
                        self.current_camera_index = 0
                        cap.release()
                        cap = cv2.VideoCapture(self.current_camera_index, cv2.CAP_DSHOW)
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
                        cap.read()  # Dummy read

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.extractor.close()
            self.tts.stop()
            print("[INFO] Motor apagado correctamente.")

    def _draw_hud(
        self,
        frame: np.ndarray,
        probs: np.ndarray,
        velocity: float,
        is_apex: bool,
    ) -> None:
        """Dibuja la interfaz visual con métricas, barras de probabilidad y traducción activa."""
        h, w, _ = frame.shape

        # Barra superior oscura
        cv2.rectangle(frame, (0, 0), (w, 65), (20, 20, 20), -1)
        cv2.line(frame, (0, 65), (w, 65), (0, 220, 255), 2)
        
        # Cámara activa
        cv2.putText(
            frame,
            f"CAM: {self.current_camera_index} (Presione '}}' para alternar)",
            (w - 320, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 150),
            1,
        )

        # FPS y Dispositivo
        dev_label = "GPU: RTX" if "cuda" in str(self.device) else "CPU"
        cv2.putText(
            frame,
            f"{dev_label} | {self.fps:.1f} FPS",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 180),
            2,
        )

        # Velocidad cinemática
        vel_color = (0, 0, 255) if is_apex else (0, 255, 255)
        cv2.putText(
            frame,
            f"Velocidad: {velocity*100:.1f} {'[APEX!]' if is_apex else ''}",
            (15, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            vel_color,
            1,
        )

        # Predicción actual en tiempo real
        pred_text = f"Candidato: {self.last_prediction} ({self.last_confidence*100:.1f}%)"
        cv2.putText(
            frame,
            pred_text,
            (w - 360, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        # Subtítulo de Traducción Activada (TTS Banner)
        if time.time() - self.trigger_display_timer < 2.0 and self.last_apex_trigger:
            cv2.rectangle(frame, (w // 2 - 200, h - 85), (w // 2 + 200, h - 25), (0, 160, 0), -1)
            cv2.rectangle(frame, (w // 2 - 200, h - 85), (w // 2 + 200, h - 25), (255, 255, 255), 2)
            cv2.putText(
                frame,
                f"Voz: \"{self.last_apex_trigger}\"",
                (w // 2 - 160, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )

        # Barras de probabilidad en el costado izquierdo inferior
        y_start = h - (len(PILOT_VOCABULARY) * 22) - 10
        for i, class_name in enumerate(PILOT_VOCABULARY):
            p = float(probs[i]) if i < len(probs) else 0.0
            y_pos = y_start + (i * 22)
            # Barra de fondo
            cv2.rectangle(frame, (15, y_pos), (165, y_pos + 14), (40, 40, 40), -1)
            # Barra de progreso
            bar_w = int(150 * p)
            color = (0, 220, 255) if p >= CONFIDENCE_THRESHOLD else (180, 180, 180)
            cv2.rectangle(frame, (15, y_pos), (15 + bar_w, y_pos + 14), color, -1)
            cv2.putText(
                frame,
                f"{class_name[:9]}: {p*100:.0f}%",
                (175, y_pos + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (230, 230, 230),
                1,
            )


def main() -> None:
    engine = LSCLiveEngine()
    engine.run()


if __name__ == "__main__":
    main()
