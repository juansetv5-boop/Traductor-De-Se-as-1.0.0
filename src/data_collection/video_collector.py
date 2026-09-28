"""
video_collector.py — Recolector interactivo de secuencias de video continuas para LSC.
Implementa el protocolo anti-data leakage: organización por sujetos (subject-wise) y
registro de gestos completos (Reposo -> Gesto -> Reposo).
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import time
from typing import Optional

import cv2
import numpy as np

from src.config_dynamic import (
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    PILOT_VOCABULARY,
    TARGET_FPS,
    VIDEOS_DIR,
)
from src.data_collection.feature_extractor import HolisticFeatureExtractor


class DynamicVideoCollector:
    """
    Graba secuencias de video para el vocabulario dinámico de LSC con separación estricta por sujetos.
    """

    def __init__(
        self,
        subject_id: str,
        output_dir: str = VIDEOS_DIR,
        samples_per_gesture: int = 15,
        clip_duration_sec: float = 1.6,
    ) -> None:
        self.subject_id = subject_id
        self.output_dir = output_dir
        self.samples_per_gesture = samples_per_gesture
        self.clip_duration_sec = clip_duration_sec
        self.target_frames = int(clip_duration_sec * TARGET_FPS)
        self.current_camera_index = CAMERA_INDEX

        self.extractor = HolisticFeatureExtractor()
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_gesture_dir(self, gesture: str) -> str:
        gesture_dir = os.path.join(self.output_dir, self.subject_id, gesture)
        os.makedirs(gesture_dir, exist_ok=True)
        return gesture_dir

    def run(self) -> None:
        """Inicia el bucle de captura interactivo con cámara OpenCV."""
        cap = cv2.VideoCapture(self.current_camera_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

        if not cap.isOpened():
            print(f"[ERROR] No se pudo abrir la cámara (índice {self.current_camera_index}, backend CAP_DSHOW).")
            return

        # Validación inmediata del primer frame antes de entrar al bucle
        first_ret, first_frame = cap.read()
        if not first_ret or first_frame is None:
            cap.release()
            print(f"[ERROR] No se pudo leer el primer frame de la cámara (índice {self.current_camera_index}, backend CAP_DSHOW).")
            print("[DIAGNÓSTICO] Verifica que la cámara no esté en uso por otra app o prueba con CAMERA_INDEX = 1.")
            return

        print("\n" + "=" * 70)
        print(f"  RECOLECTOR DE GESTOS LSC — SUJETO: {self.subject_id}")
        print("=" * 70)
        print("Instrucciones:")
        print("  - Barra ESPACIADORA: Iniciar cuenta regresiva y grabación")
        print("  - Tecla [N]: Siguiente seña del vocabulario")
        print("  - Tecla [Q]: Salir y guardar sesión\n")

        # Excluimos la clase 'Neutro' de las señas a grabar explícitamente, o permitimos grabarla como reposo
        gestures_to_record = PILOT_VOCABULARY
        current_gesture_idx = 1 if len(gestures_to_record) > 1 else 0

        while True:
            current_gesture = gestures_to_record[current_gesture_idx]
            gesture_dir = self._get_gesture_dir(current_gesture)
            existing_clips = [f for f in os.listdir(gesture_dir) if f.endswith(".mp4")]
            recorded_count = len(existing_clips)

            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Fallo al capturar frame.")
                break

            frame = cv2.flip(frame, 1)  # Modo espejo
            results = self.extractor.process_frame(frame)
            annotated = self.extractor.draw_landmarks(frame, results)

            # HUD informativo
            cv2.rectangle(annotated, (0, 0), (CAMERA_WIDTH, 70), (25, 25, 25), -1)
            cv2.putText(
                annotated,
                f"Sujeto: {self.subject_id} | Gesto [{current_gesture_idx}/{len(gestures_to_record)-1}]: {current_gesture.upper()}",
                (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 200),
                2,
            )
            cv2.putText(
                annotated,
                f"CAM: {self.current_camera_index} (Presione '}}' para alternar)",
                (CAMERA_WIDTH - 320, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 150),
                1,
            )
            cv2.putText(
                annotated,
                f"Muestras: {recorded_count}/{self.samples_per_gesture} | [ESPACIO]=Grabar  [N]=Sig  [Q]=Salir",
                (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )

            cv2.imshow("Traductor Alpha LSC - Dynamic Collector", annotated)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("[INFO] Finalizando sesión de recolección.")
                break
            elif key == ord("n"):
                current_gesture_idx = (current_gesture_idx + 1) % len(gestures_to_record)
            elif key == ord(" "):
                # Cuenta regresiva
                self._countdown(cap, current_gesture)
                # Grabar clip
                self._record_clip(cap, current_gesture, recorded_count)
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

        cap.release()
        cv2.destroyAllWindows()
        self.extractor.close()

    def _countdown(self, cap: cv2.VideoCapture, gesture: str) -> None:
        """Muestra cuenta regresiva de 3 segundos antes de capturar el gesto."""
        for count in [3, 2, 1]:
            start_t = time.time()
            while time.time() - start_t < 0.7:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                results = self.extractor.process_frame(frame)
                annotated = self.extractor.draw_landmarks(frame, results)

                cv2.putText(
                    annotated,
                    f"PREPARAR: {gesture}",
                    (CAMERA_WIDTH // 2 - 140, CAMERA_HEIGHT // 2 - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    annotated,
                    str(count),
                    (CAMERA_WIDTH // 2 - 25, CAMERA_HEIGHT // 2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.5,
                    (0, 0, 255),
                    4,
                )
                cv2.imshow("Traductor Alpha LSC - Dynamic Collector", annotated)
                cv2.waitKey(1)

    def _record_clip(self, cap: cv2.VideoCapture, gesture: str, sample_idx: int) -> None:
        """Graba un clip continuo de T fotogramas exactos."""
        gesture_dir = self._get_gesture_dir(gesture)
        timestamp = int(time.time() * 1000)
        video_filename = f"{gesture}_{self.subject_id}_{timestamp}.mp4"
        video_path = os.path.join(gesture_dir, video_filename)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, TARGET_FPS, (CAMERA_WIDTH, CAMERA_HEIGHT))

        frames_captured = 0
        print(f"[GRABANDO] Gesto: '{gesture}' -> {video_filename}...")

        while frames_captured < self.target_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            out.write(frame)

            # Feedback visual de grabación en vivo
            annotated = frame.copy()
            cv2.circle(annotated, (40, 40), 16, (0, 0, 255), -1)
            cv2.putText(
                annotated,
                f"REC [{frames_captured}/{self.target_frames}]",
                (70, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2,
            )
            cv2.imshow("Traductor Alpha LSC - Dynamic Collector", annotated)
            cv2.waitKey(1)
            frames_captured += 1

        out.release()
        print(f"[GUARDADO] Clip completado: {video_path} ({frames_captured} frames)")


def main() -> None:
    subject = input("Ingrese el identificador del sujeto (ej. sujeto_01): ").strip()
    if not subject:
        subject = "sujeto_01"
    collector = DynamicVideoCollector(subject_id=subject)
    collector.run()


if __name__ == "__main__":
    main()
