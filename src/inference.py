"""
inference.py — Módulo de traducción en tiempo real con TTS asíncrono y temporizador de lectura automatizada.

Pipeline de Inferencia:
    Webcam → MediaPipe Hands (KeypointExtractor) → Clasificación RandomForest
    → Buffer Debounce (0.5s) → Acumulador de Texto
    → Temporizador de Inactividad / Lectura Automática (3s) → TTS (pyttsx3 en hilo secundario)
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from typing import List, Optional, Tuple

import cv2
import joblib
import numpy as np

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None  # Manejo seguro en caso de falta de librería

from numpy.typing import NDArray

# Importar constantes y módulos del proyecto
sys.path.insert(0, ".")
from config import (
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    DEBOUNCE_FRAMES,
    MODEL_PKL_PATH,
    NUM_FEATURES,
    TTS_RATE,
)
from src.keypoint_extractor import KeypointExtractor


class AsyncTTSEngine:
    """
    Motor de síntesis de voz (TTS) asíncrono no bloqueante mediante pyttsx3 y colas de hilos.

    Garantiza que la reproducción de voz nunca congele el bucle principal de OpenCV ni reduzca los FPS.
    """

    def __init__(self, rate: int = TTS_RATE) -> None:
        """
        Inicializa el hilo secundario y la cola de síntesis de voz.

        Args:
            rate: Velocidad de habla en palabras por minuto.
        """
        self.rate: int = rate
        self._speech_queue: queue.Queue[Optional[str]] = queue.Queue()
        self._is_running: bool = True
        self._worker_thread: threading.Thread = threading.Thread(
            target=self._speech_worker,
            daemon=True,
        )
        self._worker_thread.start()

    def _speech_worker(self) -> None:
        """Bucle de trabajo ejecutado en hilo secundario."""
        if pyttsx3 is None:
            print("[WARNING] pyttsx3 no esta instalado. TTS desactivado.")
            return

        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
        except Exception as e:
            print(f"[ERROR TTS] No se pudo inicializar motor pyttsx3: {e}")
            return

        while self._is_running:
            try:
                text: Optional[str] = self._speech_queue.get(timeout=0.2)
                if text is None:
                    break

                # Reproducir voz de forma síncrona dentro del hilo secundario
                engine.say(text)
                engine.runAndWait()
                self._speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR TTS Worker] {e}")

    def speak(self, text: str) -> None:
        """
        Encola un texto para ser reproducido en voz alta sin bloquear.

        Args:
            text: Frase o palabra a pronunciar.
        """
        clean_text = text.strip()
        if clean_text and pyttsx3 is not None:
            self._speech_queue.put(clean_text)

    def stop(self) -> None:
        """Detiene el hilo worker de TTS limpia y de forma segura."""
        self._is_running = False
        self._speech_queue.put(None)


class SignInferenceEngine:
    """
    Motor de inferencia en tiempo real para clasificación de keypoints 3D.
    """

    def __init__(self, model_path: str = MODEL_PKL_PATH) -> None:
        """
        Inicializa el motor cargando el modelo binario de scikit-learn.

        Args:
            model_path: Ruta al archivo .pkl.
        """
        self.model_path: str = model_path
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"[ERROR CRÍTICO] No se encontró el modelo en '{self.model_path}'.\n"
                f"Ejecute primero 'python -m src.data_collector' y luego "
                f"'python -m src.model_trainer' antes de iniciar inferencia."
            )

        print(f"[INFO] Cargando modelo clasificador desde '{self.model_path}'...")
        self.model = joblib.load(self.model_path)
        self.classes_: List[str] = list(self.model.classes_)
        print(f"[OK] Modelo cargado con éxito ({len(self.classes_)} clases): {self.classes_}")

    def predict(self, keypoints: NDArray[np.float64]) -> Tuple[str, float]:
        """
        Predice la letra y su nivel de confianza.

        Args:
            keypoints: Vector normalizado de 63 floats.

        Returns:
            Tupla (letra_predicha, confianza [0.0 a 1.0]).
        """
        if keypoints.shape != (NUM_FEATURES,):
            return "?", 0.0

        X_sample: NDArray[np.float64] = keypoints.reshape(1, -1)
        pred_label: str = str(self.model.predict(X_sample)[0])

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_sample)[0]
            confidence: float = float(np.max(probs))
        else:
            confidence = 1.0

        return pred_label, confidence


class DebounceBuffer:
    """
    Buffer de confirmación por estabilidad de N frames consecutivos.
    """

    def __init__(self, required_frames: int = DEBOUNCE_FRAMES) -> None:
        self.required_frames: int = required_frames
        self.current_candidate: Optional[str] = None
        self.consecutive_count: int = 0
        self.last_confirmed_letter: Optional[str] = None

    def update(self, detected_label: Optional[str]) -> Tuple[Optional[str], float]:
        """
        Actualiza el buffer con el fotograma actual.

        Returns:
            Tupla (letra_confirmada, progreso_debounce [0.0 a 1.0]).
        """
        if detected_label is None:
            self.current_candidate = None
            self.consecutive_count = 0
            self.last_confirmed_letter = None
            return None, 0.0

        if detected_label == self.current_candidate:
            self.consecutive_count += 1
        else:
            self.current_candidate = detected_label
            self.consecutive_count = 1

        progress: float = min(1.0, self.consecutive_count / max(1, self.required_frames))

        if self.consecutive_count >= self.required_frames:
            if detected_label != self.last_confirmed_letter:
                self.last_confirmed_letter = detected_label
                return detected_label, 1.0

        return None, progress


def draw_inference_hud(
    frame: NDArray[np.uint8],
    current_label: str,
    confidence: float,
    debounce_progress: float,
    auto_tts_progress: float,
    accumulated_text: str,
    fps: float,
) -> NDArray[np.uint8]:
    """
    Renderiza la interfaz HUD completa con barras de debounce y temporizador TTS de 3 segundos.

    Args:
        frame: Fotograma BGR de la cámara.
        current_label: Letra detectada actualmente.
        confidence: Confianza del modelo (0.0 a 1.0).
        debounce_progress: Progreso de fijación de letra (0.0 a 1.0).
        auto_tts_progress: Progreso del temporizador de lectura automática (0.0 a 1.0).
        accumulated_text: Texto acumulado en pantalla.
        fps: Velocidad de procesamiento en cuadros por segundo.

    Returns:
        Fotograma con el HUD renderizado.
    """
    annotated: NDArray[np.uint8] = frame.copy()
    h, w, _ = annotated.shape

    # 1. Top Banner (Panel Superior Ampliado)
    header_h: int = 135
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), (18, 18, 24), -1)
    cv2.addWeighted(overlay, 0.82, annotated, 0.18, 0, annotated)

    # 2. Información de Predicción
    if current_label != "?":
        label_str = f"SEÑA: [{current_label}]"
        conf_str = f"Confianza: {confidence * 100:.1f}%"
        color_conf = (0, 255, 0) if confidence > 0.8 else (0, 215, 255)
    else:
        label_str = "SEÑA: ---"
        conf_str = "Confianza: ---"
        color_conf = (150, 150, 150)

    cv2.putText(
        annotated,
        label_str,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        annotated,
        conf_str,
        (220, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color_conf,
        2,
    )
    cv2.putText(
        annotated,
        f"FPS: {fps:.1f}",
        (w - 110, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        1,
    )

    # 3. Barra de Progreso 1: Debounce (Estabilidad de Letra)
    d_bar_x, d_bar_y, d_bar_w, d_bar_h = 15, 45, 180, 12
    cv2.rectangle(annotated, (d_bar_x, d_bar_y), (d_bar_x + d_bar_w, d_bar_y + d_bar_h), (50, 50, 50), -1)

    d_fill = int(d_bar_w * debounce_progress)
    d_color = (0, 255, 0) if debounce_progress >= 1.0 else (0, 165, 255)
    if d_fill > 0:
        cv2.rectangle(annotated, (d_bar_x, d_bar_y), (d_bar_x + d_fill, d_bar_y + d_bar_h), d_color, -1)
    cv2.rectangle(annotated, (d_bar_x, d_bar_y), (d_bar_x + d_bar_w, d_bar_y + d_bar_h), (200, 200, 200), 1)

    cv2.putText(
        annotated,
        "Estabilidad",
        (d_bar_x + d_bar_w + 8, d_bar_y + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (200, 200, 200),
        1,
    )

    # 4. Barra de Progreso 2: Temporizador TTS Automático (3s)
    t_bar_x, t_bar_y, t_bar_w, t_bar_h = 320, 45, 180, 12
    cv2.rectangle(annotated, (t_bar_x, t_bar_y), (t_bar_x + t_bar_w, t_bar_y + t_bar_h), (50, 50, 50), -1)

    t_fill = int(t_bar_w * auto_tts_progress)

    # Transición dinámica de color para el temporizador de 3s (Verde -> Amarillo -> Rojo)
    if auto_tts_progress < 0.5:
        t_color = (0, 255, 0)      # Verde
    elif auto_tts_progress < 0.85:
        t_color = (0, 215, 255)    # Amarillo
    else:
        t_color = (0, 0, 255)      # Rojo alertador

    if t_fill > 0:
        cv2.rectangle(annotated, (t_bar_x, t_bar_y), (t_bar_x + t_fill, t_bar_y + t_bar_h), t_color, -1)
    cv2.rectangle(annotated, (t_bar_x, t_bar_y), (t_bar_x + t_bar_w, t_bar_y + t_bar_h), (200, 200, 200), 1)

    cv2.putText(
        annotated,
        "Lector TTS (3s)",
        (t_bar_x + t_bar_w + 8, t_bar_y + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (200, 200, 200),
        1,
    )

    # 5. Texto Acumulado en Pantalla
    text_disp = f"TEXTO: {accumulated_text}" if accumulated_text else "TEXTO: (esperando señas...)"
    cv2.putText(
        annotated,
        text_disp,
        (15, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
    )

    # 6. Panel Inferior (Controles)
    footer_h: int = 40
    overlay_foot = annotated.copy()
    cv2.rectangle(overlay_foot, (0, h - footer_h), (w, h), (18, 18, 24), -1)
    cv2.addWeighted(overlay_foot, 0.82, annotated, 0.18, 0, annotated)

    cv2.putText(
        annotated,
        "[ESPACIO]: Espacio | [BACKSPACE/C]: Borrar | [Q / ESC / }]: Salir",
        (15, h - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 220, 220),
        1,
    )

    return annotated


def main() -> None:
    """Función principal de inferencia en tiempo real con TTS automático."""
    print("=" * 70)
    print(" TRADUCCIÓN EN TIEMPO REAL CON VOZ (TTS) — Traductor Alpha (Fase 4)")
    print(" Lector de voz automático activo (3 segundos de inactividad/mismo gesto).")
    print(" Presione [ESPACIO] para agregar espacio entre palabras.")
    print(" Presione [BACKSPACE] o [C] para borrar caracteres.")
    print(" Presione [Q], [ESC] o [}] para salir.")
    print("=" * 70)

    # Inicializar módulos
    extractor = KeypointExtractor()
    engine = SignInferenceEngine()
    debounce = DebounceBuffer(required_frames=15)
    tts_engine = AsyncTTSEngine(rate=TTS_RATE)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la cámara en el índice {CAMERA_INDEX}.")
        sys.exit(1)

    accumulated_text: str = ""
    inactivity_start_time: Optional[float] = None
    AUTO_TTS_TIMEOUT_SEC: float = 3.0

    prev_time: float = time.time()

    try:
        while True:
            ret: bool
            frame: NDArray[np.uint8]
            ret, frame = cap.read()

            if not ret:
                print("[ERROR] No se pudo capturar fotograma de la cámara.")
                break

            # Efecto espejo
            frame = cv2.flip(frame, 1)

            # Extracción de keypoints y dibujo de malla
            results = extractor.process_frame(frame)
            keypoints: Optional[NDArray[np.float64]] = extractor.extract(frame)
            annotated: NDArray[np.uint8] = extractor.draw_landmarks(frame, results)

            current_label: str = "?"
            confidence: float = 0.0

            if keypoints is not None:
                current_label, confidence = engine.predict(keypoints)
                confirmed_letter, debounce_progress = debounce.update(current_label)
            else:
                confirmed_letter, debounce_progress = debounce.update(None)

            # ── 1. Confirmación de Nueva Letra por Debounce ──
            if confirmed_letter is not None:
                accumulated_text += confirmed_letter
                print(f"\n[SEÑA CONFIRMADA] → '{confirmed_letter}' | Texto actual: '{accumulated_text}'")
                # Reiniciar temporizador de inactividad al ingresar nueva letra
                inactivity_start_time = time.time()

            # ── 2. Lógica del Temporizador de Inactividad (3 Segundos) ──
            auto_tts_progress: float = 0.0

            if accumulated_text.strip():
                if inactivity_start_time is None:
                    inactivity_start_time = time.time()

                elapsed_inactivity: float = time.time() - inactivity_start_time
                auto_tts_progress = min(1.0, elapsed_inactivity / AUTO_TTS_TIMEOUT_SEC)

                # Al llegar a los 3 segundos completos → Hablar y limpiar pantalla
                if elapsed_inactivity >= AUTO_TTS_TIMEOUT_SEC:
                    phrase_to_speak: str = accumulated_text.strip()
                    print(f"\n[TTS AUTOMÁTICO 3s] Pronunciando frase: '{phrase_to_speak}'")

                    # Reproducción asíncrona no bloqueante
                    tts_engine.speak(phrase_to_speak)

                    # Limpieza automática
                    accumulated_text = ""
                    inactivity_start_time = None
                    auto_tts_progress = 0.0
            else:
                inactivity_start_time = None
                auto_tts_progress = 0.0

            # Cálculo de FPS
            curr_time: float = time.time()
            fps: float = 1.0 / max(curr_time - prev_time, 1e-9)
            prev_time = curr_time

            # Renderizar HUD completo
            hud_frame: NDArray[np.uint8] = draw_inference_hud(
                annotated,
                current_label,
                confidence,
                debounce_progress,
                auto_tts_progress,
                accumulated_text,
                fps,
            )

            cv2.imshow("Traductor Alpha - Inferencia con TTS (Fase 4)", hud_frame)

            # Manejo de entrada por teclado
            key: int = cv2.waitKey(1) & 0xFF

            if key != 255:
                # ESPACIO -> Agregar espacio entre palabras
                if key == 32:
                    if accumulated_text and not accumulated_text.endswith(" "):
                        accumulated_text += " "
                        inactivity_start_time = time.time()
                        print("\n[TECLA ESPACIO] → Espacio agregado.")

                # BACKSPACE o 'C' -> Borrar último carácter o limpiar
                elif key in (8, ord("c"), ord("C")):
                    if accumulated_text:
                        accumulated_text = accumulated_text[:-1]
                        inactivity_start_time = time.time() if accumulated_text else None
                        print(f"\n[TECLA BORRAR] → Texto actual: '{accumulated_text}'")

                # Salir con 'Q', ESC (27) o '}' (125)
                elif key in (ord("q"), ord("Q"), 27, ord("}")):
                    print("\n[INFO] Saliendo del sistema de inferencia...")
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.release()
        tts_engine.stop()
        print(f"\n[FIN] Traducción finalizada. Último texto: '{accumulated_text}'")


if __name__ == "__main__":
    main()
