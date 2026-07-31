"""
inference.py — Módulo de traducción en tiempo real con TTS asíncrono robusto,
temporizador de lectura de 3s y Recaptura/Reentrenamiento Rápido en vivo.

Funcionalidades:
    1. Inferencia continua con MediaPipe Hands (21 keypoints 3D normalizados).
    2. Invariancia Diestra/Zurda: Inversión automática del eje X para manos izquierdas.
    3. Motor TTS Asíncrono (`AsyncTTSEngine`) con inicialización de COM (Windows SAPI5),
       selección forzada de voces en español y logs de diagnóstico completos.
    4. Temporizador de 3 segundos para lectura automática e higiene de pantalla.
    5. Atajo de Recaptura Rápida (`¿` / `R` / `F2`): Pausa inferencia, permite seleccionar letra,
       ofrece doble confirmación [S] SOBRESCRIBIR / [A] ACUMULAR, graba muestras en ráfaga,
       ejecuta el reentrenamiento automático y recarga el modelo en memoria en vivo sin reiniciar OpenCV.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np
from numpy.typing import NDArray

# Intentar importar pythoncom para Windows SAPI5 COM initialization
try:
    import pythoncom
except ImportError:
    pythoncom = None

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

# Importar constantes y módulos del proyecto
sys.path.insert(0, ".")
from config import (
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    DEBOUNCE_FRAMES,
    MODEL_PKL_PATH,
    NUM_FEATURES,
    STATIC_ALPHABET,
    TTS_RATE,
)
from src.data_collector import DataCollector
from src.keypoint_extractor import KeypointExtractor
from src.model_trainer import ModelTrainer


class AsyncTTSEngine:
    """
    Motor de síntesis de voz (TTS) asíncrono no bloqueante con soporte robusto
    para Windows SAPI5 / pyttsx3 y selección forzada de voz en español.
    """

    def __init__(self, rate: int = TTS_RATE) -> None:
        """
        Inicializa la cola de tareas y arranca el hilo worker dedicado.

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
        """
        Bucle worker ejecutado en hilo secundario.
        Inicializa COM en Windows y configura la voz en español.
        """
        if pyttsx3 is None:
            print("[TTS ALERTA] pyttsx3 no esta instalado. TTS desactivado.")
            return

        # Inicialización de COM en Windows para evitar silencios / bloqueos SAPI5
        if pythoncom is not None:
            try:
                pythoncom.CoInitialize()
                print("[TTS DIAGNÓSTICO] COM CoInitialize() ejecutado correctamente.")
            except Exception as e:
                print(f"[TTS AVISO] CoInitialize() aviso: {e}")

        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)

            # Buscar y seleccionar explícitamente una voz en español
            voices = engine.getProperty("voices")
            spanish_voice = None

            print(f"[TTS DIAGNÓSTICO] Voces encontradas en el sistema: {len(voices)}")

            for voice in voices:
                v_name: str = voice.name.lower()
                v_id: str = voice.id.lower()
                v_lang = [str(l).lower() for l in getattr(voice, "languages", [])]

                if any(
                    kw in v_name or kw in v_id or any(kw in l for l in v_lang)
                    for kw in ["spanish", "es_es", "es_mx", "es_us", "sabina", "helena", "laura", "pablo", "raul", "hilda", "monica", "paulina"]
                ):
                    spanish_voice = voice
                    break

            if spanish_voice is not None:
                engine.setProperty("voice", spanish_voice.id)
                print(f"[TTS ÉXITO] Voz en español configurada: '{spanish_voice.name}' (ID: {spanish_voice.id})")
            else:
                current_voice = engine.getProperty("voice")
                print(f"[TTS AVISO] No se detectó voz explícita en español. Usando voz por defecto: '{current_voice}'")

        except Exception as e:
            print(f"[ERROR CRÍTICO TTS] Fallo al inicializar pyttsx3: {e}")
            return

        print("[TTS INFO] Motor de voz secundario listo y a la escucha.")

        while self._is_running:
            try:
                text: Optional[str] = self._speech_queue.get(timeout=0.2)
                if text is None:
                    break

                clean_text: str = text.strip()
                if clean_text:
                    print(f"\n[TTS REPRODUCIENDO] 🔊 '{clean_text}'")
                    engine.say(clean_text)
                    engine.runAndWait()
                    print(f"[TTS COMPLETADO] Pronunciación de '{clean_text}' finalizada.")

                self._speech_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR TTS Worker] Excepción durante reproducción: {e}")

        if pythoncom is not None:
            try:
                pythoncom.CoUninitialize()
            except Exception:
                pass

    def speak(self, text: str) -> None:
        """Encola un texto para reproducción en voz alta sin bloquear la UI."""
        clean_text = text.strip()
        if clean_text and pyttsx3 is not None:
            self._speech_queue.put(clean_text)

    def stop(self) -> None:
        """Detiene el hilo worker de TTS."""
        self._is_running = False
        self._speech_queue.put(None)


class SignInferenceEngine:
    """
    Motor de inferencia en tiempo real para clasificación de keypoints 3D.
    """

    def __init__(self, model_path: str = MODEL_PKL_PATH) -> None:
        self.model_path: str = model_path
        self.model: Optional[object] = None
        self.classes_: List[str] = []
        self.reload_model()

    def reload_model(self) -> bool:
        """
        Carga o recarga el modelo desde disco en memoria viva.

        Returns:
            True si se cargó correctamente, False si el archivo no existe.
        """
        if not os.path.exists(self.model_path):
            print(f"[Aviso] No existe el modelo en '{self.model_path}'.")
            self.model = None
            self.classes_ = []
            return False

        try:
            print(f"[INFO] Cargando modelo clasificador desde '{self.model_path}'...")
            self.model = joblib.load(self.model_path)
            self.classes_ = list(self.model.classes_)
            print(f"[OK] Modelo cargado con éxito ({len(self.classes_)} clases): {self.classes_}")
            return True
        except Exception as e:
            print(f"[ERROR] Error al recargar modelo: {e}")
            return False

    def predict(self, keypoints: NDArray[np.float64]) -> Tuple[str, float]:
        """
        Predice la letra y su nivel de confianza.

        Args:
            keypoints: Vector normalizado de 63 floats.

        Returns:
            Tupla (letra_predicha, confianza [0.0 a 1.0]).
        """
        if self.model is None or keypoints.shape != (NUM_FEATURES,):
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
        Actualiza el buffer con la lectura del frame actual.

        Returns:
            Tupla (letra_confirmada, progreso_debounce [0.0 a 1.0]).
        """
        if detected_label is None or detected_label == "?":
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
    notification_msg: str = "",
) -> NDArray[np.uint8]:
    """
    Renderiza el HUD completo de inferencia en tiempo real.
    """
    annotated: NDArray[np.uint8] = frame.copy()
    h, w, _ = annotated.shape

    # 1. Top Banner
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

    cv2.putText(annotated, label_str, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(annotated, conf_str, (220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_conf, 2)
    cv2.putText(annotated, f"FPS: {fps:.1f}", (w - 110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

    # 3. Barra Debounce
    d_bar_x, d_bar_y, d_bar_w, d_bar_h = 15, 45, 180, 12
    cv2.rectangle(annotated, (d_bar_x, d_bar_y), (d_bar_x + d_bar_w, d_bar_y + d_bar_h), (50, 50, 50), -1)

    d_fill = int(d_bar_w * debounce_progress)
    d_color = (0, 255, 0) if debounce_progress >= 1.0 else (0, 165, 255)
    if d_fill > 0:
        cv2.rectangle(annotated, (d_bar_x, d_bar_y), (d_bar_x + d_fill, d_bar_y + d_bar_h), d_color, -1)
    cv2.rectangle(annotated, (d_bar_x, d_bar_y), (d_bar_x + d_bar_w, d_bar_y + d_bar_h), (200, 200, 200), 1)
    cv2.putText(annotated, "Estabilidad", (d_bar_x + d_bar_w + 8, d_bar_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    # 4. Barra TTS 3s
    t_bar_x, t_bar_y, t_bar_w, t_bar_h = 320, 45, 180, 12
    cv2.rectangle(annotated, (t_bar_x, t_bar_y), (t_bar_x + t_bar_w, t_bar_y + t_bar_h), (50, 50, 50), -1)

    t_fill = int(t_bar_w * auto_tts_progress)
    if auto_tts_progress < 0.5:
        t_color = (0, 255, 0)
    elif auto_tts_progress < 0.85:
        t_color = (0, 215, 255)
    else:
        t_color = (0, 0, 255)

    if t_fill > 0:
        cv2.rectangle(annotated, (t_bar_x, t_bar_y), (t_bar_x + t_fill, t_bar_y + t_bar_h), t_color, -1)
    cv2.rectangle(annotated, (t_bar_x, t_bar_y), (t_bar_x + t_bar_w, t_bar_y + t_bar_h), (200, 200, 200), 1)
    cv2.putText(annotated, "Lector TTS (3s)", (t_bar_x + t_bar_w + 8, t_bar_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    # 5. Texto Acumulado
    text_disp = f"TEXTO: {accumulated_text}" if accumulated_text else "TEXTO: (esperando señas...)"
    cv2.putText(annotated, text_disp, (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    # 6. Notificación temporal (si aplica)
    if notification_msg:
        cv2.rectangle(annotated, (0, h - 75), (w, h - 40), (0, 120, 0), -1)
        cv2.putText(annotated, notification_msg, (20, h - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 7. Panel Inferior
    footer_h: int = 40
    overlay_foot = annotated.copy()
    cv2.rectangle(overlay_foot, (0, h - footer_h), (w, h), (18, 18, 24), -1)
    cv2.addWeighted(overlay_foot, 0.82, annotated, 0.18, 0, annotated)

    cv2.putText(
        annotated,
        "[ESPACIO]: Espacio | [BS/C]: Borrar | [¿ / R]: Recapturar Letra | [Q/ESC]: Salir",
        (15, h - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (220, 220, 220),
        1,
    )

    return annotated


def draw_recapture_modal(
    frame: NDArray[np.uint8],
    step: str,  # "SELECT_LETTER", "CONFIRM_MODE", "CAPTURING"
    target_letter: str,
    captured_count: int,
    target_count: int,
    existing_count: int,
    has_hand: bool,
) -> NDArray[np.uint8]:
    """
    Renderiza el diálogo emergente para la función de Recaptura y Reentrenamiento Rápido.
    """
    annotated: NDArray[np.uint8] = frame.copy()
    h, w, _ = annotated.shape

    mw, mh = 580, 220
    mx = (w - mw) // 2
    my = (h - mh) // 2

    # Fondo oscuro emergente
    cv2.rectangle(annotated, (mx, my), (mx + mw, my + mh), (25, 25, 35), -1)
    cv2.rectangle(annotated, (mx, my), (mx + mw, my + mh), (0, 215, 255), 2)

    if step == "SELECT_LETTER":
        cv2.putText(annotated, "RECAPTURA Y REENTRENAMIENTO RÁPIDO", (mx + 20, my + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2)
        cv2.putText(annotated, f"Seleccione la letra a re-grabar: [ {target_letter} ]", (mx + 20, my + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated, "Usa las flechas [< >] o escribe la letra A-Z.", (mx + 20, my + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
        cv2.putText(annotated, "Presione [ENTER / ESPACIO] para Confirmar | [ESC] para Cancelar", (mx + 20, my + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)

    elif step == "CONFIRM_MODE":
        cv2.putText(annotated, f"¡LA LETRA '{target_letter}' TIENE {existing_count} MUESTRAS!", (mx + 20, my + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
        cv2.putText(annotated, "¿Cómo desea proceder con los datos existentes?", (mx + 20, my + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.putText(annotated, "[S] SOBRESCRIBIR : Borra las muestras anteriores de esta letra.", (mx + 30, my + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 2)
        cv2.putText(annotated, "[A] ACUMULAR    : Conserva y añade nuevas muestras al CSV.", (mx + 30, my + 145), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)
        cv2.putText(annotated, "Presione [S], [A] o [ESC] para Cancelar", (mx + 20, my + 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    elif step == "CAPTURING":
        cv2.putText(annotated, f"GRABANDO MUESTRAS PARA LETRA '{target_letter}'", (mx + 20, my + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        pct = min(1.0, captured_count / max(1, target_count))
        cv2.putText(annotated, f"Muestras: {captured_count} / {target_count} ({int(pct * 100)}%)", (mx + 20, my + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Barra de progreso
        bx, by, bw, bh = mx + 20, my + 105, mw - 40, 20
        cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (60, 60, 60), -1)
        fw = int(bw * pct)
        if fw > 0:
            cv2.rectangle(annotated, (bx, by), (bx + fw, by + bh), (0, 255, 0), -1)
        cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (200, 200, 200), 1)

        if has_hand:
            cv2.putText(annotated, "● GRABANDO (Inversión automática para mano izquierda activa)", (mx + 20, my + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1)
        else:
            cv2.putText(annotated, "¡COLOQUE LA MANO EN PANTALLA PARA GRABAR!", (mx + 20, my + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 2)

        cv2.putText(annotated, "Manten presionada la mano o ESPACIO | [ESC] Cancelar", (mx + 20, my + 195), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

    return annotated


def run_quick_retrain(
    collector: DataCollector,
    target_letter: str,
    mode: str,  # "OVERWRITE" or "ACCUMULATE"
    captured_samples: List[NDArray[np.float64]],
    engine: SignInferenceEngine,
) -> str:
    """
    Ejecuta la limpieza/añadido de muestras, desencadena el reentrenamiento
    y recarga el modelo en vivo.

    Returns:
        Mensaje de estado para mostrar en la notificación del HUD.
    """
    if mode == "OVERWRITE":
        collector.clear_class(target_letter)
        print(f"\n[RECAPTURA] Muestras antiguas de '{target_letter}' eliminadas del CSV.")

    for kp in captured_samples:
        collector.append_sample(kp, target_letter)

    print(f"[RECAPTURA] Guardadas {len(captured_samples)} nuevas muestras para '{target_letter}'.")
    print("\n[REENTRENAMIENTO] Iniciando reentrenamiento rápido del clasificador...")

    trainer = ModelTrainer()

    try:
        X, y = trainer.load_dataset()
        present_classes = trainer.validate_alphabet_coverage(y)
        clf, accuracy, report_dict = trainer.train_and_evaluate(X, y)
        trainer.save_artifacts(clf, accuracy, present_classes, len(y), report_dict)

        # Recargar modelo en vivo en memoria
        success = engine.reload_model()

        if success:
            msg = f"¡Modelo Reentrenado y Recargado! (Accuracy: {accuracy * 100:.1f}%)"
            print(f"\n[ÉXITO REENTRENAMIENTO] {msg}")
            return msg
        else:
            return "Reentrenado, pero falló la recarga en memoria."

    except Exception as e:
        print(f"[ERROR REENTRENAMIENTO] {e}")
        return f"Error en reentrenamiento: {e}"


def main() -> None:
    """Función principal del sistema de inferencia."""
    print("=" * 75)
    print(" TRADUCTOR ALPHA — INFERENCIA EN TIEMPO REAL Y RE-ENTRENAMIENTO RÁPIDO")
    print(" Lector TTS en español activo (3s inactividad).")
    print(" Atajo [¿] o [R]: Abrir Recaptura Rápida de Letra y Reentrenamiento.")
    print(" Presione [ESPACIO] para espacios | [BS/C] para borrar | [Q/ESC] para salir.")
    print("=" * 75)

    extractor = KeypointExtractor()
    engine = SignInferenceEngine()
    collector = DataCollector()
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

    # Estado del Modo de Recaptura Rápida
    recapturing: bool = False
    recapt_step: str = "SELECT_LETTER"  # "SELECT_LETTER", "CONFIRM_MODE", "CAPTURING"
    recapt_letter_idx: int = 0
    recapt_mode: str = "ACCUMULATE"  # "OVERWRITE" or "ACCUMULATE"
    recapt_samples: List[NDArray[np.float64]] = []
    RECAPT_TARGET_COUNT: int = 150

    # Notificaciones temporales del HUD
    notification_msg: str = ""
    notification_time: float = 0.0

    prev_time: float = time.time()

    try:
        while True:
            ret: bool
            frame: NDArray[np.uint8]
            ret, frame = cap.read()

            if not ret:
                print("[ERROR] No se pudo capturar fotograma de la cámara.")
                break

            # Espejo horizontal
            frame = cv2.flip(frame, 1)

            # Extracción de keypoints (con invarianza zurda/diestra automática)
            results = extractor.process_frame(frame)
            keypoints: Optional[NDArray[np.float64]] = extractor.extract(frame, auto_flip_left_hand=True)
            annotated: NDArray[np.uint8] = extractor.draw_landmarks(frame, results)

            has_hand: bool = keypoints is not None
            key: int = cv2.waitKey(1) & 0xFF

            # Limpiar notificación tras 4 segundos
            if notification_msg and (time.time() - notification_time > 4.0):
                notification_msg = ""

            # ──────────────────────────────────────────────────────────────────
            # MODAL DE RECAPTURA RÁPIDA (Pausa inferencia)
            # ──────────────────────────────────────────────────────────────────
            if recapturing:
                target_letter: str = STATIC_ALPHABET[recapt_letter_idx]
                existing_count: int = collector.counts.get(target_letter, 0)

                if key != 255:
                    key_char: str = chr(key).upper()

                    # Cancelar modal con ESC (27)
                    if key == 27:
                        recapturing = False
                        print("\n[RECAPTURA] Operación cancelada.")

                    elif recapt_step == "SELECT_LETTER":
                        if key_char in STATIC_ALPHABET:
                            recapt_letter_idx = STATIC_ALPHABET.index(key_char)
                        elif key in (81, 2, ord(",")):  # Flecha Izq
                            recapt_letter_idx = (recapt_letter_idx - 1) % len(STATIC_ALPHABET)
                        elif key in (83, 3, ord(".")):  # Flecha Der
                            recapt_letter_idx = (recapt_letter_idx + 1) % len(STATIC_ALPHABET)
                        elif key in (13, 32):  # ENTER o ESPACIO para confirmar letra
                            if existing_count > 0:
                                recapt_step = "CONFIRM_MODE"
                            else:
                                recapt_mode = "ACCUMULATE"
                                recapt_samples = []
                                recapt_step = "CAPTURING"

                    elif recapt_step == "CONFIRM_MODE":
                        if key_char in ("S", "O"):  # S = Sobrescribir
                            recapt_mode = "OVERWRITE"
                            recapt_samples = []
                            recapt_step = "CAPTURING"
                        elif key_char in ("A", "C"):  # A = Acumular
                            recapt_mode = "ACCUMULATE"
                            recapt_samples = []
                            recapt_step = "CAPTURING"

                    elif recapt_step == "CAPTURING":
                        if key == 27:  # ESC -> Cancelar captura
                            recapturing = False

                # Si está en paso CAPTURING y hay mano en pantalla
                if recapt_step == "CAPTURING":
                    if has_hand:
                        recapt_samples.append(keypoints)

                    if len(recapt_samples) >= RECAPT_TARGET_COUNT:
                        # Ejecutar reentrenamiento y recarga en memoria
                        notification_msg = run_quick_retrain(
                            collector,
                            target_letter,
                            recapt_mode,
                            recapt_samples,
                            engine,
                        )
                        notification_time = time.time()
                        recapturing = False

                hud_frame = draw_recapture_modal(
                    annotated,
                    recapt_step,
                    target_letter,
                    len(recapt_samples),
                    RECAPT_TARGET_COUNT,
                    existing_count,
                    has_hand,
                )
                cv2.imshow("Traductor Alpha - Inferencia con TTS (Fase 4)", hud_frame)
                continue

            # ──────────────────────────────────────────────────────────────────
            # MODO INFERENCIA NORMAL
            # ──────────────────────────────────────────────────────────────────
            current_label: str = "?"
            confidence: float = 0.0

            if keypoints is not None:
                current_label, confidence = engine.predict(keypoints)
                confirmed_letter, debounce_progress = debounce.update(current_label)
            else:
                confirmed_letter, debounce_progress = debounce.update(None)

            # Confirmación de nueva letra por Debounce
            if confirmed_letter is not None:
                accumulated_text += confirmed_letter
                print(f"\n[SEÑA CONFIRMADA] → '{confirmed_letter}' | Texto actual: '{accumulated_text}'")
                inactivity_start_time = time.time()

            # Temporizador de Inactividad de 3 Segundos
            auto_tts_progress: float = 0.0

            if accumulated_text.strip():
                if inactivity_start_time is None:
                    inactivity_start_time = time.time()

                elapsed_inactivity: float = time.time() - inactivity_start_time
                auto_tts_progress = min(1.0, elapsed_inactivity / AUTO_TTS_TIMEOUT_SEC)

                if elapsed_inactivity >= AUTO_TTS_TIMEOUT_SEC:
                    phrase_to_speak: str = accumulated_text.strip()
                    print(f"\n[TTS AUTOMÁTICO 3s] Disparando lectura por voz: '{phrase_to_speak}'")
                    tts_engine.speak(phrase_to_speak)

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
            hud_frame = draw_inference_hud(
                annotated,
                current_label,
                confidence,
                debounce_progress,
                auto_tts_progress,
                accumulated_text,
                fps,
                notification_msg,
            )

            cv2.imshow("Traductor Alpha - Inferencia con TTS (Fase 4)", hud_frame)

            # Captura de Teclado en Inferencia
            if key != 255:
                # 1. Atajo de Recaptura Rápida: Tecla '¿' (ASCII 168), 'R' (82/114) o F2
                if key in (168, 191, ord("r"), ord("R"), ord("¿")):
                    print("\n[ATAJO RECAPTURA] Abriendo diálogo de Recaptura Rápida...")
                    recapturing = True
                    recapt_step = "SELECT_LETTER"
                    recapt_samples = []

                # 2. ESPACIO -> Agregar espacio
                elif key == 32:
                    if accumulated_text and not accumulated_text.endswith(" "):
                        accumulated_text += " "
                        inactivity_start_time = time.time()
                        print("\n[TECLA ESPACIO] → Espacio agregado.")

                # 3. BACKSPACE o 'C' -> Borrar último carácter
                elif key in (8, ord("c"), ord("C")):
                    if accumulated_text:
                        accumulated_text = accumulated_text[:-1]
                        inactivity_start_time = time.time() if accumulated_text else None
                        print(f"\n[TECLA BORRAR] → Texto actual: '{accumulated_text}'")

                # 4. Salir con 'Q', ESC (27) o '}' (125)
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
