"""
ghost_capture_pipeline.py — Pipeline de captura y persistencia sincronizado con Ghost Buttons.

Implementa:
    1. Buffer temporal en memoria de tamaño fijo (T = 60 frames) con collections.deque(maxlen=60).
    2. Disparador de inicio sincronizado con el Ghost Button inicial (Hold de 0.5s).
    3. Criterio de completado y commit en segundo plano (threading.Thread) sin caídas de FPS.
    4. Descarte con feedback visual (alerta roja tenue) en caso de abandono o timeout.
    5. Persistencia sincronizada en dos directorios:
        a) dataset/raw_videos/<gesture>/<gesture>_<timestamp>.mp4 (30 FPS)
        b) dataset/processed_tensors/<gesture>/<gesture>_<timestamp>.npz ([60, features] normalizado)
    6. Contador persistente de muestras válidas por seña para el HUD.
"""

from __future__ import annotations

import collections
import os
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config_dynamic import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    PILOT_VOCABULARY,
    RAW_SPATIAL_FEATURES,
    TARGET_FPS,
)
from src.data_collection.feature_extractor import HolisticFeatureExtractor


class GhostCapturePipeline:
    """
    Gestiona el ciclo de vida de captura, buffer circular y persistencia en segundo plano
    para los datos recolectados mediante GhostTrainer.
    """

    def __init__(
        self,
        base_dir: str = "dataset",
        target_frames: int = 60,
        target_fps: int = TARGET_FPS,
    ) -> None:
        self.base_dir = base_dir
        self.target_frames = target_frames
        self.target_fps = target_fps

        self.raw_videos_dir = os.path.join(self.base_dir, "raw_videos")
        self.processed_tensors_dir = os.path.join(self.base_dir, "processed_tensors")

        os.makedirs(self.raw_videos_dir, exist_ok=True)
        os.makedirs(self.processed_tensors_dir, exist_ok=True)

        # Buffers temporales en memoria (T = 60 frames fijos)
        self.frame_buffer: collections.deque[NDArray[np.uint8]] = collections.deque(maxlen=target_frames)
        self.spatial_buffer: collections.deque[NDArray[np.float32]] = collections.deque(maxlen=target_frames)

        # Estado del pipeline
        self.is_recording: bool = False
        self.current_gesture: str = ""
        self.current_id: str = ""
        self.record_start_time: float = 0.0

        # Feedback visual y estado
        self.red_feedback_alpha: float = 0.0
        self.status_message: str = ""
        self.status_time: float = 0.0
        self.status_is_error: bool = False

        # Contadores en caché
        self._counts: Dict[str, int] = {}

    def reset_counts(self) -> None:
        """Reinicia la caché de conteo de muestras recolectadas a cero."""
        self._counts.clear()

    def get_sample_count(self, category: str, gesture: str) -> int:
        """Retorna la cantidad persistida de muestras válidas escaneando el directorio bajo demanda."""
        gesture_dir = os.path.join(self.processed_tensors_dir, category, gesture)
        if os.path.exists(gesture_dir):
            count = len([f for f in os.listdir(gesture_dir) if f.endswith(".npz")])
            self._counts[f"{category}/{gesture}"] = count
            return count
        return 0

    def start_recording(
        self,
        gesture: str,
        category: str = "verbos",
        initial_frame: Optional[NDArray[np.uint8]] = None,
        initial_spatial: Optional[NDArray[np.float32]] = None,
    ) -> None:
        """
        Dispara el inicio de registro activo cuando la mano alcanza el Ghost Button inicial (Hold de 0.5s).
        """
        self.frame_buffer.clear()
        self.spatial_buffer.clear()

        self.current_gesture = gesture
        self.current_category = category
        self.current_id = f"{category}_{gesture}_{int(time.time() * 1000)}"
        self.record_start_time = time.time()
        self.is_recording = True

        if initial_frame is not None:
            self.frame_buffer.append(initial_frame.copy())
        if initial_spatial is not None:
            self.spatial_buffer.append(initial_spatial.copy())

        print(f"\n[GHOST PIPELINE] Inició registro activo para: '{category}/{gesture}' (ID: {self.current_id})")

    def record_frame(
        self,
        frame: NDArray[np.uint8],
        spatial_vec: NDArray[np.float32],
    ) -> None:
        """Registra un nuevo cuadro de video y su vector espacial de MediaPipe si la captura está activa."""
        if not self.is_recording:
            return

        self.frame_buffer.append(frame.copy())
        self.spatial_buffer.append(spatial_vec.copy())

    def commit(
        self,
        gesture: Optional[str] = None,
        category: Optional[str] = None,
        is_emergency: bool = False,
    ) -> bool:
        """
        Valida y dispara la rutina de guardado asíncrono en segundo plano al completar los Ghost Buttons.
        """
        if not self.is_recording or len(self.frame_buffer) == 0:
            return False

        gesture_name = gesture or self.current_gesture
        category_name = category or getattr(self, "current_category", "verbos")
        unique_id = self.current_id or f"{category_name}_{gesture_name}_{int(time.time() * 1000)}"

        # Extraer copias instantáneas de los buffers para el hilo de fondo
        frames_snapshot = list(self.frame_buffer)
        spatials_snapshot = list(self.spatial_buffer)

        # Resetear estado de captura
        self.is_recording = False
        self.frame_buffer.clear()
        self.spatial_buffer.clear()

        self.status_message = f"Guardando '{gesture_name}' ({len(frames_snapshot)} fr)..."
        self.status_time = time.time()
        self.status_is_error = False

        # Lanzar hilo secundario para no bloquear el stream de la cámara (Zero Lag)
        worker_thread = threading.Thread(
            target=self._async_save_worker,
            args=(gesture_name, category_name, unique_id, frames_snapshot, spatials_snapshot, is_emergency),
            daemon=True,
            name=f"SaveWorker-{unique_id}",
        )
        worker_thread.start()
        return True

    def discard(self, reason: str = "Abandono de trayectoria") -> None:
        """
        Descarta el buffer temporal actual y activa el feedback visual rojo tenue de reinicio.
        """
        self.is_recording = False
        self.frame_buffer.clear()
        self.spatial_buffer.clear()

        # Activar destello de reinicio (feedback visual de error/timeout)
        self.red_feedback_alpha = 0.40
        self.status_message = f"Reiniciado: {reason}"
        self.status_time = time.time()
        self.status_is_error = True
        print(f"[GHOST PIPELINE] Buffer descartado ({reason}). Feedback visual activado.")

    def _async_save_worker(
        self,
        gesture: str,
        category: str,
        unique_id: str,
        frames: List[NDArray[np.uint8]],
        spatials: List[NDArray[np.float32]],
        is_emergency: bool,
    ) -> None:
        """
        Hilo de persistencia en disco: codifica el video a 30 FPS y genera el tensor normalizado [60, features].
        """
        try:
            # ─── 1. Guardar Video Raw (MP4 a 30 FPS) ───────────────────────────
            gesture_vid_dir = os.path.join(self.raw_videos_dir, category, gesture)
            os.makedirs(gesture_vid_dir, exist_ok=True)
            video_path = os.path.join(gesture_vid_dir, f"{unique_id}.mp4")

            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(video_path, fourcc, float(self.target_fps), (w, h))

            for f in frames:
                writer.write(f)
            writer.release()

            # ─── 2. Procesar Tensor [60, features] listo para PyTorch ───────────
            gesture_tensor_dir = os.path.join(self.processed_tensors_dir, category, gesture)
            os.makedirs(gesture_tensor_dir, exist_ok=True)
            tensor_path = os.path.join(gesture_tensor_dir, f"{unique_id}.npz")

            raw_spatial_matrix = np.array(spatials, dtype=np.float32)  # (N, 147)
            num_recorded = len(raw_spatial_matrix)

            # Ajuste de tamaño exacto T = 60 frames mediante padding / resampleado
            if num_recorded < self.target_frames:
                # Padding replicando el último fotograma
                pad_len = self.target_frames - num_recorded
                padding = np.repeat(raw_spatial_matrix[-1:], pad_len, axis=0)
                spatial_60 = np.vstack([raw_spatial_matrix, padding]).astype(np.float32)
            elif num_recorded == self.target_frames:
                spatial_60 = raw_spatial_matrix
            else:
                # Si excede 60, tomar los 60 fotogramas finales de la trayectoria
                spatial_60 = raw_spatial_matrix[-self.target_frames:]

            # Calcular derivadas cinemáticas completas (Posición 147 + Velocidad 147 + Aceleración 147 = 441)
            kinematics_60 = HolisticFeatureExtractor.compute_temporal_kinematics(spatial_60)  # (60, 441)

            # Guardar archivo .npz comprimido
            np.savez_compressed(
                tensor_path,
                tensor=kinematics_60,  # Tensor [60, 441] normalizado para PyTorch
                spatial=spatial_60,    # Vector espacial [60, 147]
                gesture=gesture,
                category=category,
                unique_id=unique_id,
                frames_count=num_recorded,
                is_emergency=is_emergency,
                created_at=time.time(),
            )

            print(
                f"[GHOST PIPELINE OK] Muestra guardada exitosamente:\n"
                f"  -> Video:  {video_path} ({num_recorded} frames)\n"
                f"  -> Tensor: {tensor_path} (forma: {kinematics_60.shape})"
            )

            self.status_message = f"Guardado OK: {unique_id}"
            self.status_time = time.time()
            self.status_is_error = False

        except Exception as e:
            print(f"[ERROR GHOST PIPELINE] Error durante persistencia de {unique_id}: {e}")
            self.status_message = f"Error al guardar {unique_id}"
            self.status_time = time.time()
            self.status_is_error = True

    def update_feedback(self, delta_time: float) -> None:
        """Actualiza el desvanecimiento del feedback visual de reinicio y expiración de mensajes."""
        if self.red_feedback_alpha > 0.0:
            self.red_feedback_alpha = max(0.0, self.red_feedback_alpha - 0.70 * delta_time)

        if self.status_message and (time.time() - self.status_time > 3.5):
            self.status_message = ""

    def apply_visual_feedback(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Aplica el tinte rojo tenue si hubo un descarte de trayectoria reciente."""
        if self.red_feedback_alpha <= 0.001:
            return frame

        # Tinte rojo translúcido elegante
        overlay = frame.copy()
        overlay[:] = (0, 0, 150)
        cv2.addWeighted(overlay, self.red_feedback_alpha, frame, 1.0 - self.red_feedback_alpha, 0, frame)
        return frame
