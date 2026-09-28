"""
feature_extractor.py — Extracción cinemática y normalización matemática de Landmarks.
Combina MediaPipe Holistic (Pose + Hands) con normalización anatómica y derivadas temporales (dX, d^2X).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from numpy.typing import NDArray

from src.config_dynamic import (
    INPUT_FEATURES_PER_FRAME,
    NUM_COORDS,
    NUM_HAND_LANDMARKS,
    POSE_LANDMARK_INDICES,
    RAW_SPATIAL_FEATURES,
    TOTAL_POINTS,
)


class HolisticFeatureExtractor:
    """
    Extrae, normaliza y calcula derivadas temporales sobre keypoints de MediaPipe Holistic.
    Garantiza invarianza espacial (traslación y escala) y modelado cinemático (velocidad y aceleración).
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._mp_holistic = mp.solutions.holistic
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles

        self._holistic = self._mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_frame(self, frame_bgr: NDArray[np.uint8]) -> object:
        """Convierte frame a RGB y procesa con MediaPipe Holistic."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._holistic.process(frame_rgb)
        frame_rgb.flags.writeable = True
        return results

    def extract_spatial_vector(
        self,
        results: object,
    ) -> Tuple[NDArray[np.float32], dict[str, Optional[NDArray[np.float32]]]]:
        """
        Extrae un vector espacial normalizado de 147 floats (49 puntos x 3 coords)
        y coordenadas clave para el cálculo del Apex cinemático.

        Returns:
            Tuple con:
                - Vector espacial 1D normalizado (147,)
                - Diccionario con posiciones absolutas de muñecas {'left_wrist', 'right_wrist'}
        """
        spatial_vec = np.zeros(RAW_SPATIAL_FEATURES, dtype=np.float32)
        wrist_pos: dict[str, Optional[NDArray[np.float32]]] = {
            "left_wrist": None,
            "right_wrist": None,
        }

        # ─── 1. Procesar Pose (7 puntos del torso superior) ──────────────────────
        pose_coords = np.zeros((len(POSE_LANDMARK_INDICES), NUM_COORDS), dtype=np.float32)
        mid_shoulder = np.zeros(NUM_COORDS, dtype=np.float32)
        shoulder_dist = 1.0

        if results.pose_landmarks:
            raw_pose = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark],
                dtype=np.float32,
            )
            # Puntos seleccionados: 0, 11, 12, 13, 14, 15, 16
            for idx, p_idx in enumerate(POSE_LANDMARK_INDICES):
                if p_idx < len(raw_pose):
                    pose_coords[idx] = raw_pose[p_idx]

            # Referencia anatómica: Centro entre hombros (11 y 12)
            left_shoulder = raw_pose[11] if len(raw_pose) > 11 else np.zeros(3)
            right_shoulder = raw_pose[12] if len(raw_pose) > 12 else np.zeros(3)
            mid_shoulder = (left_shoulder + right_shoulder) / 2.0

            # Escala: Distancia entre hombros
            shoulder_dist = float(np.linalg.norm(left_shoulder - right_shoulder))
            if shoulder_dist < 1e-4:
                shoulder_dist = 1.0

            # Normalizar pose respecto a hombros
            pose_coords = (pose_coords - mid_shoulder) / shoulder_dist

            # Guardar muñecas para cálculo de velocidad en live_engine
            if len(raw_pose) > 15:
                wrist_pos["left_wrist"] = raw_pose[15]
            if len(raw_pose) > 16:
                wrist_pos["right_wrist"] = raw_pose[16]

        # ─── 2. Procesar Mano Izquierda (21 puntos) ──────────────────────────────
        left_hand_coords = np.zeros((NUM_HAND_LANDMARKS, NUM_COORDS), dtype=np.float32)
        if results.left_hand_landmarks:
            raw_left = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
                dtype=np.float32,
            )
            wrist_left = raw_left[0]
            # Normalización wrist-relative
            centered_left = raw_left - wrist_left
            hand_scale = float(np.max(np.abs(centered_left)))
            if hand_scale < 1e-4:
                hand_scale = 1.0
            left_hand_coords = centered_left / hand_scale
            wrist_pos["left_wrist"] = wrist_left
            if len(raw_left) > 8:
                wrist_pos["left_index"] = raw_left[8]

        # ─── 3. Procesar Mano Derecha (21 puntos) ────────────────────────────────
        right_hand_coords = np.zeros((NUM_HAND_LANDMARKS, NUM_COORDS), dtype=np.float32)
        if results.right_hand_landmarks:
            raw_right = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
                dtype=np.float32,
            )
            wrist_right = raw_right[0]
            # Normalización wrist-relative
            centered_right = raw_right - wrist_right
            hand_scale = float(np.max(np.abs(centered_right)))
            if hand_scale < 1e-4:
                hand_scale = 1.0
            right_hand_coords = centered_right / hand_scale
            wrist_pos["right_wrist"] = wrist_right
            if len(raw_right) > 8:
                wrist_pos["right_index"] = raw_right[8]

        # Concatenar todos los puntos espaciales
        all_points = np.vstack([pose_coords, left_hand_coords, right_hand_coords])
        spatial_vec = all_points.flatten().astype(np.float32)
        return spatial_vec, wrist_pos

    @staticmethod
    def compute_temporal_kinematics(spatial_sequence: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Dada una ventana temporal de posiciones espaciales de forma (T, RAW_SPATIAL_FEATURES),
        calcula la velocidad (primera derivada) y la aceleración (segunda derivada).

        Retorna:
            Array de forma (T, INPUT_FEATURES_PER_FRAME) donde INPUT_FEATURES_PER_FRAME = 147 * 3 = 441.
        """
        T, num_feats = spatial_sequence.shape
        assert num_feats == RAW_SPATIAL_FEATURES, f"Esperado {RAW_SPATIAL_FEATURES}, recibido {num_feats}"

        # Velocidad: dX/dt usando diferencias finitas
        velocity = np.zeros_like(spatial_sequence)
        if T > 1:
            velocity[1:] = spatial_sequence[1:] - spatial_sequence[:-1]

        # Aceleración: d^2X/dt^2
        acceleration = np.zeros_like(velocity)
        if T > 2:
            acceleration[1:] = velocity[1:] - velocity[:-1]

        # Concatenar en la dimensión de características: [P, V, A]
        kinematics_window = np.concatenate([spatial_sequence, velocity, acceleration], axis=1)
        return kinematics_window.astype(np.float32)

    def draw_landmarks(self, frame_bgr: NDArray[np.uint8], results: object) -> NDArray[np.uint8]:
        """Dibuja de forma limpia los landmarks de pose y manos en el frame."""
        annotated = frame_bgr.copy()

        # Pose (solo conexiones superiores)
        if results.pose_landmarks:
            self._mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                self._mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self._mp_styles.get_default_pose_landmarks_style(),
            )

        # Mano Izquierda
        if results.left_hand_landmarks:
            self._mp_drawing.draw_landmarks(
                annotated,
                results.left_hand_landmarks,
                self._mp_holistic.HAND_CONNECTIONS,
                self._mp_styles.get_default_hand_landmarks_style(),
                self._mp_styles.get_default_hand_connections_style(),
            )

        # Mano Derecha
        if results.right_hand_landmarks:
            self._mp_drawing.draw_landmarks(
                annotated,
                results.right_hand_landmarks,
                self._mp_holistic.HAND_CONNECTIONS,
                self._mp_styles.get_default_hand_landmarks_style(),
                self._mp_styles.get_default_hand_connections_style(),
            )

        return annotated

    def close(self) -> None:
        """Libera los recursos de MediaPipe."""
        self._holistic.close()
