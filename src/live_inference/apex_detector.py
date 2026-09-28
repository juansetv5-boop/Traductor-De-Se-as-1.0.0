"""
apex_detector.py — Algoritmo de detección del "Apex" cinemático (pico articulatorio).
Monitorea la velocidad vectorial y aceleración de la muñeca para disparar la traducción
en el punto exacto de culminación del gesto sin depender de temporizadores ciegos de inactividad.
"""

from __future__ import annotations

import collections
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import Deque, Optional, Tuple

import numpy as np

from src.config_dynamic import (
    APEX_DECELERATION_RATIO,
    MIN_STROKE_VELOCITY,
    TRIGGER_COOLDOWN_FRAMES,
)


class ApexKinematicDetector:
    """
    Detecta el ápice cinemático de un gesto de LSC mediante análisis del perfil de velocidad.

    Estados:
        IDLE: Mano en reposo o velocidad por debajo del umbral mínimo de stroke.
        ACCELERATING: Aceleración rápida hacia el movimiento del gesto.
        STROKE_PEAK: Alcanzó la velocidad máxima del stroke.
        APEX_TRIGGER: Desaceleración súbita o parada momentánea en el ápice.
    """

    def __init__(
        self,
        min_stroke_velocity: float = MIN_STROKE_VELOCITY,
        deceleration_ratio: float = APEX_DECELERATION_RATIO,
        cooldown_frames: int = TRIGGER_COOLDOWN_FRAMES,
        history_len: int = 15,
    ) -> None:
        self.min_stroke_velocity = min_stroke_velocity
        self.deceleration_ratio = deceleration_ratio
        self.cooldown_frames = cooldown_frames
        self.history_len = history_len

        self._velocity_history: Deque[float] = collections.deque(maxlen=history_len)
        self._prev_wrist_pos: Optional[np.ndarray] = None
        self._max_velocity_observed: float = 0.0
        self._in_stroke: bool = False
        self._cooldown_counter: int = 0

    def reset(self) -> None:
        """Reinicia el estado del detector."""
        self._velocity_history.clear()
        self._prev_wrist_pos = None
        self._max_velocity_observed = 0.0
        self._in_stroke = False
        self._cooldown_counter = 0

    def update(self, wrist_coords: Optional[np.ndarray]) -> Tuple[bool, float, float]:
        """
        Actualiza el estado con la posición actual de la muñeca (3D).

        Args:
            wrist_coords: Coordenadas (x, y, z) de la muñeca activa, o None si no se detecta.

        Returns:
            Tuple (is_apex_triggered, current_velocity, peak_velocity)
        """
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            return False, 0.0, 0.0

        if wrist_coords is None:
            self.reset()
            return False, 0.0, 0.0

        current_pos = np.asarray(wrist_coords, dtype=np.float32)

        if self._prev_wrist_pos is None:
            self._prev_wrist_pos = current_pos
            return False, 0.0, 0.0

        # Cálculo de la magnitud euclidiana de velocidad entre frames sucesivos
        velocity = float(np.linalg.norm(current_pos - self._prev_wrist_pos))
        self._prev_wrist_pos = current_pos
        self._velocity_history.append(velocity)

        # Suavizado simple (media móvil de los últimos 3 frames)
        smoothed_velocity = float(np.mean(list(self._velocity_history)[-3:]))

        # ─── Lógica de Detección de Ápice ────────────────────────────────────
        is_apex = False

        if not self._in_stroke:
            # Detectar inicio del movimiento significativo
            if smoothed_velocity >= self.min_stroke_velocity:
                self._in_stroke = True
                self._max_velocity_observed = smoothed_velocity
        else:
            # Estamos dentro de la trayectoria del gesto
            if smoothed_velocity > self._max_velocity_observed:
                self._max_velocity_observed = smoothed_velocity
            else:
                # Comprobar si la velocidad ha descendido por debajo del umbral de frenada (ápice)
                drop_threshold = self._max_velocity_observed * self.deceleration_ratio
                if (
                    smoothed_velocity <= drop_threshold
                    and self._max_velocity_observed >= self.min_stroke_velocity
                ):
                    # ¡ÁPICE DETECTADO! (El usuario culminó la trayectoria)
                    is_apex = True
                    self._cooldown_counter = self.cooldown_frames
                    self._in_stroke = False
                    self._max_velocity_observed = 0.0

        return is_apex, smoothed_velocity, self._max_velocity_observed
