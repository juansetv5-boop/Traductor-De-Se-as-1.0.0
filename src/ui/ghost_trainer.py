"""
ghost_trainer.py — Sistema Avanzado de Entrenamiento Visual con Trayectoria Fantasma (GhostPath).
Características:
  1. Ventana Pantalla Completa nativa sin bordes con escalado a resolución de pantalla real.
  2. Ghost Buttons Reactivos (Hitbox Dynamics): pasivo (tenue) vs colisión activa (brillo + llenado radial).
  3. FSM de Entrenamiento Secuencial: flujo guiado paso a paso por seña (Hola -> Gracias -> Por favor -> Ayuda -> Buenos días).
  4. Selector de Cursor Interactivo: Punta del índice (Landmark 8) o Centro de la muñeca (Landmark 0).
  5. Sincronización con síntesis de voz (TTS) al completar exitosamente cada seña.
"""

from __future__ import annotations

import ctypes
import math
import os
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Final, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import cv2
import numpy as np

from src.audio.async_tts import AsyncTTSEngine
from src.config_dynamic import (
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    PILOT_VOCABULARY,
    TARGET_FPS,
)
from src.core.hand_pose_validator import (
    HandPoseValidator,
    GESTURE_REQUIREMENTS as CORE_GESTURE_REQUIREMENTS,
    classify_finger_state,
    compute_palm_normal,
    is_hand_fist,
    is_walking_active_hand,
    is_walking_base_hand,
    identify_walking_hands,
    are_helping_hands_coupled,
    identify_helping_hands,
    is_helping_active_hand,
    is_helping_base_hand,
    LM_WRIST,
    LM_THUMB_MCP,
    LM_THUMB_TIP,
)
from src.data_collection.feature_extractor import HolisticFeatureExtractor
from src.data_collection.ghost_capture_pipeline import GhostCapturePipeline

# Señas que exigen encuadre de plano medio / entero con caderas visibles
FULL_BODY_REQUIRED: Final[List[str]] = ["caminar", "jugar"]

# Perfiles de intención comunicativa y requerimientos posturales (Queremas)
GESTURE_REQUIREMENTS: Final[Dict[str, Dict[str, Any]]] = {
    "hola": {
        "fingers": {"thumb": "ANY", "index": "EXTENDED", "middle": "EXTENDED", "ring": "CURLED", "pinky": "CURLED"},
        "palm_orientation": "FORWARD_OR_DIAGONAL",
    },
    "gracias": {
        "fingers": {"thumb": "EXTENDED", "index": "EXTENDED", "middle": "EXTENDED", "ring": "EXTENDED", "pinky": "EXTENDED"},
        "palm_start": "INWARD_FACE",
        "palm_end": "UPWARD_FORWARD",
    },
    "por_favor": {
        "fingers": {
            "thumb": "CURLED_OVER_FINGERS",  # o descansando sobre el índice
            "index": "CURLED",
            "middle": "CURLED",
            "ring": "CURLED",
            "pinky": "CURLED",
        },
        "hand_shape": "FIST",  # Mano cerrada en puño
        "palm_orientation": "INWARD_CHEST",  # Base del puño / nudillos contra el pecho
        "motion": "CIRCULAR_CLOCKWISE",      # Trayectoria circular sobre el esternón
        "anchor": "CHEST_STERNUM",
    },
    "caminar": {
        "active_hand": {
            "fingers": {
                "thumb": "ANY",
                "index": "EXTENDED",
                "middle": "EXTENDED",
                "ring": "CURLED",
                "pinky": "CURLED",
            },
            "orientation": "DOWNWARD",
        },
        "base_hand": {
            "fingers": {
                "thumb": "EXTENDED",
                "index": "EXTENDED",
                "middle": "EXTENDED",
                "ring": "EXTENDED",
                "pinky": "EXTENDED",
            },
            "orientation": "HORIZONTAL_UP",
        },
        "motion": "WALK_OVER_PALM",
    },
    "ayudar": {
        "active_hand": {
            "fingers": {
                "thumb": "EXTENDED_UP",
                "index": "CURLED",
                "middle": "CURLED",
                "ring": "CURLED",
                "pinky": "CURLED",
            },
            "orientation": "THUMB_UP",
        },
        "base_hand": {
            "fingers": {
                "thumb": "EXTENDED",
                "index": "EXTENDED",
                "middle": "EXTENDED",
                "ring": "EXTENDED",
                "pinky": "EXTENDED",
            },
            "orientation": "HORIZONTAL_UP",
        },
        "motion": "COUPLED_TRANSLATION_TO_CHEST",
    },
    "ayuda": {
        "alias_of": "ayudar",
    },
}


class CursorType(Enum):
    INDEX_TIP = "Índice (LM 8)"
    WRIST = "Muñeca (LM 0)"
    AUTO = "Auto (Índice o Muñeca)"


class FSMState(Enum):
    WAITING_START = "Esperando Inicio"
    IN_SEQUENCE = "En Trayectoria"
    SUCCESS = "¡Gesto Completado!"
    TIMEOUT_RESET = "Tiempo Agotado"


@dataclass
class ReactiveGhostButton:
    """Representa un Ghost Button reactivo con dinámica de colisión y progreso de llenado."""

    id: int
    x_norm: float = 0.0  # Coordenada normalizada X [0.0, 1.0]
    y_norm: float = 0.0  # Coordenada normalizada Y [0.0, 1.0]
    label: str = ""
    is_active_target: bool = False
    is_completed: bool = False
    hover_progress: float = 0.0  # Progreso de llenado [0.0, 1.0]
    pulse_radius: float = 0.0
    pulse_alpha: float = 0.0
    completion_time: float = 0.0
    px_x: int = 0  # Coordenada absoluta X en píxeles (antropométrica)
    px_y: int = 0  # Coordenada absoluta Y en píxeles (antropométrica)
    is_bimanual: bool = False  # True si forma parte de un par simultáneo en FASE 2
    pair_id: int = -1  # ID del botón compañero simultáneo (-1 si es unimanual)
    side: str = ""  # "left", "right" o ""
    require_open_palm: bool = False  # Exige dedos extendidos (LM 8, 12, 16, 20)
    require_palm_up: bool = False  # Exige rotación de palma orientada hacia arriba
    has_posture_error: bool = False  # Colisión inválida: mano en hitbox con postura incorrecta
    posture_error_msg: str = ""  # Mensaje correctivo específico para este botón


def is_hand_palm_open(hand_lms: Optional[object]) -> bool:
    """Verifica si la mano tiene dedos extendidos (LM 8, 12, 16, 20) para palma abierta."""
    if hand_lms is None or not hasattr(hand_lms, "landmark"):
        return False
    w_lm = hand_lms.landmark[0]
    extended = 0
    # Comparar distancia muñeca-punta vs muñeca-articulación PIP para índice, medio, anular y meñique
    for tip_idx, pip_idx in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        tip = hand_lms.landmark[tip_idx]
        pip = hand_lms.landmark[pip_idx]
        d_tip = math.hypot(tip.x - w_lm.x, tip.y - w_lm.y)
        d_pip = math.hypot(pip.x - w_lm.x, pip.y - w_lm.y)
        if d_tip > d_pip * 1.15:
            extended += 1
    return extended >= 3  # Al menos 3 dedos claramente extendidos


def is_hand_palm_up(hand_lms: Optional[object]) -> bool:
    """Verifica si la mano está rotada mostrando la palma orientada hacia arriba/frente."""
    if hand_lms is None or not hasattr(hand_lms, "landmark"):
        return False
    w_lm = hand_lms.landmark[0]
    mid_tip = hand_lms.landmark[12]
    # En proyección frontal/arriba, las puntas proyectan horizontal o hacia arriba respecto a la muñeca
    return (mid_tip.y <= w_lm.y + 0.18)


def check_palm_parallel_to_torso(
    hand_landmarks: Optional[object],
    pose_landmarks: Optional[object] = None,
    min_parallel_threshold: float = 0.38,
) -> Tuple[bool, float]:
    """
    Verifica que la orientación de la palma se mantenga paralela al plano del torso (pecho)
    utilizando el vector normal formado entre:
      - Muñeca (LM 0)
      - Base del índice (LM 5)
      - Base del meñique (LM 17)

    Retorna:
      Tuple[bool, float]: (is_parallel, parallelism_score) donde parallelism_score está en [0.0, 1.0].
    """
    if hand_landmarks is None:
        return True, 1.0

    lms = getattr(hand_landmarks, "landmark", None)
    if lms is None or len(lms) <= 17:
        return True, 1.0

    w = np.array([lms[0].x, lms[0].y, lms[0].z], dtype=np.float32)
    idx_mcp = np.array([lms[5].x, lms[5].y, lms[5].z], dtype=np.float32)
    pnk_mcp = np.array([lms[17].x, lms[17].y, lms[17].z], dtype=np.float32)

    v1 = idx_mcp - w
    v2 = pnk_mcp - w

    normal_palm = np.cross(v1, v2)
    norm_p = np.linalg.norm(normal_palm)
    if norm_p < 1e-7:
        return True, 1.0
    unit_palm = normal_palm / norm_p

    # Normal de referencia del torso (pecho hacia la cámara)
    unit_torso = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if pose_landmarks is not None and getattr(pose_landmarks, "landmark", None):
        plms = pose_landmarks.landmark
        if len(plms) > 12:
            ls = np.array([plms[11].x, plms[11].y, plms[11].z], dtype=np.float32)
            rs = np.array([plms[12].x, plms[12].y, plms[12].z], dtype=np.float32)
            u_shoulder = rs - ls

            if len(plms) > 24 and plms[23].visibility > 0.4 and plms[24].visibility > 0.4:
                lh = np.array([plms[23].x, plms[23].y, plms[23].z], dtype=np.float32)
                rh = np.array([plms[24].x, plms[24].y, plms[24].z], dtype=np.float32)
                mid_s = (ls + rs) / 2.0
                mid_h = (lh + rh) / 2.0
                u_spine = mid_h - mid_s
            else:
                u_spine = np.array([0.0, 1.0, 0.0], dtype=np.float32)

            normal_torso = np.cross(u_shoulder, u_spine)
            norm_t = np.linalg.norm(normal_torso)
            if norm_t > 1e-6:
                unit_torso = normal_torso / norm_t

    # Paralelismo: producto punto absoluto entre normal de la palma y normal del torso
    parallelism = float(abs(np.dot(unit_palm, unit_torso)))
    is_parallel = parallelism >= min_parallel_threshold
    return is_parallel, parallelism


def get_screen_resolution() -> Tuple[int, int]:
    """Obtiene la resolución real del monitor en Windows mediante user32 DPI-Aware."""
    try:
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        w = int(user32.GetSystemMetrics(0))
        h = int(user32.GetSystemMetrics(1))
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    return 1920, 1080


class GhostTrainer:
    """
    Orquestador del módulo visual interactivo con pantalla completa y FSM de entrenamiento.
    """

    def __init__(
        self,
        cursor_type: CursorType = CursorType.INDEX_TIP,
        initial_hold_time: float = 0.50,  # Hold de 0.5s en Ghost Button inicial
        dwell_activation_time: float = 0.28,  # Segundos requeridos para activar waypoints siguientes
        step_timeout_sec: float = 4.5,
    ) -> None:
        self.cursor_type = cursor_type
        self.initial_hold_time = initial_hold_time
        self.dwell_activation_time = dwell_activation_time
        self.step_timeout_sec = step_timeout_sec
        self.current_camera_index = CAMERA_INDEX

        # Resolución nativa de pantalla
        self.screen_w, self.screen_h = get_screen_resolution()

        # ── 1. Métrica de Escala Facial (Unidad Base) ──
        # Fallback inicial antes de detectar la cara (~150 px en pantalla completa 1080p)
        self.last_valid_face_scale: float = float(int(0.14 * self.screen_h))
        self.face_scale: float = self.last_valid_face_scale
        self.face_scale_alpha: float = 0.15  # Suavizado exponencial para evitar saltos bruscos
        self.face_detected: bool = False

        # ── 2. Dimensionamiento Dinámico de Ghost Buttons ──
        # R_visual = int(0.35 * FACE_SCALE)
        # R_hitbox = int(0.45 * FACE_SCALE) para tolerancia amplia y cómoda
        self.r_visual: int = int(0.35 * self.face_scale)
        self.r_hitbox: int = int(0.45 * self.face_scale)
        self.hit_radius_px: int = self.r_hitbox

        # Anclajes anatómicos de referencia suavizados (EMA)
        self.smoothed_nose: Tuple[float, float] = (0.50 * self.screen_w, 0.32 * self.screen_h)
        self.smoothed_ear_right: Tuple[float, float] = (0.56 * self.screen_w, 0.30 * self.screen_h)
        self.smoothed_ear_left: Tuple[float, float] = (0.44 * self.screen_w, 0.30 * self.screen_h)
        self.smoothed_mid_shoulder: Tuple[float, float] = (0.50 * self.screen_w, 0.50 * self.screen_h)

        # Vocabularios activos para el modo de entrenamiento
        self.dynamic_signs = ["ayudar" if g.lower() == "ayuda" else g for g in PILOT_VOCABULARY if g != "Neutro"]
        if "ayudar" not in [x.lower() for x in self.dynamic_signs]:
            self.dynamic_signs.append("ayudar")
        for g in FULL_BODY_REQUIRED:
            if g not in [x.lower() for x in self.dynamic_signs]:
                self.dynamic_signs.append(g)
        self.alphabet_signs = list("abcdefghijklmnñopqrstuvwxyz")
        
        self.training_mode = "VERBOS"  # "VERBOS" o "ABECEDARIO"
        self.training_vocabulary = self.dynamic_signs
        self.current_vocab_idx = 0
        self.current_gesture = self.training_vocabulary[self.current_vocab_idx]

        # Estado FSM
        self.fsm_state = FSMState.WAITING_START
        self.buttons: List[ReactiveGhostButton] = []
        self.active_button_idx = 0
        self.state_enter_time = time.time()
        self.last_interaction_time = time.time()
        self.posture_warning: bool = False
        self.bimanual_warning: bool = False
        self.active_hand_idx: int = 0

        # Validador de configuración manual postural (Queremas)
        self.hand_validator = HandPoseValidator(GESTURE_REQUIREMENTS)
        self.hand_posture_error: bool = False
        self.hand_posture_error_message: str = ""
        self.hand_posture_error_time: float = 0.0

        # Atributos específicos para la cinemática circular de "Por favor"
        self.por_favor_center: Tuple[float, float] = (0.50 * self.screen_w, 0.50 * self.screen_h)
        self.por_favor_radius: float = 0.60 * self.face_scale
        self.prev_cursor_pos: Optional[Tuple[int, int]] = None
        self.prev_cursor_angle: Optional[float] = None
        self.reverse_rotation_accum: float = 0.0
        self.palm_warning: bool = False
        self.palm_invalid_duration: float = 0.0
        self.palm_orientation_score: float = 1.0

        # Motor de voz
        self.tts = AsyncTTSEngine()

        # Pipeline de captura automática y persistencia sincronizada
        self.capture_pipeline = GhostCapturePipeline()
        self.target_samples_per_gesture: int = 30
        self.capture_pipeline.reset_counts()

        # Inicializar botones del primer gesto
        self._load_gesture_buttons(self.current_gesture)

    def reset_sample_counters(self) -> None:
        """Reinicia los contadores de muestras para que comiencen en 0/30 para todas las señas."""
        self.capture_pipeline.reset_counts()

    def update_anthropometric_calibration(self, results: object) -> None:
        """
        Calcula la escala antropométrica facial (FACE_SCALE) usando MediaPipe Pose,
        actualiza los radios dinámicos (R_visual = 0.35 * FS, R_hitbox = 0.45 * FS)
        y recalcula las coordenadas anatómicas de los Ghost Buttons.
        """
        measured_scale: Optional[float] = None

        if results and hasattr(results, "pose_landmarks") and results.pose_landmarks:
            lms = results.pose_landmarks.landmark

            if len(lms) > 12:
                # Landmark 0: Nariz
                # Landmark 7: Oreja Izquierda, Landmark 8: Oreja Derecha
                # Landmark 11: Hombro Izquierdo, Landmark 12: Hombro Derecho
                nx, ny = lms[0].x * self.screen_w, lms[0].y * self.screen_h
                le_x, le_y = lms[7].x * self.screen_w, lms[7].y * self.screen_h
                re_x, re_y = lms[8].x * self.screen_w, lms[8].y * self.screen_h
                ls_x, ls_y = lms[11].x * self.screen_w, lms[11].y * self.screen_h
                rs_x, rs_y = lms[12].x * self.screen_w, lms[12].y * self.screen_h

                # Distancia entre orejas (LM 7 a LM 8)
                d_ears = math.hypot(le_x - re_x, le_y - re_y)

                # Punto medio entre hombros y distancia a la nariz (LM 0 a mid-shoulder)
                mid_sh_x = (ls_x + rs_x) / 2.0
                mid_sh_y = (ls_y + rs_y) / 2.0
                d_nose_torso = math.hypot(nx - mid_sh_x, ny - mid_sh_y)

                ear_vis = min(lms[7].visibility, lms[8].visibility)
                
                # Desacoplar escala base de los hombros usando la distancia entre orejas como primaria
                if ear_vis > 0.35 and d_ears > 25.0:
                    measured_scale = float(d_ears * 1.5)
                elif d_nose_torso > 35.0:
                    measured_scale = float(d_nose_torso * 0.9)

                # Actualizar anclajes anatómicos con suavizado exponencial (EMA)
                alpha_pos = 0.15
                self.smoothed_nose = (
                    alpha_pos * nx + (1.0 - alpha_pos) * self.smoothed_nose[0],
                    alpha_pos * ny + (1.0 - alpha_pos) * self.smoothed_nose[1],
                )
                self.smoothed_ear_right = (
                    alpha_pos * re_x + (1.0 - alpha_pos) * self.smoothed_ear_right[0],
                    alpha_pos * re_y + (1.0 - alpha_pos) * self.smoothed_ear_right[1],
                )
                self.smoothed_ear_left = (
                    alpha_pos * le_x + (1.0 - alpha_pos) * self.smoothed_ear_left[0],
                    alpha_pos * le_y + (1.0 - alpha_pos) * self.smoothed_ear_left[1],
                )
                self.smoothed_mid_shoulder = (
                    alpha_pos * mid_sh_x + (1.0 - alpha_pos) * self.smoothed_mid_shoulder[0],
                    alpha_pos * mid_sh_y + (1.0 - alpha_pos) * self.smoothed_mid_shoulder[1],
                )
                self.face_detected = True

        # Suavizado exponencial (EMA) y clamp de 5% para la escala facial (BASE_SCALE)
        if measured_scale is not None and measured_scale > 20.0:
            # Limitar la variación máxima de escala por fotograma al 5% (Clamp de seguridad)
            max_delta = 0.05 * self.last_valid_face_scale
            measured_scale = np.clip(
                measured_scale,
                self.last_valid_face_scale - max_delta,
                self.last_valid_face_scale + max_delta
            )
            
            alpha = self.face_scale_alpha  # 0.15
            self.face_scale = alpha * measured_scale + (1.0 - alpha) * self.face_scale
            self.last_valid_face_scale = self.face_scale
        else:
            # Fallback con el último valor válido registrado
            self.face_scale = self.last_valid_face_scale

        # ── 2. Dimensionamiento Dinámico de Ghost Buttons ──
        self.r_visual = max(18, int(0.35 * self.face_scale))
        self.r_hitbox = max(24, int(0.45 * self.face_scale))
        self.hit_radius_px = self.r_hitbox

        # ── 3. Posicionamiento Anatómico Dinámico ──
        self._reposition_buttons_anatomically()

    def _reposition_buttons_anatomically(self) -> None:
        """
        Calcula las coordenadas en pantalla [px_x, px_y] de los Ghost Buttons
        anclados a la geometría facial y corporal del usuario según la seña activa.
        """
        gesture = self.current_gesture
        fs = self.face_scale

        # Anclajes anatómicos suavizados
        nx, ny = self.smoothed_nose
        re_x, re_y = self.smoothed_ear_right
        mid_x, mid_y = self.smoothed_mid_shoulder

        coords_specs: List[Tuple[float, float, str, bool, int, str, bool, bool]] = []
        gesture_clean = gesture.strip().lower().replace("_", " ")

        if self.training_mode == "ABECEDARIO":
            letra = gesture_clean
            if letra == "j":
                coords_specs = [
                    (nx + 0.8 * fs, ny, "1. Sien (Inicio J)", False, -1, "", False, False),
                    (nx + 0.8 * fs, ny + 0.8 * fs, "2. Abajo", False, -1, "", False, False),
                    (nx, ny + 0.8 * fs, "3. Gancho (Fin J)", False, -1, "", False, False),
                ]
            elif letra == "z":
                coords_specs = [
                    (mid_x - 0.5 * fs, ny, "1. Inicio Z", False, -1, "", False, False),
                    (mid_x + 0.5 * fs, ny, "2. Trazo horiz.", False, -1, "", False, False),
                    (mid_x - 0.5 * fs, ny + 0.8 * fs, "3. Diagonal", False, -1, "", False, False),
                    (mid_x + 0.5 * fs, ny + 0.8 * fs, "4. Fin Z", False, -1, "", False, False),
                ]
            elif letra == "ñ":
                coords_specs = [
                    (mid_x, ny + 0.5 * fs, "1. Centro (Ñ)", False, -1, "", False, False),
                    (mid_x + 0.6 * fs, ny + 0.5 * fs, "2. Derecha", False, -1, "", False, False),
                    (mid_x, ny + 0.5 * fs, "3. Centro", False, -1, "", False, False),
                ]
            else:
                # ── MODO ALFABETO: Único Ghost Anchor fijo a la altura del hombro dominante ──
                anchor_x = mid_x + 1.20 * fs
                anchor_y = mid_y - 0.20 * fs
                coords_specs = [(anchor_x, anchor_y, "Sostener Señal", False, -1, "", False, False)]

        elif gesture_clean == "hola":
            # ── 1. GESTO: "HOLA" (Unimanual) ──
            # Origen (P0): Anclado en la sien (Nose.x + 1.2 * FACE_SCALE, Nose.y - 0.4 * FACE_SCALE)
            p0_x = nx + 1.20 * fs
            p0_y = ny - 0.40 * fs
            # Punto 1 (P1): Desplazamiento oblicuo hacia afuera (Nose.x + 1.8 * FACE_SCALE, Nose.y - 0.5 * FACE_SCALE)
            p1_x = nx + 1.80 * fs
            p1_y = ny - 0.50 * fs
            # Hitbox: Validado posturalmente por HandPoseValidator (index, middle extendidos, ring, pinky flexionados)
            coords_specs = [
                (p0_x, p0_y, "1. Sien", False, -1, "", True, False),
                (p1_x, p1_y, "2. Oblicuo Afuera", False, -1, "", True, False),
            ]

        elif gesture_clean == "gracias":
            # ── 2. GESTO: "GRACIAS" (Unimanual) ──
            # Origen (P0): Anclado en mentón/labio (Nose.x, Nose.y + 0.45 * FACE_SCALE)
            p0_x = nx
            p0_y = ny + 0.45 * fs
            # Punto 1 (P1): Proyección frontal descendente a 45 grados (Nose.x + 0.5 * FACE_SCALE, Nose.y + 1.5 * FACE_SCALE)
            p1_x = nx + 0.50 * fs
            p1_y = ny + 1.50 * fs
            # Transición: rotar mostrando palma orientada hacia arriba al completar
            coords_specs = [
                (p0_x, p0_y, "1. Mentón/Labio", False, -1, "", False, False),
                (p1_x, p1_y, "2. Proyección (Palma Arriba)", False, -1, "", False, True),
            ]

        elif gesture_clean in ("buenos dias", "buenos_dias"):
            # ── 3. GESTO: "BUENOS DIAS" (Bifásico) ──
            # FASE 1 ("buenos"): idéntica a "gracias" (mentón hacia adelante)
            p0_x = nx
            p0_y = ny + 0.45 * fs
            p1_x = nx + 0.50 * fs
            p1_y = ny + 1.50 * fs

            # FASE 2 ("dias"): Bimanual obligatoria con trayectoria simultánea
            # Origen Izquierdo (P_izq_0): Centro pecho izquierdo (Nose.x - 0.3 * FACE_SCALE, Nose.y + 1.2 * FACE_SCALE)
            p_izq_0_x = nx - 0.30 * fs
            p_izq_0_y = ny + 1.20 * fs
            # Origen Derecho (P_der_0): Centro pecho derecho (Nose.x + 0.3 * FACE_SCALE, Nose.y + 1.2 * FACE_SCALE)
            p_der_0_x = nx + 0.30 * fs
            p_der_0_y = ny + 1.20 * fs

            # Trayectoria simultánea: Arcos curvos hacia los laterales externos
            # P_izq_1 hacia afuera-arriba a la izquierda con palma abierta
            p_izq_1_x = nx - 1.60 * fs
            p_izq_1_y = ny + 0.85 * fs
            # P_der_1 hacia afuera-arriba a la derecha con palma abierta
            p_der_1_x = nx + 1.60 * fs
            p_der_1_y = ny + 0.85 * fs

            coords_specs = [
                # FASE 1: "buenos"
                (p0_x, p0_y, "1. Buenos: Mentón", False, -1, "", False, False),
                (p1_x, p1_y, "2. Buenos: Adelante", False, -1, "", False, True),
                # FASE 2: "dias" (Orígenes Simultáneos en Pecho)
                (p_izq_0_x, p_izq_0_y, "3. Días: Pecho Izq", True, 3, "left", False, False),
                (p_der_0_x, p_der_0_y, "3. Días: Pecho Der", True, 2, "right", False, False),
                # FASE 2: "dias" (Arcos Curvos Laterales Simultáneos con palmas abiertas)
                (p_izq_1_x, p_izq_1_y, "4. Días: Arco Izq", True, 5, "left", True, False),
                (p_der_1_x, p_der_1_y, "4. Días: Arco Der", True, 4, "right", True, False),
            ]

        elif gesture_clean in ("por favor", "por_favor"):
            # ── "POR FAVOR": Círculo en órbita horaria sobre el esternón (6 waypoints) ──
            # Centro = Punto medio entre Hombro Izquierdo (LM 11) y Hombro Derecho (LM 12),
            # o (Nose.x, Nose.y + 1.3 * FACE_SCALE)
            centro_x = mid_x
            centro_y = ny + 1.30 * fs if self.face_detected else mid_y + 0.35 * fs
            r_circulo = 0.60 * fs

            self.por_favor_center = (float(centro_x), float(centro_y))
            self.por_favor_radius = float(r_circulo)

            labels = [
                "1. Arriba",
                "2. Arriba-Der",
                "3. Abajo-Der",
                "4. Abajo",
                "5. Abajo-Izq",
                "6. Arriba-Izq",
            ]
            coords_specs = []
            for i in range(6):
                angle = - (i * (2.0 * math.pi / 6.0)) + (math.pi / 2.0)  # Inicia arriba
                wp_x = centro_x + r_circulo * math.cos(angle)
                wp_y = centro_y - r_circulo * math.sin(angle)
                coords_specs.append(
                    (wp_x, wp_y, labels[i], False, -1, "", False, False)
                )

        elif gesture_clean == "caminar":
            # ── GESTO ASIMÉTRICO: "CAMINAR" (Mano Base Superficie vs Mano Activa Piernas) ──
            # Anclajes iniciales por defecto (se actualizarán dinámicamente sobre la Mano Base en tiempo real)
            base_x = mid_x - 0.40 * fs
            base_y = mid_y + 1.20 * fs
            span_x = 0.55 * fs
            coords_specs = [
                (base_x, base_y, "1. Muñeca Base", False, -1, "", False, False),
                (base_x + span_x, base_y, "2. Nudillos Base", False, -1, "", False, False),
                (base_x + 2.0 * span_x, base_y, "3. Puntas Dedos Base", False, -1, "", False, False),
            ]

        elif gesture_clean in ("ayuda", "ayudar"):
            # ── GESTO CINEMÁTICO Y POSTURAL: "AYUDAR" (Mano Base Soporte vs Mano Activa 'Bien') ──
            # Origen (P0): Proyectado al frente exterior adelantado (Nose.x, Nose.y + 1.8 * FACE_SCALE)
            p0_x = nx
            p0_y = ny + 1.80 * fs
            # Destino (P1): Aproximación hacia el esternón (Nose.x, Nose.y + 1.2 * FACE_SCALE)
            p1_x = nx
            p1_y = ny + 1.20 * fs
            coords_specs = [
                (p0_x, p0_y, "1. Frontal (Pulgar Arriba sobre Palma)", False, -1, "", False, False),
                (p1_x, p1_y, "2. Esternón (Aproximación al Pecho)", False, -1, "", False, False),
            ]

        else:
            # Fallback genérico centrado en el torso
            coords_specs = [
                (mid_x - 1.20 * fs, mid_y + 0.50 * fs, "Punto 1", False, -1, "", False, False),
                (mid_x, mid_y + 0.20 * fs, "Punto 2", False, -1, "", False, False),
                (mid_x + 1.20 * fs, mid_y + 0.50 * fs, "Punto 3", False, -1, "", False, False),
            ]

        # Actualizar cada botón con las coordenadas calculadas y delimitadas
        min_x = self.r_hitbox + 20
        max_x = self.screen_w - self.r_hitbox - 20
        min_y = 130 + self.r_hitbox
        max_y = self.screen_h - self.r_hitbox - 20

        for i, (rx, ry, lbl, is_bim, pair_id, side, req_palm_open, req_palm_up) in enumerate(coords_specs):
            clamped_x = int(np.clip(rx, min_x, max_x))
            clamped_y = int(np.clip(ry, min_y, max_y))

            if i < len(self.buttons):
                btn = self.buttons[i]
                btn.px_x = clamped_x
                btn.px_y = clamped_y
                btn.x_norm = clamped_x / self.screen_w
                btn.y_norm = clamped_y / self.screen_h
                btn.label = lbl
                btn.is_bimanual = is_bim
                btn.pair_id = pair_id
                btn.side = side
                btn.require_open_palm = req_palm_open
                btn.require_palm_up = req_palm_up
            else:
                btn = ReactiveGhostButton(
                    id=i,
                    x_norm=clamped_x / self.screen_w,
                    y_norm=clamped_y / self.screen_h,
                    label=lbl,
                    is_active_target=(i == 0),
                    px_x=clamped_x,
                    px_y=clamped_y,
                    is_bimanual=is_bim,
                    pair_id=pair_id,
                    side=side,
                    require_open_palm=req_palm_open,
                    require_palm_up=req_palm_up,
                )
                self.buttons.append(btn)

        if len(self.buttons) > len(coords_specs):
            self.buttons = self.buttons[:len(coords_specs)]

    def _load_gesture_buttons(self, gesture: str) -> None:
        """Carga los waypoints secuenciales para la seña dada adaptados anatómicamente."""
        if hasattr(self, "capture_pipeline") and self.capture_pipeline.is_recording:
            self.capture_pipeline.discard("Reinicio o cambio de seña")

        self.buttons.clear()
        self.active_button_idx = 0
        self.fsm_state = FSMState.WAITING_START
        self.state_enter_time = time.time()
        self.last_interaction_time = time.time()
        self.reverse_rotation_accum = 0.0
        self.palm_invalid_duration = 0.0
        self.prev_cursor_pos = None
        self.prev_cursor_angle = None
        self.palm_warning = False
        self.hand_posture_error = False
        self.hand_posture_error_message = ""

        self._reposition_buttons_anatomically()
        if self.buttons:
            self.buttons[0].is_active_target = True

    def toggle_cursor(self) -> CursorType:
        """Alterna entre cursor de Punta de Índice y Muñeca."""
        if self.cursor_type == CursorType.INDEX_TIP:
            self.cursor_type = CursorType.WRIST
        elif self.cursor_type == CursorType.WRIST:
            self.cursor_type = CursorType.AUTO
        else:
            self.cursor_type = CursorType.INDEX_TIP
        return self.cursor_type

    def resolve_active_hand_cursor(
        self,
        results_hands: Optional[object],
        fallback_points: Optional[dict[str, Optional[np.ndarray]]] = None,
    ) -> Tuple[Optional[Tuple[int, int]], int]:
        """
        Heurística de mano dominante / más cercana (Single-Hand Targeting).
        Si se detectan 2 manos en pantalla, calcula la distancia euclidiana 2D
        entre el Ghost Button activo y el LM 0 (muñeca) o LM 8 (índice) de cada mano.
        Selecciona como 'Mano Activa' la más cercana para colisión y seguimiento.

        Returns:
            Tuple[cursor_px, active_hand_idx]
        """
        if results_hands is None or getattr(results_hands, "multi_hand_landmarks", None) is None:
            if fallback_points:
                cursor_norm = None
                if self.cursor_type == CursorType.INDEX_TIP:
                    cursor_norm = fallback_points.get("right_index")
                    if cursor_norm is None:
                        cursor_norm = fallback_points.get("left_index")
                elif self.cursor_type == CursorType.WRIST:
                    cursor_norm = fallback_points.get("right_wrist")
                    if cursor_norm is None:
                        cursor_norm = fallback_points.get("left_wrist")
                else:
                    for key in ("right_index", "left_index", "right_wrist", "left_wrist"):
                        val = fallback_points.get(key)
                        if val is not None:
                            cursor_norm = val
                            break
                if cursor_norm is not None:
                    return (int(cursor_norm[0] * self.screen_w), int(cursor_norm[1] * self.screen_h)), -1
            return None, -1

        hands_list = results_hands.multi_hand_landmarks

        # Coordenada en píxeles del Ghost Button activo
        btn_x = int(0.5 * self.screen_w)
        btn_y = int(0.5 * self.screen_h)
        if self.active_button_idx < len(self.buttons):
            target_btn = self.buttons[self.active_button_idx]
            btn_x = target_btn.px_x if target_btn.px_x > 0 else int(target_btn.x_norm * self.screen_w)
            btn_y = target_btn.px_y if target_btn.px_y > 0 else int(target_btn.y_norm * self.screen_h)

        candidates = []
        for idx, hand in enumerate(hands_list):
            lm0 = hand.landmark[0]  # Muñeca
            lm8 = hand.landmark[8]  # Punta del índice
            p0 = (int(lm0.x * self.screen_w), int(lm0.y * self.screen_h))
            p8 = (int(lm8.x * self.screen_w), int(lm8.y * self.screen_h))

            dist_0 = math.hypot(p0[0] - btn_x, p0[1] - btn_y)
            dist_8 = math.hypot(p8[0] - btn_x, p8[1] - btn_y)

            if self.cursor_type == CursorType.WRIST:
                dist = dist_0
                cursor_pt = p0
            elif self.cursor_type == CursorType.INDEX_TIP:
                dist = dist_8
                cursor_pt = p8
            else:  # AUTO
                if dist_8 <= dist_0:
                    dist = dist_8
                    cursor_pt = p8
                else:
                    dist = dist_0
                    cursor_pt = p0
            candidates.append((dist, cursor_pt, idx))

        candidates.sort(key=lambda c: c[0])
        best_dist, best_pt, active_idx = candidates[0]
        return best_pt, active_idx

    def update(
        self,
        cursor_pos: Optional[Tuple[int, int]],
        delta_time: float,
        current_frame: Optional[np.ndarray] = None,
        current_spatial: Optional[np.ndarray] = None,
        results: Optional[object] = None,
        results_hands: Optional[object] = None,
    ) -> None:
        """
        Actualiza la máquina de estados FSM, calibración antropométrica y colisión/llenado
        soportando transiciones bifásicas y requerimientos unimanuales / bimanuales.
        """
        now = time.time()
        self.capture_pipeline.update_feedback(delta_time)

        # ── 1. Actualización de Calibración Antropométrica en Tiempo Real ──
        if results is not None:
            self.update_anthropometric_calibration(results)

        # ── 1.1 Verificación de Postura de Cuerpo Completo (FULL_BODY_REQUIRED) ──
        is_posture_valid = True
        if self.current_gesture.strip().lower() in [g.lower() for g in FULL_BODY_REQUIRED]:
            if results is None or not hasattr(results, "pose_landmarks") or not results.pose_landmarks:
                is_posture_valid = False
            else:
                lms = results.pose_landmarks.landmark
                if len(lms) > 24:
                    hip_left = lms[23]
                    hip_right = lms[24]
                    if (
                        hip_left.visibility < 0.65
                        or hip_right.visibility < 0.65
                        or not (0.0 <= hip_left.x <= 1.0 and 0.0 <= hip_left.y <= 1.0)
                        or not (0.0 <= hip_right.x <= 1.0 and 0.0 <= hip_right.y <= 1.0)
                    ):
                        is_posture_valid = False
                else:
                    is_posture_valid = False

        self.posture_warning = not is_posture_valid

        # Manejo de estado de ÉXITO
        if self.fsm_state == FSMState.SUCCESS:
            if now - self.state_enter_time > 2.2:
                self.current_vocab_idx = (self.current_vocab_idx + 1) % len(self.training_vocabulary)
                self.current_gesture = self.training_vocabulary[self.current_vocab_idx]
                self._load_gesture_buttons(self.current_gesture)
            return

        # Manejo de estado de TIEMPO AGOTADO
        if self.fsm_state == FSMState.TIMEOUT_RESET:
            if now - self.state_enter_time > 1.4:
                self._load_gesture_buttons(self.current_gesture)
            return

        # Timeout entre pasos
        if self.fsm_state == FSMState.IN_SEQUENCE:
            if now - self.last_interaction_time > self.step_timeout_sec:
                self.fsm_state = FSMState.TIMEOUT_RESET
                self.state_enter_time = now
                self.capture_pipeline.discard("Tiempo límite de paso excedido")
                return

        # Animación de pulso
        for btn in self.buttons:
            if btn.is_completed and btn.pulse_alpha > 0.0:
                btn.pulse_radius += 70.0 * delta_time
                btn.pulse_alpha = max(0.0, btn.pulse_alpha - 2.2 * delta_time)

        if self.active_button_idx >= len(self.buttons):
            return

        target_btn = self.buttons[self.active_button_idx]

        # ─── A. CASO BIMANUAL SIMULTÁNEO (Buenos días: FASE 2 "días") ─────────
        if target_btn.is_bimanual and 0 <= target_btn.pair_id < len(self.buttons):
            pair_btn = self.buttons[target_btn.pair_id]
            target_btn.is_active_target = True
            pair_btn.is_active_target = True

            num_detected = len(results_hands.multi_hand_landmarks) if (results_hands and getattr(results_hands, "multi_hand_landmarks", None)) else 0
            if num_detected < 2:
                # Bimanual obligatoria: advertencia en HUD
                self.bimanual_warning = True
                target_btn.hover_progress = max(0.0, target_btn.hover_progress - 2.0 * delta_time)
                pair_btn.hover_progress = max(0.0, pair_btn.hover_progress - 2.0 * delta_time)
            else:
                self.bimanual_warning = False
                # Ordenar manos horizontalmente por muñeca (LM 0)
                hands = sorted(results_hands.multi_hand_landmarks, key=lambda h: h.landmark[0].x)
                h_left, h_right = hands[0], hands[1]

                p_left = (int(h_left.landmark[8].x * self.screen_w), int(h_left.landmark[8].y * self.screen_h))
                p_right = (int(h_right.landmark[8].x * self.screen_w), int(h_right.landmark[8].y * self.screen_h))

                btn_left = target_btn if target_btn.side == "left" else pair_btn
                btn_right = pair_btn if pair_btn.side == "right" else target_btn

                d_left = math.hypot(p_left[0] - btn_left.px_x, p_left[1] - btn_left.px_y)
                d_right = math.hypot(p_right[0] - btn_right.px_x, p_right[1] - btn_right.px_y)

                # Requiere palmas abiertas si el botón lo especifica
                left_palm_ok = (not btn_left.require_open_palm) or is_hand_palm_open(h_left)
                right_palm_ok = (not btn_right.require_open_palm) or is_hand_palm_open(h_right)

                hover_left = (d_left <= self.r_hitbox) and left_palm_ok
                hover_right = (d_right <= self.r_hitbox) and right_palm_ok

                if hover_left and hover_right:
                    fill_rate = 1.0 / max(1e-3, self.dwell_activation_time)
                    btn_left.hover_progress = min(1.0, btn_left.hover_progress + fill_rate * delta_time)
                    btn_right.hover_progress = min(1.0, btn_right.hover_progress + fill_rate * delta_time)

                    if btn_left.hover_progress >= 1.0 and btn_right.hover_progress >= 1.0:
                        btn_left.is_completed = True
                        btn_left.is_active_target = False
                        btn_left.pulse_radius = float(self.r_hitbox)
                        btn_left.pulse_alpha = 1.0

                        btn_right.is_completed = True
                        btn_right.is_active_target = False
                        btn_right.pulse_radius = float(self.r_hitbox)
                        btn_right.pulse_alpha = 1.0

                        self.last_interaction_time = now
                        next_idx = max(btn_left.id, btn_right.id) + 1
                        self.active_button_idx = next_idx

                        if self.active_button_idx >= len(self.buttons):
                            self.fsm_state = FSMState.SUCCESS
                            self.state_enter_time = now
                            self.capture_pipeline.commit(gesture=self.current_gesture, category=self.training_mode.lower())
                            self.tts.speak(f"Excelente, has completado {self.current_gesture}")
                        else:
                            self.fsm_state = FSMState.IN_SEQUENCE
                            next_target = self.buttons[self.active_button_idx]
                            next_target.is_active_target = True
                            if next_target.is_bimanual and next_target.pair_id >= 0:
                                self.buttons[next_target.pair_id].is_active_target = True
                else:
                    if not hover_left:
                        btn_left.hover_progress = max(0.0, btn_left.hover_progress - 2.5 * delta_time)
                    if not hover_right:
                        btn_right.hover_progress = max(0.0, btn_right.hover_progress - 2.5 * delta_time)
            return

        # ─── B. GESTO ASIMÉTRICO: "CAMINAR" (Mano Base Superficie vs Mano Activa Piernas) ───
        is_caminar = self.current_gesture.strip().lower().replace("_", " ") == "caminar"
        if is_caminar:
            self.bimanual_warning = False

            # 1. Asignación asimétrica de roles (Mano Base vs Mano Activa)
            base_pts = None
            active_pts = None
            base_idx = -1
            active_idx = -1

            hands_list = getattr(results_hands, "multi_hand_landmarks", None) if results_hands is not None else None
            if hands_list is not None and len(hands_list) >= 2:
                base_pts, active_pts, base_idx, active_idx = identify_walking_hands(hands_list)
            elif results is not None and getattr(results, "right_hand_landmarks", None) and getattr(results, "left_hand_landmarks", None):
                base_pts, active_pts, base_idx, active_idx = identify_walking_hands([results.left_hand_landmarks, results.right_hand_landmarks])

            # 2. Generación dinámica de Waypoints relativos a la Mano Base
            if base_pts is not None:
                min_x = self.r_hitbox + 20
                max_x = self.screen_w - self.r_hitbox - 20
                min_y = 130 + self.r_hitbox
                max_y = self.screen_h - self.r_hitbox - 20

                # WP_0 (Inicio): Muñeca / Base de la palma (LM 0)
                # WP_1 (Paso intermedio): Nudillo medio (LM 9)
                # WP_2 (Final): Puntas dedos medio (LM 12)
                anchors = [
                    (int(base_pts[0, 0] * self.screen_w), int(base_pts[0, 1] * self.screen_h)),
                    (int(base_pts[9, 0] * self.screen_w), int(base_pts[9, 1] * self.screen_h)),
                    (int(base_pts[12, 0] * self.screen_w), int(base_pts[12, 1] * self.screen_h)),
                ]
                for idx_btn, (ax, ay) in enumerate(anchors):
                    if idx_btn < len(self.buttons):
                        clamped_x = int(np.clip(ax, min_x, max_x))
                        clamped_y = int(np.clip(ay, min_y, max_y))
                        btn = self.buttons[idx_btn]
                        btn.px_x = clamped_x
                        btn.px_y = clamped_y
                        btn.x_norm = clamped_x / self.screen_w
                        btn.y_norm = clamped_y / self.screen_h

            # Hitbox: "La colisión la valida la posición de las puntas de los dedos 'caminantes' recorriendo WP_0 -> WP_1 -> WP_2"
            eval_pos = cursor_pos
            if active_pts is not None:
                p_idx_tip = (int(active_pts[8, 0] * self.screen_w), int(active_pts[8, 1] * self.screen_h))
                p_mid_tip = (int(active_pts[12, 0] * self.screen_w), int(active_pts[12, 1] * self.screen_h))
                eval_pos = ((p_idx_tip[0] + p_mid_tip[0]) // 2, (p_idx_tip[1] + p_mid_tip[1]) // 2)

            btn_px_x = target_btn.px_x if target_btn.px_x > 0 else int(target_btn.x_norm * self.screen_w)
            btn_px_y = target_btn.px_y if target_btn.px_y > 0 else int(target_btn.y_norm * self.screen_h)

            dist_to_btn = 999999.0
            if eval_pos is not None:
                dist_to_btn = math.hypot(eval_pos[0] - btn_px_x, eval_pos[1] - btn_px_y)

            is_in_hitbox = (dist_to_btn <= self.r_hitbox)
            is_hovering = False

            if is_in_hitbox:
                if active_pts is not None and base_pts is not None:
                    ok_act, err_act = is_walking_active_hand(active_pts)
                    ok_base, err_base = is_walking_base_hand(base_pts)
                    if ok_act and ok_base:
                        # Cumple simultáneamente posición Hitbox y configuración postural
                        is_hovering = True
                        target_btn.has_posture_error = False
                        target_btn.posture_error_msg = ""
                        self.hand_posture_error = False
                        self.hand_posture_error_message = ""
                    else:
                        # Colisión inválida: dentro del hitbox con postura errónea
                        is_hovering = False
                        err_msg = err_act or err_base or "Corrige la postura para 'Caminar'"
                        target_btn.has_posture_error = True
                        target_btn.posture_error_msg = err_msg
                        self.hand_posture_error = True
                        self.hand_posture_error_message = err_msg
                        self.hand_posture_error_time = now
                elif results_hands is None:
                    # Simulación sintética en tests de solo cursor
                    is_hovering = True
                    target_btn.has_posture_error = False
                    self.hand_posture_error = False
                else:
                    is_hovering = False
                    err_msg = "Ubica ambas manos: una base plana y una caminante"
                    target_btn.has_posture_error = True
                    target_btn.posture_error_msg = err_msg
                    self.hand_posture_error = True
                    self.hand_posture_error_message = err_msg
                    self.hand_posture_error_time = now
            else:
                target_btn.has_posture_error = False
                target_btn.posture_error_msg = ""
                if now - self.hand_posture_error_time > 1.2:
                    self.hand_posture_error = False
                    self.hand_posture_error_message = ""

            # Bloqueo si la seña exige cuerpo completo y la postura es inválida
            if self.posture_warning and target_btn.id == 0:
                is_hovering = False
                target_btn.hover_progress = 0.0

            if is_hovering:
                req_time = self.initial_hold_time if target_btn.id == 0 else self.dwell_activation_time
                fill_rate = 1.0 / max(1e-3, req_time)
                target_btn.hover_progress = min(1.0, target_btn.hover_progress + fill_rate * delta_time)

                if target_btn.hover_progress >= 1.0:
                    target_btn.is_completed = True
                    target_btn.is_active_target = False
                    target_btn.pulse_radius = float(self.r_hitbox)
                    target_btn.pulse_alpha = 1.0
                    target_btn.completion_time = now
                    self.last_interaction_time = now

                    # 3. Persistencia: iniciar registro al pasar WP_0
                    if target_btn.id == 0:
                        self.capture_pipeline.start_recording(
                            gesture="caminar",
                            category=self.training_mode.lower(),
                            initial_frame=current_frame,
                            initial_spatial=current_spatial,
                        )

                    self.active_button_idx += 1
                    if self.active_button_idx < len(self.buttons):
                        self.fsm_state = FSMState.IN_SEQUENCE
                        next_btn = self.buttons[self.active_button_idx]
                        next_btn.is_active_target = True
                    else:
                        # Secuencia WP_0 -> WP_1 -> WP_2 completada en orden sobre la mano base
                        self.fsm_state = FSMState.SUCCESS
                        self.state_enter_time = now
                        self.capture_pipeline.commit(gesture="caminar", category=self.training_mode.lower())
                        self.tts.speak("Excelente, has completado caminar")
            else:
                target_btn.hover_progress = max(0.0, target_btn.hover_progress - 3.0 * delta_time)

            return

        # ─── B2. GESTO ACOPLADO: "AYUDAR" (Mano Base Soporte vs Mano Activa 'Bien') ───
        is_ayudar = self.current_gesture.strip().lower().replace("_", " ") in ("ayuda", "ayudar")
        if is_ayudar:
            self.bimanual_warning = False

            # 1. Asignación de roles: Mano Base (Soporte) vs Mano Activa ('Bien')
            base_pts = None
            active_pts = None
            base_idx = -1
            active_idx = -1

            hands_list = getattr(results_hands, "multi_hand_landmarks", None) if results_hands is not None else None
            if hands_list is not None and len(hands_list) >= 2:
                base_pts, active_pts, base_idx, active_idx = identify_helping_hands(hands_list)
            elif results is not None and getattr(results, "right_hand_landmarks", None) and getattr(results, "left_hand_landmarks", None):
                base_pts, active_pts, base_idx, active_idx = identify_helping_hands([results.left_hand_landmarks, results.right_hand_landmarks])

            # Anclar cursor de seguimiento a la mano activa (muñeca y pulgar MCP/centro)
            eval_pos = cursor_pos
            if active_pts is not None:
                p_wrist = (int(active_pts[LM_WRIST, 0] * self.screen_w), int(active_pts[LM_WRIST, 1] * self.screen_h))
                p_mcp = (int(active_pts[LM_THUMB_MCP, 0] * self.screen_w), int(active_pts[LM_THUMB_MCP, 1] * self.screen_h))
                eval_pos = ((p_wrist[0] + p_mcp[0]) // 2, (p_wrist[1] + p_mcp[1]) // 2)

            btn_px_x = target_btn.px_x if target_btn.px_x > 0 else int(target_btn.x_norm * self.screen_w)
            btn_px_y = target_btn.px_y if target_btn.px_y > 0 else int(target_btn.y_norm * self.screen_h)

            dist_to_btn = 999999.0
            if eval_pos is not None:
                dist_to_btn = math.hypot(eval_pos[0] - btn_px_x, eval_pos[1] - btn_px_y)

            is_in_hitbox = (dist_to_btn <= self.r_hitbox)
            is_hovering = False

            if is_in_hitbox:
                if active_pts is not None and base_pts is not None:
                    ok_act, err_act = is_helping_active_hand(active_pts)
                    ok_base, err_base = is_helping_base_hand(base_pts)
                    ok_cpl, err_cpl = are_helping_hands_coupled(base_pts, active_pts)

                    if ok_act and ok_base and ok_cpl:
                        is_hovering = True
                        target_btn.has_posture_error = False
                        target_btn.posture_error_msg = ""
                        self.hand_posture_error = False
                        self.hand_posture_error_message = ""
                    else:
                        is_hovering = False
                        err_msg = err_act or err_base or err_cpl or "Corrige la postura para 'Ayudar'"
                        target_btn.has_posture_error = True
                        target_btn.posture_error_msg = err_msg
                        self.hand_posture_error = True
                        self.hand_posture_error_message = err_msg
                        self.hand_posture_error_time = now
                elif results_hands is None:
                    # Simulación sintética en tests sin cámara
                    is_hovering = True
                    target_btn.has_posture_error = False
                    self.hand_posture_error = False
                else:
                    is_hovering = False
                    err_msg = "Ubica ambas manos: base abierta y mano activa con pulgar arriba"
                    target_btn.has_posture_error = True
                    target_btn.posture_error_msg = err_msg
                    self.hand_posture_error = True
                    self.hand_posture_error_message = err_msg
                    self.hand_posture_error_time = now
            else:
                target_btn.has_posture_error = False
                target_btn.posture_error_msg = ""
                if now - self.hand_posture_error_time > 1.2:
                    self.hand_posture_error = False
                    self.hand_posture_error_message = ""

            # Verificación continua durante la trayectoria (FSMState.IN_SEQUENCE):
            # Mantener pulgar arriba y manos acopladas hacia el pecho
            if self.fsm_state == FSMState.IN_SEQUENCE:
                if active_pts is not None:
                    ok_act_seq, err_act_seq = is_helping_active_hand(active_pts)
                    if not ok_act_seq:
                        self.hand_posture_error = True
                        self.hand_posture_error_message = err_act_seq or "Mantén el pulgar arriba durante la traslación"
                        self.hand_posture_error_time = now
                        is_hovering = False
                if base_pts is not None and active_pts is not None:
                    ok_cpl_seq, err_cpl_seq = are_helping_hands_coupled(base_pts, active_pts)
                    if not ok_cpl_seq:
                        self.hand_posture_error = True
                        self.hand_posture_error_message = err_cpl_seq or "Mantén las manos acopladas hacia el pecho"
                        self.hand_posture_error_time = now
                        is_hovering = False

            # Bloqueo si la seña exige cuerpo completo y la postura es inválida
            if self.posture_warning and target_btn.id == 0:
                is_hovering = False
                target_btn.hover_progress = 0.0

            if is_hovering:
                req_time = self.initial_hold_time if target_btn.id == 0 else self.dwell_activation_time
                fill_rate = 1.0 / max(1e-3, req_time)
                target_btn.hover_progress = min(1.0, target_btn.hover_progress + fill_rate * delta_time)

                if target_btn.hover_progress >= 1.0:
                    target_btn.is_completed = True
                    target_btn.is_active_target = False
                    target_btn.pulse_radius = float(self.r_hitbox)
                    target_btn.pulse_alpha = 1.0
                    target_btn.completion_time = now
                    self.last_interaction_time = now

                    # Persistencia: Iniciar grabación en P0
                    if target_btn.id == 0:
                        self.capture_pipeline.start_recording(
                            gesture="ayudar",
                            category="verbos",
                            initial_frame=current_frame,
                            initial_spatial=current_spatial,
                        )

                    self.active_button_idx += 1
                    if self.active_button_idx < len(self.buttons):
                        self.fsm_state = FSMState.IN_SEQUENCE
                        next_btn = self.buttons[self.active_button_idx]
                        next_btn.is_active_target = True
                    else:
                        # Secuencia P0 -> P1 completada
                        self.fsm_state = FSMState.SUCCESS
                        self.state_enter_time = now
                        self.capture_pipeline.commit(gesture="ayudar", category="verbos")
                        self.tts.speak("Excelente, has completado ayudar")
            else:
                target_btn.hover_progress = max(0.0, target_btn.hover_progress - 3.0 * delta_time)

            return

        # ─── C. CASO UNIMANUAL (Hola, Gracias, Por favor, Buenos Días FASE 1, Abecedario) ───
        self.bimanual_warning = False
        is_por_favor = self.current_gesture.strip().lower().replace("_", " ") == "por favor"

        btn_px_x = target_btn.px_x if target_btn.px_x > 0 else int(target_btn.x_norm * self.screen_w)
        btn_px_y = target_btn.px_y if target_btn.px_y > 0 else int(target_btn.y_norm * self.screen_h)

        # Distancia euclidiana al centro del Ghost Button
        dist_to_btn = 999999.0
        if cursor_pos is not None:
            dist_to_btn = math.hypot(cursor_pos[0] - btn_px_x, cursor_pos[1] - btn_px_y)

        is_in_hitbox = (dist_to_btn <= self.r_hitbox)

        # Obtener landmarks de la mano activa
        active_lms = None
        if results_hands is not None and getattr(results_hands, "multi_hand_landmarks", None) is not None:
            num_h = len(results_hands.multi_hand_landmarks)
            if num_h > 0:
                idx = self.active_hand_idx if 0 <= self.active_hand_idx < num_h else 0
                active_lms = results_hands.multi_hand_landmarks[idx]
        elif results is not None:
            active_lms = getattr(results, "right_hand_landmarks", None)
            if active_lms is None:
                active_lms = getattr(results, "left_hand_landmarks", None)

        is_hovering = False

        if is_in_hitbox:
            # 1. Validación postural mediante HandPoseValidator (Queremas por seña)
            has_hand_pipeline = (
                results_hands is not None
                or (
                    results is not None
                    and (
                        getattr(results, "right_hand_landmarks", None) is not None
                        or getattr(results, "left_hand_landmarks", None) is not None
                    )
                )
            )
            if has_hand_pipeline:
                posture_ok, posture_err = self.hand_validator.validate_pose(
                    self.current_gesture,
                    active_lms,
                    waypoint_idx=target_btn.id,
                    total_waypoints=len(self.buttons),
                )
            else:
                # En ausencia de pipeline de detección de manos (tests de cinemática/antropometría), asumir válido
                posture_ok, posture_err = True, None

            # Requisitos articulares adicionales solo si la seña no tiene especificación postural en GESTURE_REQUIREMENTS
            spec = self.hand_validator.get_gesture_spec(self.current_gesture)
            if spec is None:
                if posture_ok and target_btn.require_open_palm and not is_hand_palm_open(active_lms):
                    posture_ok = False
                    posture_err = "Extiende la palma de la mano"

                if posture_ok and target_btn.require_palm_up and not is_hand_palm_up(active_lms):
                    posture_ok = False
                    posture_err = "Orienta la palma hacia arriba"

            if posture_ok:
                # Cumple SIMULTÁNEAMENTE:
                # a) Posición espacial dentro del radio Hitbox.
                # b) Configuración postural correcta según GESTURE_REQUIREMENTS.
                is_hovering = True
                target_btn.has_posture_error = False
                target_btn.posture_error_msg = ""
                self.hand_posture_error = False
                self.hand_posture_error_message = ""
            else:
                # Colisión inválida: mano dentro del radio Hitbox pero con postura errónea
                is_hovering = False
                target_btn.has_posture_error = True
                target_btn.posture_error_msg = posture_err or "Corrige la postura de la mano"
                self.hand_posture_error = True
                self.hand_posture_error_message = posture_err or "Corrige la postura de la mano"
                self.hand_posture_error_time = now
        else:
            # Fuera del radio Hitbox
            target_btn.has_posture_error = False
            target_btn.posture_error_msg = ""
            if now - self.hand_posture_error_time > 1.2:
                self.hand_posture_error = False
                self.hand_posture_error_message = ""

        # Validaciones cinemáticas activas durante la secuencia para "Por favor"
        if is_por_favor:
            if self.fsm_state == FSMState.IN_SEQUENCE and cursor_pos is not None:
                cx, cy = cursor_pos
                centro_x, centro_y = self.por_favor_center
                r_circulo = self.por_favor_radius

                # a) Desviación radial excesiva (> 1.5 * R_circulo)
                dist_center = math.hypot(cx - centro_x, cy - centro_y)
                if dist_center > 1.50 * r_circulo:
                    self.fsm_state = FSMState.TIMEOUT_RESET
                    self.state_enter_time = now
                    self.capture_pipeline.discard("Desviación radial excesiva (> 1.5 * R)")
                    return

                # b) Inversión del sentido de giro (debe ser horario)
                cur_angle = math.atan2(cy - centro_y, cx - centro_x)
                if self.prev_cursor_pos is not None:
                    px, py = self.prev_cursor_pos
                    move_dist = math.hypot(cx - px, cy - py)
                    if move_dist > 5.0 and self.prev_cursor_angle is not None:
                        delta_angle = (cur_angle - self.prev_cursor_angle + math.pi) % (2.0 * math.pi) - math.pi
                        if delta_angle < -0.08:
                            self.reverse_rotation_accum += abs(delta_angle)
                            if self.reverse_rotation_accum > 0.60:
                                self.fsm_state = FSMState.TIMEOUT_RESET
                                self.state_enter_time = now
                                self.capture_pipeline.discard("Giro invertido (debe ser en sentido horario)")
                                return
                        elif delta_angle > 0.05:
                            self.reverse_rotation_accum = max(0.0, self.reverse_rotation_accum - delta_angle * 0.5)

                self.prev_cursor_angle = cur_angle
                self.prev_cursor_pos = (cx, cy)
        else:
            self.reverse_rotation_accum = 0.0

        # Si la seña exige cuerpo completo y la postura es inválida, bloquear inicio de captura
        if self.posture_warning and target_btn.id == 0:
            is_hovering = False
            target_btn.hover_progress = 0.0

        if is_hovering:
            req_time = self.initial_hold_time if target_btn.id == 0 else self.dwell_activation_time
            # Timer extendido (0.8s) para letras estáticas del abecedario
            if self.training_mode == "ABECEDARIO" and target_btn.id == 0 and len(self.buttons) == 1:
                req_time = 0.80

            fill_rate = 1.0 / max(1e-3, req_time)
            target_btn.hover_progress = min(1.0, target_btn.hover_progress + fill_rate * delta_time)

            if target_btn.hover_progress >= 1.0:
                target_btn.is_completed = True
                target_btn.is_active_target = False
                target_btn.pulse_radius = float(self.r_hitbox)
                target_btn.pulse_alpha = 1.0
                target_btn.completion_time = now
                self.last_interaction_time = now
                self.reverse_rotation_accum = 0.0

                # Disparador inicial
                if target_btn.id == 0:
                    save_label = "por_favor" if is_por_favor else self.current_gesture
                    self.capture_pipeline.start_recording(
                        gesture=save_label,
                        category=self.training_mode.lower(),
                        initial_frame=current_frame,
                        initial_spatial=current_spatial,
                    )

                self.active_button_idx += 1
                if self.active_button_idx < len(self.buttons):
                    self.fsm_state = FSMState.IN_SEQUENCE
                    next_btn = self.buttons[self.active_button_idx]
                    next_btn.is_active_target = True
                    # Transición inmediata a FASE 2 si el siguiente es par bimanual
                    if next_btn.is_bimanual and next_btn.pair_id >= 0:
                        self.buttons[next_btn.pair_id].is_active_target = True
                else:
                    self.fsm_state = FSMState.SUCCESS
                    self.state_enter_time = now
                    save_label = "por_favor" if is_por_favor else self.current_gesture
                    self.capture_pipeline.commit(gesture=save_label, category=self.training_mode.lower())
                    if is_por_favor:
                        self.tts.speak("Excelente, por favor registrado")
                    else:
                        self.tts.speak(f"Excelente, has completado {self.current_gesture}")
        else:
            target_btn.hover_progress = max(0.0, target_btn.hover_progress - 3.0 * delta_time)

    def _render_circular_guide(self, overlay: np.ndarray, glow_layer: np.ndarray) -> None:
        """
        Renderiza la guía cinemática como un círculo punteado suave en el esternón
        con flecha direccional en sentido horario y marcador central.
        """
        cx, cy = int(self.por_favor_center[0]), int(self.por_favor_center[1])
        r = int(self.por_favor_radius)
        if r < 10:
            return

        pulse = 0.5 + 0.5 * math.sin(time.time() * 5.0)

        # 1. Marcador del anclaje central (Esternón)
        cv2.circle(overlay, (cx, cy), 4, (0, 200, 255), -1, lineType=cv2.LINE_AA)
        cv2.line(overlay, (cx - 8, cy), (cx + 8, cy), (0, 200, 255), 1, lineType=cv2.LINE_AA)
        cv2.line(overlay, (cx, cy - 8), (cx, cy + 8), (0, 200, 255), 1, lineType=cv2.LINE_AA)
        cv2.putText(overlay, "Esternon", (cx - 28, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 255), 1)

        # 2. Círculo punteado suave (36 segmentos discontinuos)
        num_segments = 36
        dash_step = 360.0 / num_segments
        dash_len = dash_step * 0.60
        circle_color = (0, int(210 + 45 * pulse), 255)

        for s in range(num_segments):
            start_ang = s * dash_step
            end_ang = start_ang + dash_len
            cv2.ellipse(
                overlay,
                (cx, cy),
                (r, r),
                0,
                start_ang,
                end_ang,
                circle_color,
                2,
                lineType=cv2.LINE_AA,
            )
            cv2.ellipse(
                glow_layer,
                (cx, cy),
                (r, r),
                0,
                start_ang,
                end_ang,
                (0, 180, 255),
                4,
                lineType=cv2.LINE_AA,
            )

        # 3. Flecha direccional en sentido horario (Clockwise arrow)
        # Posición: cuadrante superior derecho (ángulo 35 grados en convención trigonométrica)
        ang_deg = 35.0
        ang_rad = math.radians(ang_deg)
        tip_x = int(cx + r * math.cos(ang_rad))
        tip_y = int(cy - r * math.sin(ang_rad))

        # Vector tangente horario en coords de pantalla (Y hacia abajo)
        tangent_x = math.sin(ang_rad)
        tangent_y = math.cos(ang_rad)
        norm_t = math.hypot(tangent_x, tangent_y)
        if norm_t > 1e-6:
            tangent_x /= norm_t
            tangent_y /= norm_t

        arrow_size = max(14, int(0.14 * self.face_scale))
        p_base = (tip_x - tangent_x * arrow_size, tip_y - tangent_y * arrow_size)
        perp_x = -tangent_y
        perp_y = tangent_x

        p_wing1 = (
            int(p_base[0] + perp_x * (arrow_size * 0.55)),
            int(p_base[1] + perp_y * (arrow_size * 0.55)),
        )
        p_wing2 = (
            int(p_base[0] - perp_x * (arrow_size * 0.55)),
            int(p_base[1] - perp_y * (arrow_size * 0.55)),
        )

        arrow_pts = np.array([(tip_x, tip_y), p_wing1, p_wing2], dtype=np.int32)
        cv2.fillPoly(overlay, [arrow_pts], (0, 255, 255), lineType=cv2.LINE_AA)
        cv2.fillPoly(glow_layer, [arrow_pts], (0, 220, 255), lineType=cv2.LINE_AA)

        # Arco continuo de aproximación a la flecha
        cv2.ellipse(
            overlay,
            (cx, cy),
            (r, r),
            0,
            290,
            325,
            (0, 255, 255),
            3,
            lineType=cv2.LINE_AA,
        )

    def render(
        self,
        frame: np.ndarray,
        cursor_pos: Optional[Tuple[int, int]],
        results: object = None,
        results_hands: object = None,
        active_hand_idx: int = 0,
    ) -> np.ndarray:
        """
        Renderiza los Ghost Buttons, estela guía, retícula del cursor, esqueleto articular
        completo (21 landmarks) con resaltado unimanual y HUD en resolución de pantalla.
        """
        # Capas para mezclas alfa translúcidas
        overlay = frame.copy()
        glow_layer = np.zeros_like(frame)

        # ─── 1. Trazar Estela Guía entre Waypoints ────────────────────────────
        btn_coords = [
            (b.px_x if b.px_x > 0 else int(b.x_norm * self.screen_w),
             b.px_y if b.px_y > 0 else int(b.y_norm * self.screen_h))
            for b in self.buttons
        ]

        gesture_clean = self.current_gesture.strip().lower().replace("_", " ")
        if gesture_clean in ("buenos dias", "buenos_dias") and len(self.buttons) >= 6:
            # Separar FASE 1 (mentón -> adelante) de FASE 2 (pecho izq -> arco izq, pecho der -> arco der)
            segments = [(0, 1), (2, 4), (3, 5)]
        elif gesture_clean == "por favor" and len(self.buttons) == 6:
            # Recorrido circular cerrado conectando 0->1->2->3->4->5->0
            segments = [(i, (i + 1) % 6) for i in range(6)]
            self._render_circular_guide(overlay, glow_layer)
        else:
            segments = [(i, i + 1) for i in range(len(btn_coords) - 1)]

        for idx1, idx2 in segments:
            if idx1 < len(btn_coords) and idx2 < len(btn_coords):
                p1 = btn_coords[idx1]
                p2 = btn_coords[idx2]
                b1 = self.buttons[idx1]
                b2 = self.buttons[idx2]

                if b1.is_completed and b2.is_completed:
                    line_color = (0, 255, 200)
                    thickness = 3
                elif b1.is_active_target or b2.is_active_target:
                    pulse = 0.5 + 0.5 * math.sin(time.time() * 8.0)
                    line_color = (0, int(200 + 55 * pulse), 255)
                    thickness = 4
                else:
                    line_color = (90, 90, 90)
                    thickness = 1

                cv2.line(glow_layer, p1, p2, line_color, thickness * 2, lineType=cv2.LINE_AA)
                cv2.line(overlay, p1, p2, line_color, thickness, lineType=cv2.LINE_AA)

        # ─── 2. Estela Dinámica Guía desde Cursor hacia Objetivo Activo ───────
        if self.active_button_idx < len(self.buttons):
            active_btn = self.buttons[self.active_button_idx]
            if active_btn.is_bimanual and 0 <= active_btn.pair_id < len(self.buttons) and results_hands and getattr(results_hands, "multi_hand_landmarks", None) and len(results_hands.multi_hand_landmarks) >= 2:
                # Guías duales simultáneas desde cada mano hacia su botón lateral correspondiente
                hands = sorted(results_hands.multi_hand_landmarks, key=lambda h: h.landmark[0].x)
                cl_x = int(hands[0].landmark[8].x * self.screen_w)
                cl_y = int(hands[0].landmark[8].y * self.screen_h)
                cr_x = int(hands[1].landmark[8].x * self.screen_w)
                cr_y = int(hands[1].landmark[8].y * self.screen_h)

                pair_btn = self.buttons[active_btn.pair_id]
                bl = active_btn if active_btn.side == "left" else pair_btn
                br = pair_btn if pair_btn.side == "right" else active_btn

                cv2.line(glow_layer, (cl_x, cl_y), (bl.px_x, bl.px_y), (0, 255, 255), 3, lineType=cv2.LINE_AA)
                cv2.line(overlay, (cl_x, cl_y), (bl.px_x, bl.px_y), (255, 255, 255), 1, lineType=cv2.LINE_AA)
                cv2.line(glow_layer, (cr_x, cr_y), (br.px_x, br.px_y), (0, 255, 255), 3, lineType=cv2.LINE_AA)
                cv2.line(overlay, (cr_x, cr_y), (br.px_x, br.px_y), (255, 255, 255), 1, lineType=cv2.LINE_AA)
            elif cursor_pos:
                tx, ty = btn_coords[self.active_button_idx]
                cx, cy = cursor_pos
                cv2.line(glow_layer, (cx, cy), (tx, ty), (0, 255, 255), 3, lineType=cv2.LINE_AA)
                cv2.line(overlay, (cx, cy), (tx, ty), (255, 255, 255), 1, lineType=cv2.LINE_AA)

        # ─── 2.5 Renderizado Completo de Dedos (21 Landmarks) y Selección Unimanual ───
        if results_hands and getattr(results_hands, "multi_hand_landmarks", None):
            import mediapipe as mp
            mp_drawing = mp.solutions.drawing_utils
            mp_hands = mp.solutions.hands

            num_detected = len(results_hands.multi_hand_landmarks)
            for idx, hand_lms in enumerate(results_hands.multi_hand_landmarks):
                is_active = (idx == active_hand_idx or num_detected == 1)

                if is_active:
                    # Mano Activa: Verde brillante
                    lm_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4)
                    conn_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                else:
                    # Otra Mano: Tono tenue/gris para retroalimentación visual clara
                    lm_spec = mp_drawing.DrawingSpec(color=(120, 120, 120), thickness=1, circle_radius=3)
                    conn_spec = mp_drawing.DrawingSpec(color=(70, 70, 70), thickness=1)

                mp_drawing.draw_landmarks(
                    overlay,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                    lm_spec,
                    conn_spec,
                )

            # En MODO ABECEDARIO: evaluar vector postural sobre la Mano Activa
            if self.training_mode == "ABECEDARIO":
                effective_idx = active_hand_idx if 0 <= active_hand_idx < num_detected else 0
                active_lms = results_hands.multi_hand_landmarks[effective_idx]
                h, w = overlay.shape[:2]
                lms_array = np.array([(lm.x, lm.y, lm.z) for lm in active_lms.landmark])
                wrist = lms_array[0]
                centered = lms_array - wrist
                max_abs = np.max(np.abs(centered))
                normalized = centered / max_abs if max_abs > 1e-9 else centered

                # Panel HUD Dactilológico
                panel_x, panel_y = 25, 200
                overlay_panel = overlay.copy()
                cv2.rectangle(overlay_panel, (panel_x - 15, panel_y - 30), (panel_x + 360, panel_y + 160), (20, 20, 25), -1)
                cv2.addWeighted(overlay_panel, 0.8, overlay, 0.2, 0, overlay)

                cv2.putText(overlay, f"POSTURA CANONICA: '{self.current_gesture.upper()}'", (panel_x, panel_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                tips = [4, 8, 12, 16, 20]
                tip_names = ["Pulgar", "Indice", "Medio", "Anular", "Menique"]

                for i, (tip_idx, name) in enumerate(zip(tips, tip_names)):
                    # Resaltar la punta del dedo real en el frame
                    cx, cy = int(active_lms.landmark[tip_idx].x * w), int(active_lms.landmark[tip_idx].y * h)
                    cv2.circle(overlay, (cx, cy), 10, (255, 0, 255), 2, lineType=cv2.LINE_AA)

                    # Mostrar vector normalizado en el panel HUD
                    vec = normalized[tip_idx]
                    text_vec = f"{name:>7}: [{vec[0]:+0.2f}, {vec[1]:+0.2f}, {vec[2]:+0.2f}]"
                    cv2.putText(overlay, text_vec, (panel_x, panel_y + 25 + i * 25), cv2.FONT_HERSHEY_DUPLEX, 0.5, (220, 220, 220), 1)

        # ─── 3. Renderizado de Ghost Buttons (Hitbox Dynamics) ────────────────
        for idx, btn in enumerate(self.buttons):
            bx, by = btn_coords[idx]
            r = self.r_visual  # Radio visual del botón: R_visual = int(0.35 * FACE_SCALE)

            # Efecto de pulso expansivo al acertar
            if btn.pulse_alpha > 0.0:
                p_r = int(btn.pulse_radius)
                p_color = (0, 255, 100)
                cv2.circle(glow_layer, (bx, by), p_r, p_color, 4, lineType=cv2.LINE_AA)

            if btn.is_completed:
                # ── ESTADO COMPLETADO ──
                cv2.circle(overlay, (bx, by), r, (0, 200, 100), -1, lineType=cv2.LINE_AA)
                cv2.circle(overlay, (bx, by), r, (255, 255, 255), 2, lineType=cv2.LINE_AA)
                cv2.putText(
                    overlay,
                    "OK",
                    (bx - 16, by + 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            elif btn.is_active_target:
                # ── ESTADO OBJETIVO ACTIVO (Esperando interacción de la mano) ──
                if getattr(btn, "has_posture_error", False):
                    # ── COLISIÓN INVÁLIDA: Mano dentro del radio Hitbox pero con postura errónea ──
                    # Dibujar Hitbox en color rojo/naranja tenue
                    hitbox_warn_color = (0, 80, 240)  # BGR: naranja/rojo tenue
                    # Relleno translúcido tenue del Hitbox
                    cv2.circle(glow_layer, (bx, by), self.r_hitbox, (0, 60, 210), -1, lineType=cv2.LINE_AA)
                    cv2.circle(overlay, (bx, by), self.r_hitbox, hitbox_warn_color, 2, lineType=cv2.LINE_AA)

                    # Borde exterior del botón en rojo/naranja pulsante
                    pulse_err = 0.5 + 0.5 * math.sin(time.time() * 8.0)
                    btn_border_err = (0, int(60 + 50 * pulse_err), int(210 + 45 * pulse_err))
                    cv2.circle(glow_layer, (bx, by), r + 4, btn_border_err, 6, lineType=cv2.LINE_AA)
                    cv2.circle(overlay, (bx, by), r, (0, 80, 240), 3, lineType=cv2.LINE_AA)

                    # Icono de advertencia en el centro
                    cv2.putText(
                        overlay,
                        "!",
                        (bx - 6, by + 9),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.85,
                        (0, 210, 255),
                        2,
                    )
                    # Mensaje correctivo rápido inmediatamente bajo el Hitbox
                    err_text = getattr(btn, "posture_error_msg", "Corrige la postura de la mano")
                    t_size = cv2.getTextSize(err_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)[0]
                    tx = max(15, bx - t_size[0] // 2)
                    ty = by + self.r_hitbox + 26
                    cv2.putText(
                        overlay,
                        err_text,
                        (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (0, 100, 255),
                        2,
                    )
                else:
                    # Mostrar zona de tolerancia de colisión estándar (Hitbox R_hitbox)
                    cv2.circle(overlay, (bx, by), self.r_hitbox, (60, 60, 60), 1, lineType=cv2.LINE_AA)
                    # Borde exterior pulsante
                    pulse = 0.5 + 0.5 * math.sin(time.time() * 6.0)
                    border_color = (0, int(210 + 45 * pulse), 255)
                    cv2.circle(glow_layer, (bx, by), r + 4, border_color, 6, lineType=cv2.LINE_AA)
                    cv2.circle(overlay, (bx, by), r, (0, 230, 255), 3, lineType=cv2.LINE_AA)

                    # Llenado radial progresivo de colisión (Feedback de progreso)
                    if btn.hover_progress > 0.0:
                        fill_r = int(r * btn.hover_progress)
                        # Disco interior brillante de llenado
                        fill_color = (0, 255, 120) if btn.hover_progress >= 0.8 else (0, 240, 255)
                        cv2.circle(overlay, (bx, by), fill_r, fill_color, -1, lineType=cv2.LINE_AA)
                        cv2.circle(glow_layer, (bx, by), fill_r + 6, fill_color, 4, lineType=cv2.LINE_AA)

                    # Etiqueta / Número
                    num_text = str(idx + 1)
                    cv2.putText(
                        overlay,
                        num_text,
                        (bx - 8, by + 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    if btn.label:
                        cv2.putText(
                            overlay,
                            btn.label,
                            (bx - 35, by + r + 24),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 240, 255),
                            2,
                        )

            else:
                # ── ESTADO PASIVO (Waypoints futuros: borde sutil y baja opacidad) ──
                cv2.circle(overlay, (bx, by), r, (140, 140, 140), 1, lineType=cv2.LINE_AA)
                cv2.circle(overlay, (bx, by), 4, (160, 160, 160), -1, lineType=cv2.LINE_AA)
                cv2.putText(
                    overlay,
                    str(idx + 1),
                    (bx - 6, by + 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (160, 160, 160),
                    1,
                )

        # ─── 4. Renderizado del Cursor Reactivo en la Mano ────────────────────
        if cursor_pos:
            cx, cy = cursor_pos
            # Anillo de la retícula del cursor
            cv2.circle(glow_layer, (cx, cy), 18, (0, 255, 255), 3, lineType=cv2.LINE_AA)
            cv2.circle(overlay, (cx, cy), 14, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.circle(overlay, (cx, cy), 4, (0, 255, 255), -1, lineType=cv2.LINE_AA)

        # Fusión alfa de capas translúcidas
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.addWeighted(glow_layer, 0.45, frame, 1.0, 0, frame)

        # ─── 5. Banner de HUD y Notificaciones FSM ───────────────────────────
        self._render_hud(frame)

        # ─── 6. Feedback Visual de Descarte (Tinte rojo tenue) ───────────────
        frame = self.capture_pipeline.apply_visual_feedback(frame)
        return frame

    def _render_hud(self, frame: np.ndarray) -> None:
        """Dibuja el panel superior, estado de la FSM, controles y banner de éxito."""
        h, w, _ = frame.shape

        # Barra superior traslúcida oscura (altura 115 px para métricas y buffer)
        cv2.rectangle(frame, (0, 0), (w, 115), (12, 12, 12), -1)
        cv2.line(frame, (0, 115), (w, 115), (0, 220, 255), 2)

        # Fila 1: Título y seña activa
        cv2.putText(
            frame,
            f"MODO {self.training_mode} — SEÑA: {self.current_gesture.upper()}",
            (25, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 240, 255),
            2,
        )
        # Cámara activa
        cv2.putText(
            frame,
            f"CAM: {self.current_camera_index} (Presione '}}' para alternar)",
            (w - 750, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 150),
            2,
        )

        # Fila 2: Estado del cursor activo y Calibración Antropométrica
        cv2.putText(
            frame,
            f"Cursor: {self.cursor_type.value} [C]",
            (25, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 160),
            1,
        )
        scale_tag = "FACE_SCALE" if self.face_detected else "FALLBACK"
        cv2.putText(
            frame,
            f"Escala Facial ({scale_tag}): {int(self.face_scale)}px | R_vis: {self.r_visual}px | R_hit: {self.r_hitbox}px",
            (260, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 220, 255),
            1,
        )

        # Progreso de la trayectoria
        total_pts = len(self.buttons)
        prog_text = f"Punto: {min(self.active_button_idx + 1, total_pts)} / {total_pts}"
        cv2.putText(
            frame,
            prog_text,
            (w - 380, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # Controles disponibles
        cv2.putText(
            frame,
            "[ESP/M]=Modo [A-Z]=Letra [1-9/TAB]=Verbo [G]=Guardar [}]=Cam",
            (w - 650, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (200, 200, 200),
            1,
        )

        # ─── Fila 3: Contador Persistente de Muestras y Buffer de Grabación (0/30) ───
        g_clean = self.current_gesture.strip().lower()
        lbl_count = "por_favor" if g_clean in ("por favor", "por_favor") else self.current_gesture
        sample_count = self.capture_pipeline.get_sample_count(self.training_mode.lower(), lbl_count)
        if sample_count == 0 and g_clean in ("por favor", "por_favor"):
            sample_count = self.capture_pipeline.get_sample_count(self.training_mode.lower(), "Por favor")
        if sample_count == 0:
            sample_count = self.capture_pipeline.get_sample_count(self.training_mode.lower(), g_clean)

        target_goal = getattr(self, "target_samples_per_gesture", 30)
        col_samples = (0, 255, 120) if sample_count >= target_goal else (0, 255, 220)
        cv2.putText(
            frame,
            f"Muestras: {sample_count}/{target_goal}",
            (25, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            col_samples,
            2,
        )

        # Indicador visual de estado del Buffer Temporal (T = 60)
        if self.capture_pipeline.is_recording:
            n_f = len(self.capture_pipeline.frame_buffer)
            pulse = 0.5 + 0.5 * math.sin(time.time() * 8.0)
            rec_color = (0, 0, int(200 + 55 * pulse))
            cv2.circle(frame, (275, 93), 8, rec_color, -1)
            cv2.putText(
                frame,
                f"REC [{n_f}/{self.capture_pipeline.target_frames} frames]",
                (295, 98),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 0, 255),
                2,
            )
        else:
            cv2.putText(
                frame,
                "BUFFER: ESPERANDO BOTÓN 1 (HOLD 0.5s)",
                (275, 98),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (160, 160, 160),
                1,
            )

        # Notificación dinámica de guardado / reinicio
        if self.capture_pipeline.status_message:
            msg_col = (0, 70, 255) if self.capture_pipeline.status_is_error else (0, 255, 120)
            cv2.putText(
                frame,
                self.capture_pipeline.status_message,
                (w - 650, 98),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                msg_col,
                2,
            )

        # ─── Banner Destacado: Advertencia de Cuerpo Completo / Postura de Pie ─
        if getattr(self, "posture_warning", False):
            banner_w = int(w * 0.92)
            banner_h = 56
            bx1 = (w - banner_w) // 2
            by1 = 125

            # Fondo rojo oscuro con borde amarillo reflectante
            overlay_warn = frame.copy()
            cv2.rectangle(overlay_warn, (bx1, by1), (bx1 + banner_w, by1 + banner_h), (0, 0, 190), -1)
            cv2.addWeighted(overlay_warn, 0.85, frame, 0.15, 0, frame)
            cv2.rectangle(frame, (bx1, by1), (bx1 + banner_w, by1 + banner_h), (0, 220, 255), 2)

            # Icono triangular de advertencia
            tri_pt1 = (bx1 + 32, by1 + 14)
            tri_pt2 = (bx1 + 18, by1 + 42)
            tri_pt3 = (bx1 + 46, by1 + 42)
            cv2.drawContours(frame, [np.array([tri_pt1, tri_pt2, tri_pt3])], 0, (0, 220, 255), -1)
            cv2.putText(frame, "!", (bx1 + 29, by1 + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

            warn_msg = "ADVERTENCIA: ESTA SENA REQUIERE CUERPO COMPLETO. PONTE DE PIE O ALEJATE DE LA CAMARA"
            cv2.putText(
                frame,
                warn_msg,
                (bx1 + 58, by1 + 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
            )

        # ─── Banner Destacado: Advertencia Bimanual Obligatoria (FASE 2) ──────
        elif getattr(self, "bimanual_warning", False):
            banner_w = int(w * 0.92)
            banner_h = 56
            bx1 = (w - banner_w) // 2
            by1 = 125

            overlay_warn = frame.copy()
            cv2.rectangle(overlay_warn, (bx1, by1), (bx1 + banner_w, by1 + banner_h), (180, 80, 0), -1)
            cv2.addWeighted(overlay_warn, 0.85, frame, 0.15, 0, frame)
            cv2.rectangle(frame, (bx1, by1), (bx1 + banner_w, by1 + banner_h), (0, 240, 255), 2)

            warn_msg = "FASE 2 'DIAS': BIMANUAL OBLIGATORIA — UBICA AMBAS MANOS EN EL PECHO"
            cv2.putText(
                frame,
                warn_msg,
                (bx1 + 35, by1 + 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (255, 255, 255),
                2,
            )

        # ─── Banner Destacado: Advertencia de Postura Manual Errónea (Queremas) ─
        elif getattr(self, "hand_posture_error", False) and self.hand_posture_error_message:
            banner_w = int(w * 0.92)
            banner_h = 56
            bx1 = (w - banner_w) // 2
            by1 = 125

            overlay_warn = frame.copy()
            cv2.rectangle(overlay_warn, (bx1, by1), (bx1 + banner_w, by1 + banner_h), (0, 50, 190), -1)
            cv2.addWeighted(overlay_warn, 0.85, frame, 0.15, 0, frame)
            cv2.rectangle(frame, (bx1, by1), (bx1 + banner_w, by1 + banner_h), (0, 120, 255), 2)

            tri_pt1 = (bx1 + 32, by1 + 14)
            tri_pt2 = (bx1 + 18, by1 + 42)
            tri_pt3 = (bx1 + 46, by1 + 42)
            cv2.drawContours(frame, [np.array([tri_pt1, tri_pt2, tri_pt3])], 0, (0, 140, 255), -1)
            cv2.putText(frame, "!", (bx1 + 29, by1 + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

            warn_msg = f"POSTURA: {self.hand_posture_error_message.upper()}"
            cv2.putText(
                frame,
                warn_msg,
                (bx1 + 58, by1 + 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.66,
                (255, 255, 255),
                2,
            )

        # ── Notificación de ÉXITO / COMPLETADO ──
        if self.fsm_state == FSMState.SUCCESS:
            box_w, box_h = 600, 110
            x1 = (w - box_w) // 2
            y1 = h // 2 - 55
            cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), (0, 160, 40), -1)
            cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), (255, 255, 255), 3)
            cv2.putText(
                frame,
                f"¡EXCELENTE! GESTO COMPLETADO",
                (x1 + 45, y1 + 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.95,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Cargando siguiente seña...",
                (x1 + 160, y1 + 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (240, 255, 200),
                2,
            )

        elif self.fsm_state == FSMState.TIMEOUT_RESET:
            box_w, box_h = 520, 80
            x1 = (w - box_w) // 2
            y1 = h // 2 - 40
            cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), (0, 40, 180), -1)
            cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), (255, 255, 255), 2)
            cv2.putText(
                frame,
                "¡Tiempo agotado! Reiniciando seña...",
                (x1 + 35, y1 + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

        elif self.fsm_state == FSMState.WAITING_START:
            # Guía animada para el punto inicial
            pulse = 0.5 + 0.5 * math.sin(time.time() * 5.0)
            alpha_color = (0, int(200 + 55 * pulse), 255)
            cv2.putText(
                frame,
                ">> TOCA EL BOTON 1 PARA INICIAR <<",
                (w // 2 - 250, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                alpha_color,
                2,
            )


def main() -> None:
    """Ejecutor interactivo en pantalla completa nativa sin bordes."""
    WINDOW_TITLE = "Traductor Alpha LSC"

    print("\n" + "=" * 80)
    print("  GHOSTPATH TRAINER — MODO ENTRENAMIENTO VISUAL PANTALLA COMPLETA")
    print("=" * 80)

    # 1. Configurar ventana de OpenCV para pantalla completa nativa sin bordes
    cv2.namedWindow(WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # 2. Inicializar cámara con DirectShow (CAP_DSHOW) y resolución
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la cámara (índice {CAMERA_INDEX}, backend CAP_DSHOW).")
        cv2.destroyAllWindows()
        return

    # Validación inmediata del primer frame antes del bucle
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        cv2.destroyAllWindows()
        print(f"[ERROR] No se pudo leer el primer frame de la cámara (índice {CAMERA_INDEX}, backend CAP_DSHOW).")
        print("[DIAGNÓSTICO] Verifica que la cámara no esté en uso por otra app o prueba con CAMERA_INDEX = 1.")
        return

    import mediapipe as mp
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    extractor = HolisticFeatureExtractor()
    trainer = GhostTrainer(cursor_type=CursorType.INDEX_TIP)

    prev_time = time.time()

    try:
        while True:
            cur_time = time.time()
            dt = max(1e-4, cur_time - prev_time)
            prev_time = cur_time

            ret, frame = cap.read()
            if not ret or frame is None:
                print("[ERROR] Fallo al capturar frame.")
                break

            # Efecto espejo
            frame = cv2.flip(frame, 1)

            # Escalar a la resolución nativa de la pantalla completa
            frame_fullscreen = cv2.resize(
                frame,
                (trainer.screen_w, trainer.screen_h),
                interpolation=cv2.INTER_LINEAR,
            )

            # 2.5 Inferencia de MediaPipe Hands (Conversión explícita a RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results_hands = hands_detector.process(rgb_frame)
            rgb_frame.flags.writeable = True

            # 3. Procesar frame con MediaPipe Holistic (Pose + Kinematics)
            results = extractor.process_frame(frame)
            spatial_vec, interaction_points = extractor.extract_spatial_vector(results)

            # Si la toma está activa en el buffer temporal, registrar cuadro y vector (T=60)
            if trainer.capture_pipeline.is_recording:
                trainer.capture_pipeline.record_frame(frame, spatial_vec)

            # 4. Heurística de Mano Dominante / Más Cercana (Single-Hand Targeting)
            cursor_px, active_hand_idx = trainer.resolve_active_hand_cursor(
                results_hands,
                interaction_points,
            )
            trainer.active_hand_idx = active_hand_idx

            # 5. Actualizar lógica de colisión, calibración antropométrica y FSM
            trainer.update(
                cursor_px,
                dt,
                current_frame=frame,
                current_spatial=spatial_vec,
                results=results,
                results_hands=results_hands,
            )

            # 6. Renderizar interfaz GhostPath con esqueleto de manos y botones reactivos
            rendered_frame = trainer.render(
                frame_fullscreen,
                cursor_px,
                results=results,
                results_hands=results_hands,
                active_hand_idx=active_hand_idx,
            )

            cv2.imshow(WINDOW_TITLE, rendered_frame)
            key = cv2.waitKey(1) & 0xFF

            if key != 255:
                if key == ord("q") or key == 27:  # Q o ESC para salir
                    break
                elif key == ord("m") or key == ord("M") or key == ord(" "):  # Tecla M o ESPACIO: Alternar Modo
                    if trainer.training_mode == "VERBOS":
                        trainer.training_mode = "ABECEDARIO"
                        trainer.training_vocabulary = trainer.alphabet_signs
                    else:
                        trainer.training_mode = "VERBOS"
                        trainer.training_vocabulary = trainer.dynamic_signs
                    trainer.current_vocab_idx = 0
                    trainer.current_gesture = trainer.training_vocabulary[0]
                    trainer._load_gesture_buttons(trainer.current_gesture)
                    print(f"[MODO] Cambiado a {trainer.training_mode}")
                elif key == ord("g") or key == ord("G"):  # Tecla de guardado manual de emergencia
                    if trainer.capture_pipeline.is_recording:
                        save_label = "por_favor" if trainer.current_gesture.strip().lower().replace("_", " ") == "por favor" else trainer.current_gesture
                        success = trainer.capture_pipeline.commit(
                            gesture=save_label,
                            category=trainer.training_mode.lower(),
                            is_emergency=True
                        )
                        if success:
                            print(f"[EMERGENCIA] Guardado manual forzado ejecutado para '{save_label}'")
                            trainer.tts.speak("Muestra guardada manualmente")
                    else:
                        print("[EMERGENCIA] No hay toma activa en el buffer para guardar.")
                elif key == ord("c") or key == ord("C"):  # Tecla C: alternar cursor
                    new_cursor = trainer.toggle_cursor()
                    print(f"[CURSOR] Modo de cursor cambiado a: {new_cursor.value}")
                elif key == 9:  # TAB: Navegar Verbos
                    if trainer.training_mode == "VERBOS":
                        trainer.current_vocab_idx = (trainer.current_vocab_idx + 1) % len(trainer.training_vocabulary)
                        trainer.current_gesture = trainer.training_vocabulary[trainer.current_vocab_idx]
                        trainer._load_gesture_buttons(trainer.current_gesture)
                elif key == ord("}"):  # Cambio de cámara en caliente (Hot-Swap)
                    print(f"[HOT-SWAP] Cambiando de cámara...")
                    cap.release()
                    trainer.current_camera_index += 1
                    cap = cv2.VideoCapture(trainer.current_camera_index, cv2.CAP_DSHOW)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
                    test_ret, _ = cap.read()
                    if not test_ret:
                        print(f"[HOT-SWAP] Cámara {trainer.current_camera_index} no disponible. Volviendo a índice 0.")
                        trainer.current_camera_index = 0
                        cap.release()
                        cap = cv2.VideoCapture(trainer.current_camera_index, cv2.CAP_DSHOW)
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
                        cap.read()  # Dummy read
                else:
                    # Selección Inteligente de Letras y Verbos
                    key_char = chr(key).lower()
                    if trainer.training_mode == "ABECEDARIO" and key_char in trainer.alphabet_signs:
                        trainer.current_vocab_idx = trainer.alphabet_signs.index(key_char)
                        trainer.current_gesture = key_char
                        trainer._load_gesture_buttons(trainer.current_gesture)
                    elif trainer.training_mode == "VERBOS" and key_char in [str(i) for i in range(1, 10)]:
                        idx = int(key_char) - 1
                        if idx < len(trainer.dynamic_signs):
                            trainer.current_vocab_idx = idx
                            trainer.current_gesture = trainer.dynamic_signs[idx]
                            trainer._load_gesture_buttons(trainer.current_gesture)

    finally:
        hands_detector.close()
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
        trainer.tts.stop()
        print("[INFO] GhostTrainer finalizado correctamente.")


if __name__ == "__main__":
    main()
