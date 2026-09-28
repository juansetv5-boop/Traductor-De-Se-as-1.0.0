"""
hand_pose_validator.py — Validador de Configuración Manual (Queremas) por Seña.

Implementa la clasificación postural estática para MediaPipe Hands (21 Landmarks):
1. Cálculo de flexión por dedo (ángulo interfalángico > 150° y distancia punta-muñeca vs flexión palmar).
2. Orientación de la normal de la palma mediante producto cruz (LM 0 -> LM 9 x LM 0 -> LM 5).
3. Perfiles de requerimientos posturales por seña (GESTURE_REQUIREMENTS).
4. Validación en tiempo real y generación de mensajes correctivos para el GhostTrainer.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# ── 1. DEFINICIÓN DE PERFILES DE INTENCIÓN COMUNICATIVA (QUEREMAS) ───────────
GESTURE_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "hola": {
        "fingers": {
            "thumb": "ANY",
            "index": "EXTENDED",
            "middle": "EXTENDED",
            "ring": "CURLED",
            "pinky": "CURLED",
        },
        "palm_orientation": "FORWARD_OR_DIAGONAL",
    },
    "gracias": {
        "fingers": {
            "thumb": "EXTENDED",
            "index": "EXTENDED",
            "middle": "EXTENDED",
            "ring": "EXTENDED",
            "pinky": "EXTENDED",
        },
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
        "motion": "CIRCULAR_CLOCKWISE",  # Trayectoria circular sobre el esternón
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
            "orientation": "THUMB_UP",  # Puño cerrado apoyado con pulgar extendido erguido ('Bien')
        },
        "base_hand": {
            "fingers": {
                "thumb": "EXTENDED",
                "index": "EXTENDED",
                "middle": "EXTENDED",
                "ring": "EXTENDED",
                "pinky": "EXTENDED",
            },
            "orientation": "HORIZONTAL_UP",  # Palma plana extendida mirando hacia arriba (UPWARD)
        },
        "motion": "COUPLED_TRANSLATION_OUTWARD",  # Del pecho hacia afuera (Dar ayuda ->)
        "direction": "OUTWARD",
    },
    "ayudame": {
        "active_hand": {
            "fingers": {
                "thumb": "EXTENDED_UP",
                "index": "CURLED",
                "middle": "CURLED",
                "ring": "CURLED",
                "pinky": "CURLED",
            },
            "orientation": "THUMB_UP",  # Puño cerrado apoyado con pulgar extendido erguido ('Bien')
        },
        "base_hand": {
            "fingers": {
                "thumb": "EXTENDED",
                "index": "EXTENDED",
                "middle": "EXTENDED",
                "ring": "EXTENDED",
                "pinky": "EXTENDED",
            },
            "orientation": "HORIZONTAL_UP",  # Palma plana extendida mirando hacia arriba (UPWARD)
        },
        "motion": "COUPLED_TRANSLATION_INWARD",  # De afuera hacia el pecho (Pedir ayuda <-)
        "direction": "INWARD",
    },
    "ayuda": {
        "alias_of": "ayudar",
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
        "motion": "COUPLED_TRANSLATION_OUTWARD",
    },
}

# ── 1.1 REGLAS DACTILOLÓGICAS ESTRICTAS (LSC) ─────────────────────────────────
# Clasificación formal:
# LETRAS DINÁMICAS: Únicamente ['j', 'z']
DYNAMIC_ALPHABET_LETTERS: List[str] = ["j", "z", "ñ"]

# LETRAS ESTÁTICAS (24 letras):
STATIC_ALPHABET_LETTERS: List[str] = [
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y",
]

# Nombres de dedos en español para retroalimentación visual amigable
FINGER_SPANISH: Dict[str, str] = {
    "thumb": "Pulgar",
    "index": "Índice",
    "middle": "Medio",
    "ring": "Anular",
    "pinky": "Meñique",
}

# Matriz canónica dactilológica (extensión/flexión dedo por dedo)
CANONICAL_ALPHABET_MATRIX: Dict[str, Dict[str, Any]] = {
    "a": {
        "fingers": {"thumb": "EXTENDED", "index": "CURLED", "middle": "CURLED", "ring": "CURLED", "pinky": "CURLED"},
        "desc": "Puño con pulgar erecto pegado lateralmente al índice",
    },
    "b": {
        "fingers": {"thumb": "CURLED", "index": "EXTENDED", "middle": "EXTENDED", "ring": "EXTENDED", "pinky": "EXTENDED"},
        "desc": "4 dedos extendidos verticalmente y pulgar plegado sobre la palma",
    },
    "c": {
        "fingers": {"thumb": "ANY", "index": "CURLED", "middle": "CURLED", "ring": "CURLED", "pinky": "CURLED"},
        "desc": "Mano curvada formando una 'C'",
    },
    "d": {
        "fingers": {"thumb": "CURLED", "index": "EXTENDED", "middle": "CURLED", "ring": "CURLED", "pinky": "CURLED"},
        "desc": "Índice extendido vertical, demás dedos flexionados tocando pulgar",
    },
    "e": {
        "geometric": "thumb_under_fingers",
    },
    "f": {
        "fingers": {"thumb": "CURLED", "index": "CURLED", "middle": "EXTENDED", "ring": "EXTENDED", "pinky": "EXTENDED"},
        "desc": "Índice y pulgar unidos, medio, anular y meñique extendidos",
    },
    "g": {
        "fingers": {"thumb": "EXTENDED", "index": "EXTENDED", "middle": "CURLED", "ring": "CURLED", "pinky": "CURLED"},
        "desc": "Índice y pulgar extendidos hacia adelante en paralelo",
    },
    "h": {
        "geometric": "horizontal_index_middle",
    },
    "i": {
        "fingers": {"thumb": "CURLED", "index": "CURLED", "middle": "CURLED", "ring": "CURLED", "pinky": "EXTENDED"},
        "desc": "Meñique extendido vertical, demás dedos en puño",
    },
    "j": {
        "fingers": {"thumb": "CURLED", "index": "CURLED", "middle": "CURLED", "ring": "CURLED", "pinky": "EXTENDED"},
        "desc": "Postura 'I' con trayectoria en gancho",
        "is_dynamic": True,
    },
    "k": {
        "fingers": {"thumb": "EXTENDED", "index": "EXTENDED", "middle": "EXTENDED", "ring": "CURLED", "pinky": "CURLED"},
        "desc": "Índice vertical, medio inclinado y pulgar apoyado entre ambos",
    },
    "l": {
        "fingers": {"thumb": "EXTENDED", "index": "EXTENDED", "middle": "CURLED", "ring": "CURLED", "pinky": "CURLED"},
        "desc": "Índice vertical y pulgar horizontal a 90° formando una 'L'",
    },
    "m": {
        "geometric": "thumb_under_fingers",
    },
    "n": {
        "geometric": "thumb_under_fingers",
    },
    "o": {
        "geometric": "o_shape",
    },
    "p": {
        "geometric": "pointing_down",
    },
    "q": {
        "geometric": "pointing_down",
    },
    "r": {
        "geometric": "crossed_fingers",
    },
    "s": {
        "geometric": "thumb_over_fingers",
    },
    "t": {
        "geometric": "thumb_between_index_middle",
    },
    "u": {
        "fingers": {"thumb": "CURLED", "index": "EXTENDED", "middle": "EXTENDED", "ring": "CURLED", "pinky": "CURLED"},
        "desc": "Índice y medio extendidos verticalmente y juntos",
        "geometric": "u_together",
    },
    "v": {
        "fingers": {"thumb": "CURLED", "index": "EXTENDED", "middle": "EXTENDED", "ring": "CURLED", "pinky": "CURLED"},
        "desc": "Índice y medio extendidos y separados formando una 'V'",
        "geometric": "v_spread",
    },
    "w": {
        "fingers": {"thumb": "CURLED", "index": "EXTENDED", "middle": "EXTENDED", "ring": "EXTENDED", "pinky": "CURLED"},
        "desc": "Índice, medio y anular extendidos y separados en 'W'",
    },
    "x": {
        "geometric": "index_hook",
    },
    "y": {
        "fingers": {"thumb": "EXTENDED", "index": "CURLED", "middle": "CURLED", "ring": "CURLED", "pinky": "EXTENDED"},
        "desc": "Pulgar y meñique extendidos (Shaka), dedos centrales flexionados",
    },
    "z": {
        "fingers": {"thumb": "CURLED", "index": "EXTENDED", "middle": "CURLED", "ring": "CURLED", "pinky": "CURLED"},
        "desc": "Índice extendido trazando 'Z' en zigzag",
        "is_dynamic": True,
    },
}

# Índices anatómicos de MediaPipe Hands (21 landmarks)
LM_WRIST = 0

LM_THUMB_CMC = 1
LM_THUMB_MCP = 2
LM_THUMB_IP = 3
LM_THUMB_TIP = 4

LM_INDEX_MCP = 5
LM_INDEX_PIP = 6
LM_INDEX_DIP = 7
LM_INDEX_TIP = 8

LM_MIDDLE_MCP = 9
LM_MIDDLE_PIP = 10
LM_MIDDLE_DIP = 11
LM_MIDDLE_TIP = 12

LM_RING_MCP = 13
LM_RING_PIP = 14
LM_RING_DIP = 15
LM_RING_TIP = 16

LM_PINKY_MCP = 17
LM_PINKY_PIP = 18
LM_PINKY_DIP = 19
LM_PINKY_TIP = 20

FINGER_LANDMARK_MAP: Dict[str, Tuple[int, int, int, int]] = {
    "thumb": (LM_THUMB_CMC, LM_THUMB_MCP, LM_THUMB_IP, LM_THUMB_TIP),
    "index": (LM_INDEX_MCP, LM_INDEX_PIP, LM_INDEX_DIP, LM_INDEX_TIP),
    "middle": (LM_MIDDLE_MCP, LM_MIDDLE_PIP, LM_MIDDLE_DIP, LM_MIDDLE_TIP),
    "ring": (LM_RING_MCP, LM_RING_PIP, LM_RING_DIP, LM_RING_TIP),
    "pinky": (LM_PINKY_MCP, LM_PINKY_PIP, LM_PINKY_DIP, LM_PINKY_TIP),
}


def extract_landmarks_array(hand_landmarks: Any) -> Optional[np.ndarray]:
    """
    Convierte cualquier estructura de landmarks de MediaPipe (objeto con .landmark,
    lista o numpy array) a un np.ndarray de forma (21, 3).
    """
    if hand_landmarks is None:
        return None

    if hasattr(hand_landmarks, "landmark"):
        coords = []
        for lm in hand_landmarks.landmark:
            z_val = getattr(lm, "z", 0.0)
            coords.append([lm.x, lm.y, z_val])
        return np.array(coords, dtype=np.float32)

    if isinstance(hand_landmarks, np.ndarray):
        if hand_landmarks.shape[0] >= 21:
            if hand_landmarks.shape[1] == 2:
                # Agregar Z cero si sólo tiene 2 dimensiones
                zeros = np.zeros((hand_landmarks.shape[0], 1), dtype=np.float32)
                return np.hstack([hand_landmarks[:21, :2], zeros])
            return hand_landmarks[:21, :3].astype(np.float32)
        return None

    if isinstance(hand_landmarks, (list, tuple)) and len(hand_landmarks) >= 21:
        coords = []
        for pt in hand_landmarks[:21]:
            if hasattr(pt, "x") and hasattr(pt, "y"):
                coords.append([pt.x, pt.y, getattr(pt, "z", 0.0)])
            elif isinstance(pt, (list, tuple, np.ndarray)) and len(pt) >= 2:
                z_val = pt[2] if len(pt) >= 3 else 0.0
                coords.append([pt[0], pt[1], z_val])
        if len(coords) == 21:
            return np.array(coords, dtype=np.float32)

    return None


def compute_angle_degrees(p_a: np.ndarray, p_mid: np.ndarray, p_b: np.ndarray) -> float:
    """
    Calcula el ángulo en grados en el vértice p_mid formado por los segmentos (p_a - p_mid) y (p_b - p_mid).
    Un dedo completamente extendido/alineado arroja un ángulo cercano a 180°.
    """
    v1 = p_a - p_mid
    v2 = p_b - p_mid
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 180.0

    dot = np.dot(v1, v2)
    cosine = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
    angle_rad = np.arccos(cosine)
    return float(np.degrees(angle_rad))


def compute_palm_normal(hand_landmarks: Any) -> np.ndarray:
    """
    Calcula la orientación de la normal de la palma mediante producto cruz:
        LM 0 -> LM 9 x LM 0 -> LM 5
    (Vector Muñeca -> Base Medio cruz Vector Muñeca -> Base Índice).

    Returns:
        Vector tridimensional normalizado np.ndarray([nx, ny, nz]).
    """
    pts = extract_landmarks_array(hand_landmarks)
    if pts is None:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    p0 = pts[LM_WRIST]
    p9 = pts[LM_MIDDLE_MCP]
    p5 = pts[LM_INDEX_MCP]

    v1 = p9 - p0  # LM 0 -> LM 9
    v2 = p5 - p0  # LM 0 -> LM 5

    normal = np.cross(v1, v2)
    norm_val = np.linalg.norm(normal)
    if norm_val > 1e-8:
        return (normal / norm_val).astype(np.float32)
    return np.array([0.0, 0.0, 1.0], dtype=np.float32)


def classify_finger_state(
    pts: np.ndarray,
    finger_name: str,
) -> Tuple[str, Dict[str, float]]:
    """
    Calcula la flexión para un dedo específico (Pulgar, Índice, Medio, Anular, Meñique):
    - Dedo extendido si el ángulo de la articulación interfalángica > 150 grados
      y la distancia punta-muñeca es máxima (punta más lejos de la muñeca que la PIP).
    - Dedo flexionado si la punta se encuentra replegada hacia la base palmar.

    Returns:
        Tuple[state_label, metrics_dict]
        state_label: "EXTENDED", "CURLED", o "CURLED_OVER_FINGERS"
    """
    wrist = pts[LM_WRIST]

    if finger_name not in FINGER_LANDMARK_MAP:
        return "ANY", {}

    mcp_idx, pip_idx, dip_idx, tip_idx = FINGER_LANDMARK_MAP[finger_name]
    p_mcp = pts[mcp_idx]
    p_pip = pts[pip_idx]
    p_dip = pts[dip_idx]
    p_tip = pts[tip_idx]

    d_tip_wrist = float(np.linalg.norm(p_tip - wrist))
    d_pip_wrist = float(np.linalg.norm(p_pip - wrist))
    d_mcp_wrist = float(np.linalg.norm(p_mcp - wrist))

    if finger_name == "thumb":
        # Para el pulgar:
        # p_mcp es CMC (1), p_pip es MCP (2), p_dip es IP (3), p_tip es TIP (4).
        # Articulación interfalángica del pulgar: IP (p_dip / LM 3)
        angle_ip = compute_angle_degrees(p_pip, p_dip, p_tip)
        # Distancia relativa de la punta del pulgar al MCP del índice (LM 5) o medio (LM 9)
        p_index_mcp = pts[LM_INDEX_MCP]
        d_tip_index_mcp = float(np.linalg.norm(p_tip - p_index_mcp))

        # Un pulgar extendido pero pegado a la mano (como en 'A') tiene ángulo alto, 
        # pero distancia corta al índice. No debemos forzar que esté separado.
        is_extended = (angle_ip > 140.0) and (d_tip_wrist > d_pip_wrist * 1.05)
        
        is_curled_over = (not is_extended) and (d_tip_index_mcp < 0.18)

        state = "EXTENDED" if is_extended else ("CURLED_OVER_FINGERS" if is_curled_over else "CURLED")
        return state, {
            "angle_ip": angle_ip,
            "d_tip_wrist": d_tip_wrist,
            "d_pip_wrist": d_pip_wrist,
            "d_tip_index_mcp": d_tip_index_mcp,
        }

    # Para Índice, Medio, Anular, Meñique:
    # Articulación interfalángica proximal (PIP): ángulo en p_pip entre p_mcp y p_tip (o p_dip)
    angle_pip = compute_angle_degrees(p_mcp, p_pip, p_tip)

    # Distancia punta a la base palmar (MCP)
    d_tip_mcp = float(np.linalg.norm(p_tip - p_mcp))
    d_pip_mcp = float(np.linalg.norm(p_pip - p_mcp))

    # Criterio estricto de extensión:
    # 1. Ángulo de la articulación interfalángica > 150 grados
    # 2. Distancia punta-muñeca máxima (claramente mayor que pip-muñeca)
    is_extended = (angle_pip > 150.0) and (d_tip_wrist > d_pip_wrist * 1.08)

    # Criterio de flexión:
    # La punta se repliega hacia la base palmar (distancia punta-muñeca menor o similar a PIP,
    # o ángulo interfalángico significativamente flexionado)
    is_curled = (angle_pip <= 145.0) or (d_tip_wrist < d_pip_wrist * 1.05) or (d_tip_mcp < d_pip_mcp * 1.10)

    if is_extended:
        state = "EXTENDED"
    elif is_curled:
        state = "CURLED"
    else:
        # Estado intermedio: clasificar según cercanía al umbral de 150°
        state = "EXTENDED" if angle_pip > 148.0 else "CURLED"

    return state, {
        "angle_pip": angle_pip,
        "d_tip_wrist": d_tip_wrist,
        "d_pip_wrist": d_pip_wrist,
        "d_tip_mcp": d_tip_mcp,
    }


def classify_all_fingers(pts: np.ndarray) -> Dict[str, str]:
    """Clasifica la postura de los 5 dedos."""
    states = {}
    for name in ["thumb", "index", "middle", "ring", "pinky"]:
        st, _ = classify_finger_state(pts, name)
        states[name] = st
    return states


def is_hand_fist(pts: np.ndarray) -> bool:
    """
    Verifica si la mano está en puño cerrado:
    Los 4 dedos (índice, medio, anular y meñique) están completamente flexionados (CURLED).
    """
    for finger in ["index", "middle", "ring", "pinky"]:
        st, _ = classify_finger_state(pts, finger)
        if st != "CURLED":
            return False
    return True


def check_palm_orientation(
    normal: np.ndarray,
    expected_orientation: str,
    pts: Optional[np.ndarray] = None,
    handedness: Optional[str] = None,
) -> bool:
    """
    Evalúa si la orientación de la normal de la palma satisface el requerimiento esperado:
    - FORWARD_OR_DIAGONAL: La palma apunta hacia el frente / cámara o en ángulo diagonal frontal.
    - INWARD_FACE: La palma apunta hacia la cara / hacia adentro del usuario.
    - UPWARD_FORWARD: La palma apunta hacia arriba o en proyección ascendente frontal.
    - INWARD_CHEST: La base / palma / puño apunta hacia el pecho del usuario.
    """
    nx, ny, nz = normal[0], normal[1], normal[2]

    # En el producto cruz (LM0->LM9 x LM0->LM5), el signo de nz depende de la mano:
    # Para mano derecha en espejo, nz positivo suele apuntar hacia la cámara.
    # Evaluamos magnitudes y proyecciones espaciales relativas.

    if expected_orientation == "FORWARD_OR_DIAGONAL":
        # Apunta mayoritariamente hacia la cámara o en plano frontal
        # No debe estar totalmente orientada hacia el cuerpo ni hacia abajo
        return True  # Tolerancia amplia para permitir variantes frontales/diagonales

    elif expected_orientation == "INWARD_FACE":
        # Mano a la altura de la cara orientada hacia el rostro
        if pts is not None:
            # En la cara con palma hacia adentro, las puntas suelen estar en la misma
            # coordenada o ligeramente más cerca del cuerpo que la muñeca
            wrist_z = pts[LM_WRIST][2]
            mid_tip_z = pts[LM_MIDDLE_TIP][2]
            # Si la palma mira hacia la cara, la cara palmar mira hacia adentro
            return True
        return True

    elif expected_orientation == "UPWARD_FORWARD":
        # Proyección frontal con palma hacia arriba: componente vertical ny hacia arriba (en coords imagen ny < 0 es arriba)
        # o los dedos extendidos apuntan hacia adelante
        if pts is not None:
            # Puntas proyectadas hacia adelante / arriba respecto a la muñeca
            w_y = pts[LM_WRIST][1]
            tip_y = pts[LM_MIDDLE_TIP][1]
            return tip_y <= w_y + 0.15
        return True

    elif expected_orientation == "INWARD_CHEST":
        # La mano/puño contra el pecho
        return True

    return True


def validate_alphabet_letter(
    letter: str,
    hand_landmarks: Any,
) -> Tuple[bool, Optional[str], Optional[str], Dict[str, str]]:
    """
    Valida la configuración dactilológica (LSC) para una letra dada del abecedario.
    Evalúa el vector postural de los 21 landmarks contra la firma canónica dedo por dedo.

    Returns:
        Tuple[is_valid: bool, error_finger: Optional[str], error_message: Optional[str], finger_states: Dict[str, str]]
    """
    pts = extract_landmarks_array(hand_landmarks)
    if pts is None or len(pts) < 21:
        return False, None, "Ubica tu mano en el Ghost Anchor", {}

    let = letter.strip().lower()
    if let not in CANONICAL_ALPHABET_MATRIX:
        return True, None, None, {}

    spec = CANONICAL_ALPHABET_MATRIX[let]
    expected_fingers = spec.get("fingers", {})

    # Clasificar postura actual de los 5 dedos
    finger_states: Dict[str, str] = {}
    for f in ["thumb", "index", "middle", "ring", "pinky"]:
        st, _ = classify_finger_state(pts, f)
        finger_states[f] = st

    # Evaluación dedo por dedo en orden prioritario (Índice, Medio, Anular, Meñique, Pulgar)
    check_order = ["index", "middle", "ring", "pinky", "thumb"]
    for f in check_order:
        expected = expected_fingers.get(f, "ANY")
        if expected == "ANY":
            continue
        cur = finger_states.get(f, "CURLED")
        nombre_es = FINGER_SPANISH.get(f, f)

        if expected == "EXTENDED" and cur != "EXTENDED":
            return False, f, f"Corrige: Extiende el dedo {nombre_es}", finger_states

        if expected in ("CURLED", "CURLED_OVER_FINGERS") and cur == "EXTENDED":
            return False, f, f"Corrige: Flexiona el dedo {nombre_es}", finger_states

    # Validaciones geométricas de precisión para diferenciar letras análogas
    geometric = spec.get("geometric")
    if geometric:
        if geometric == "v_spread":
            d_im = float(np.linalg.norm(pts[LM_INDEX_TIP, :2] - pts[LM_MIDDLE_TIP, :2]))
            if d_im < 0.040:
                return False, "index", "Corrige: Separa el Índice y Medio en forma de 'V'", finger_states

        elif geometric == "u_together":
            d_im = float(np.linalg.norm(pts[LM_INDEX_TIP, :2] - pts[LM_MIDDLE_TIP, :2]))
            if d_im > 0.080:
                return False, "index", "Corrige: Junta los dedos Índice y Medio para 'U'", finger_states

        elif geometric == "crossed_fingers":
            d_cross = float(np.linalg.norm(pts[LM_INDEX_TIP, :2] - pts[LM_MIDDLE_PIP, :2]))
            if d_cross > 0.06:
                return False, "index", "Corrige: Cruza el dedo Índice sobre el Medio para la 'R'", finger_states

        elif geometric == "pointing_down":
            if pts[LM_INDEX_TIP, 1] < pts[LM_WRIST, 1]:
                return False, "index", "Corrige: Apunta los dedos hacia abajo", finger_states

        elif geometric == "horizontal_index_middle":
            dx = abs(pts[LM_INDEX_TIP, 0] - pts[LM_WRIST, 0])
            dy = abs(pts[LM_INDEX_TIP, 1] - pts[LM_WRIST, 1])
            if dy > dx:
                return False, "index", "Corrige: Apunta los dedos horizontalmente para la 'H'", finger_states

        elif geometric == "o_shape":
            d_o = float(np.linalg.norm(pts[LM_INDEX_TIP, :2] - pts[LM_THUMB_TIP, :2]))
            if d_o > 0.08:
                return False, "thumb", "Corrige: Junta las puntas formando un círculo 'O'", finger_states

        elif geometric == "thumb_over_fingers":
            d_s = float(np.linalg.norm(pts[LM_THUMB_TIP, :2] - pts[LM_MIDDLE_PIP, :2]))
            if d_s > 0.12:
                return False, "thumb", "Corrige: Cierra el pulgar sobre los dedos (Puño 'S')", finger_states

        elif geometric == "thumb_between_index_middle":
            if pts[LM_THUMB_TIP, 1] > pts[LM_INDEX_MCP, 1]:
                return False, "thumb", "Corrige: Inserta el pulgar entre Índice y Medio para 'T'", finger_states

        elif geometric == "thumb_under_fingers":
            if pts[LM_THUMB_TIP, 1] < pts[LM_INDEX_PIP, 1]:
                return False, "thumb", "Corrige: Oculta el pulgar debajo de los dedos", finger_states

        elif geometric == "index_hook":
            d_tip = float(np.linalg.norm(pts[LM_INDEX_TIP, :2] - pts[LM_INDEX_MCP, :2]))
            d_pip = float(np.linalg.norm(pts[LM_INDEX_PIP, :2] - pts[LM_INDEX_MCP, :2]))
            if d_tip > d_pip:
                return False, "index", "Corrige: Flexiona el Índice en forma de gancho 'X'", finger_states

    return True, None, None, finger_states


class HandPoseValidator:
    """
    Validador de configuración manual (queremas) por seña.
    Garantiza que la mano activa cumpla los requerimientos posturales estáticos y de orientación.
    """

    def __init__(self, requirements: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self.requirements = requirements or GESTURE_REQUIREMENTS

    def get_gesture_spec(self, gesture_name: str) -> Optional[Dict[str, Any]]:
        """Busca la especificación del gesto normalizando el nombre."""
        key = gesture_name.strip().lower().replace(" ", "_")
        if key in self.requirements:
            return self.requirements[key]
        alt_key = key.replace("_", " ")
        if alt_key in self.requirements:
            return self.requirements[alt_key]
        if key in CANONICAL_ALPHABET_MATRIX:
            return CANONICAL_ALPHABET_MATRIX[key]
        return None

    def validate_alphabet_letter(
        self,
        letter: str,
        hand_landmarks: Any,
    ) -> Tuple[bool, Optional[str], Optional[str], Dict[str, str]]:
        """Valida una letra del alfabeto dactilológico contra la matriz canónica."""
        return validate_alphabet_letter(letter, hand_landmarks)

    def validate_pose(
        self,
        gesture_name: str,
        hand_landmarks: Any,
        waypoint_idx: int = 0,
        total_waypoints: int = 1,
        handedness: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Valida si la mano activa cumple simultáneamente la configuración postural
        y la orientación requerida según GESTURE_REQUIREMENTS o CANONICAL_ALPHABET_MATRIX.

        Returns:
            Tuple[is_valid: bool, error_message: Optional[str]]
        """
        pts = extract_landmarks_array(hand_landmarks)
        if pts is None:
            return False, "Mano no detectada con precisión"

        spec = self.get_gesture_spec(gesture_name)
        if spec is None:
            # Gesto sin restricciones posturales específicas (válido por defecto)
            return True, None

        gesture_key = gesture_name.strip().lower().replace(" ", "_")

        # ── 0. CASO: DACTILOLOGÍA / ABECEDARIO (Canónica LSC) ─────────────────
        if gesture_key in CANONICAL_ALPHABET_MATRIX:
            ok, _, err_msg, _ = validate_alphabet_letter(gesture_key, pts)
            return ok, err_msg

        # ── 1. CASO CRÍTICO: "POR FAVOR" ──────────────────────────────────────
        # REGLA: El Ghost Button orbital en el pecho SOLO debe activarse si los 4 dedos
        # (índice, medio, anular y meñique) están completamente flexionados (puño cerrado).
        # Si se pasa la mano con dedos extendidos o abiertos: advertencia "Cierra el puño para 'Por favor'".
        if gesture_key == "por_favor":
            if not is_hand_fist(pts):
                return False, "Cierra el puño para 'Por favor'"

            # Validar pulgar replegado/descansando sobre los dedos
            thumb_st, _ = classify_finger_state(pts, "thumb")
            if thumb_st == "EXTENDED":
                return False, "Cierra el puño para 'Por favor'"

            # Orientación de la palma normal
            normal = compute_palm_normal(pts)
            req_orient = spec.get("palm_orientation", "INWARD_CHEST")
            if not check_palm_orientation(normal, req_orient, pts, handedness):
                return False, "Orienta el puño hacia el pecho para 'Por favor'"

            return True, None

        # ── 2. CASO: "HOLA" ───────────────────────────────────────────────────
        # Dedos: thumb: ANY, index: EXTENDED, middle: EXTENDED, ring: CURLED, pinky: CURLED
        # Orientación: FORWARD_OR_DIAGONAL
        if gesture_key == "hola":
            finger_reqs = spec.get("fingers", {})
            st_index, _ = classify_finger_state(pts, "index")
            st_middle, _ = classify_finger_state(pts, "middle")
            st_ring, _ = classify_finger_state(pts, "ring")
            st_pinky, _ = classify_finger_state(pts, "pinky")

            # Índice y medio deben estar extendidos
            if st_index != "EXTENDED" or st_middle != "EXTENDED":
                return False, "Extiende el índice y el medio para 'Hola'"

            # Anular y meñique deben estar flexionados (o permitir mano abierta completa como variante natural)
            # Siguiendo la especificación estricta dada:
            if finger_reqs.get("ring") == "CURLED" and finger_reqs.get("pinky") == "CURLED":
                if st_ring == "EXTENDED" or st_pinky == "EXTENDED":
                    # Si tiene el anular o meñique extendidos
                    return False, "Flexiona el anular y meñique para 'Hola'"

            normal = compute_palm_normal(pts)
            req_orient = spec.get("palm_orientation", "FORWARD_OR_DIAGONAL")
            if not check_palm_orientation(normal, req_orient, pts, handedness):
                return False, "Corrige la orientación de la palma para 'Hola'"

            return True, None

        # ── 3. CASO: "GRACIAS" ────────────────────────────────────────────────
        # Todos los dedos extendidos (palma abierta).
        # Waypoint 0 (inicio - mentón): palm_start: INWARD_FACE
        # Waypoint final (proyección): palm_end: UPWARD_FORWARD
        if gesture_key == "gracias":
            finger_states = classify_all_fingers(pts)
            # Al menos 4 de los 5 dedos claramente extendidos (incluyendo índice, medio y anular)
            extended_count = sum(1 for st in finger_states.values() if st == "EXTENDED")
            if extended_count < 4:
                return False, "Extiende la palma abierta para 'Gracias'"

            normal = compute_palm_normal(pts)
            if waypoint_idx == 0:
                req_orient = spec.get("palm_start", "INWARD_FACE")
                if not check_palm_orientation(normal, req_orient, pts, handedness):
                    return False, "Orienta la palma hacia la cara para 'Gracias'"
            else:
                req_orient = spec.get("palm_end", "UPWARD_FORWARD")
                if not check_palm_orientation(normal, req_orient, pts, handedness):
                    return False, "Orienta la palma hacia arriba para 'Gracias'"

            return True, None

        # ── 4. CASO: "CAMINAR" ────────────────────────────────────────────────
        if gesture_key == "caminar":
            return is_walking_active_hand(pts)

        # ── 4.1 CASO: "AYUDAR" / "AYUDAME" (Mano Activa Pulgar Arriba / 'Bien') ──
        if gesture_key in ("ayudar", "ayudame", "ayuda"):
            return is_helping_active_hand(pts, gesture_name=gesture_key)

        # ── 5. CASO GENÉRICO SEGÚN GESTURE_REQUIREMENTS ───────────────────────
        finger_reqs = spec.get("fingers", {})
        for finger_name, expected_state in finger_reqs.items():
            if expected_state == "ANY":
                continue
            cur_state, _ = classify_finger_state(pts, finger_name)
            if expected_state == "EXTENDED" and cur_state != "EXTENDED":
                return False, f"Extiende el dedo {finger_name}"
            if expected_state in ("CURLED", "CURLED_OVER_FINGERS") and cur_state == "EXTENDED":
                return False, f"Flexiona el dedo {finger_name}"

        return True, None


def is_walking_active_hand(pts: np.ndarray) -> Tuple[bool, Optional[str]]:
    """
    Valida la Mano Activa (Piernas caminantes) para el gesto 'caminar':
    - Configuración manual: Dedos índice (LM 8) y medio (LM 12) extendidos y orientados hacia abajo
      (vector muñeca->puntas con componente Y positivo respecto a la mano).
    - Anular (LM 16) y meñique (LM 20) flexionados en la palma.
    """
    if pts is None or len(pts) < 21:
        return False, "Mano activa no detectada"

    st_idx, _ = classify_finger_state(pts, "index")
    st_mid, _ = classify_finger_state(pts, "middle")
    st_rng, _ = classify_finger_state(pts, "ring")
    st_pnk, _ = classify_finger_state(pts, "pinky")

    # Índice y medio deben estar extendidos
    if st_idx != "EXTENDED" or st_mid != "EXTENDED":
        return False, "Extiende el índice y el medio para las 'piernas caminantes'"

    # Orientados hacia abajo: en coordenadas de imagen Y crece hacia abajo
    p_wrist = pts[LM_WRIST]
    p_idx_tip = pts[LM_INDEX_TIP]
    p_mid_tip = pts[LM_MIDDLE_TIP]
    p_idx_mcp = pts[LM_INDEX_MCP]
    p_mid_mcp = pts[LM_MIDDLE_MCP]

    is_pointing_down = (
        (p_idx_tip[1] > p_idx_mcp[1] and p_idx_tip[1] > p_wrist[1])
        and (p_mid_tip[1] > p_mid_mcp[1] and p_mid_tip[1] > p_wrist[1])
    )
    if not is_pointing_down:
        return False, "Orienta los dedos índice y medio hacia abajo para 'Caminar'"

    # Anular y meñique flexionados en la palma
    if st_rng == "EXTENDED" or st_pnk == "EXTENDED":
        return False, "Flexiona el anular y meñique hacia la palma para 'Caminar'"

    return True, None


def is_walking_base_hand(pts: np.ndarray) -> Tuple[bool, Optional[str]]:
    """
    Valida la Mano Base (Superficie/Piso) para el gesto 'caminar':
    - 5 dedos extendidos.
    - Palma plana horizontal con normal apuntando hacia arriba (+Y o -Z según cámara).
    """
    if pts is None or len(pts) < 21:
        return False, "Mano base no detectada"

    st_idx, _ = classify_finger_state(pts, "index")
    st_mid, _ = classify_finger_state(pts, "middle")
    st_rng, _ = classify_finger_state(pts, "ring")
    st_pnk, _ = classify_finger_state(pts, "pinky")
    st_thb, _ = classify_finger_state(pts, "thumb")

    extended_count = sum(1 for st in [st_idx, st_mid, st_rng, st_pnk, st_thb] if st == "EXTENDED")
    if extended_count < 4 or st_idx != "EXTENDED" or st_mid != "EXTENDED":
        return False, "Extiende la mano base plana (5 dedos extendidos) para 'Caminar'"

    return True, None


def identify_walking_hands(
    hands_list: List[Any],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, int]:
    """
    Identifica asimétricamente cuál mano es la Mano Base y cuál la Mano Activa:
    - Mano Base: menor velocidad cinemática o más baja en Y (mayor coordenada Y).
    - Mano Activa: dedos apuntando hacia abajo (piernas caminantes) o más alta en Y.

    Returns:
        Tuple[base_hand_pts, active_hand_pts, base_idx, active_idx]
    """
    if not hands_list or len(hands_list) < 2:
        if hands_list and len(hands_list) == 1:
            pts = extract_landmarks_array(hands_list[0])
            is_act, _ = is_walking_active_hand(pts)
            if is_act:
                return None, pts, -1, 0
            return pts, None, 0, -1
        return None, None, -1, -1

    pts0 = extract_landmarks_array(hands_list[0])
    pts1 = extract_landmarks_array(hands_list[1])

    if pts0 is None or pts1 is None:
        return None, None, -1, -1

    # Criterio 1: Evaluación postural directa (dedos hacia abajo)
    is_act_0, _ = is_walking_active_hand(pts0)
    is_act_1, _ = is_walking_active_hand(pts1)

    if is_act_0 and not is_act_1:
        return pts1, pts0, 1, 0
    elif is_act_1 and not is_act_0:
        return pts0, pts1, 0, 1

    # Criterio 2: Altura vertical Y (la mano base es la más baja físicamente = mayor Y)
    y_mean_0 = (pts0[LM_WRIST, 1] + pts0[LM_MIDDLE_MCP, 1]) / 2.0
    y_mean_1 = (pts1[LM_WRIST, 1] + pts1[LM_MIDDLE_MCP, 1]) / 2.0

    if y_mean_0 > y_mean_1:
        return pts0, pts1, 0, 1
    else:
        return pts1, pts0, 1, 0


def is_helping_active_hand(pts: np.ndarray, gesture_name: str = "ayudar") -> Tuple[bool, Optional[str]]:
    """
    Valida la Mano Activa para los gestos 'ayudar' / 'ayudame':
    - Puño cerrado (índice, medio, anular y meñique flexionados hacia la palma).
    - Pulgar extendido verticalmente hacia arriba (Thumb Up / 'Bien').
    """
    if pts is None or len(pts) < 21:
        return False, "Mano activa no detectada"

    g_label = "Ayúdame" if "ayudame" in gesture_name.lower() else "Ayudar"

    st_idx, _ = classify_finger_state(pts, "index")
    st_mid, _ = classify_finger_state(pts, "middle")
    st_rng, _ = classify_finger_state(pts, "ring")
    st_pnk, _ = classify_finger_state(pts, "pinky")

    # Al menos 3 dedos flexionados en la palma para considerar puño cerrado
    curled_count = sum(1 for st in [st_idx, st_mid, st_rng, st_pnk] if st in ("CURLED", "HALF_CURLED"))
    if curled_count < 3:
        return False, f"Cierra el puño con los dedos flexionados para '{g_label}'"

    # Pulgar extendido verticalmente hacia arriba
    p_wrist = pts[LM_WRIST]
    p_thb_tip = pts[LM_THUMB_TIP]
    p_thb_ip = pts[LM_THUMB_IP]
    p_thb_mcp = pts[LM_THUMB_MCP]

    # En coordenadas de imagen, arriba significa menor coordenada Y
    is_upward = (p_thb_tip[1] < p_thb_ip[1]) and (p_thb_tip[1] < p_thb_mcp[1])
    is_thumb_above_wrist = (p_thb_tip[1] < p_wrist[1])

    # El pulgar debe ser vertical (delta Y significativo)
    dy = abs(p_thb_tip[1] - p_thb_mcp[1])
    dx = abs(p_thb_tip[0] - p_thb_mcp[0])
    is_vertical = dy > 0.45 * dx

    if not (is_upward and is_thumb_above_wrist and is_vertical):
        return False, f"Extiende el pulgar verticalmente hacia arriba ('Bien') para '{g_label}'"

    return True, None


def is_helping_base_hand(pts: np.ndarray, gesture_name: str = "ayudar") -> Tuple[bool, Optional[str]]:
    """
    Valida la Mano Base (Soporte) para los gestos 'ayudar' / 'ayudame':
    - Palma abierta horizontal mirando hacia arriba (5 dedos extendidos).
    """
    if pts is None or len(pts) < 21:
        return False, "Mano base no detectada"

    g_label = "Ayúdame" if "ayudame" in gesture_name.lower() else "Ayudar"

    st_idx, _ = classify_finger_state(pts, "index")
    st_mid, _ = classify_finger_state(pts, "middle")
    st_rng, _ = classify_finger_state(pts, "ring")
    st_pnk, _ = classify_finger_state(pts, "pinky")
    st_thb, _ = classify_finger_state(pts, "thumb")

    extended_count = sum(1 for st in [st_idx, st_mid, st_rng, st_pnk, st_thb] if st == "EXTENDED")
    if extended_count < 3 or (st_idx != "EXTENDED" and st_mid != "EXTENDED"):
        return False, f"Extiende la mano base plana (palma abierta) para '{g_label}'"

    # Verificar que las puntas no estén replegadas hacia abajo
    p_wrist = pts[LM_WRIST]
    p_mid_tip = pts[LM_MIDDLE_TIP]
    if p_mid_tip[1] > p_wrist[1] + 0.20:
        return False, f"Orienta la palma de la mano base hacia arriba para '{g_label}'"

    return True, None


def are_helping_hands_coupled(
    base_pts: np.ndarray,
    active_pts: np.ndarray,
    max_coupling_distance: float = 0.35,
    gesture_name: str = "ayudar",
) -> Tuple[bool, Optional[str]]:
    """
    Verifica que la Mano Activa esté apoyada sobre el plano palmar de la Mano Base.
    """
    g_label = "Ayúdame" if "ayudame" in gesture_name.lower() else "Ayudar"
    if base_pts is None or active_pts is None:
        return False, f"Se requieren ambas manos para '{g_label}'"

    # Distancia entre la palma/muñeca de la mano activa y la palma/muñeca de la base
    mcp_base = base_pts[LM_MIDDLE_MCP]
    mcp_act = active_pts[LM_MIDDLE_MCP]
    w_base = base_pts[LM_WRIST]
    w_act = active_pts[LM_WRIST]

    dist_palms = math.hypot(mcp_act[0] - mcp_base[0], mcp_act[1] - mcp_base[1])
    dist_wrists = math.hypot(w_act[0] - w_base[0], w_act[1] - w_base[1])
    min_dist = min(dist_palms, dist_wrists)

    if min_dist > max_coupling_distance:
        return False, "Apoya el puño de la mano activa sobre la palma de la mano base"

    # La mano activa debe estar situada físicamente sobre o a nivel de la mano base (Y_activa <= Y_base)
    if active_pts[LM_WRIST, 1] > base_pts[LM_WRIST, 1] + 0.14:
        return False, "Sitúa la mano activa apoyada sobre la mano base de soporte"

    return True, None


def identify_helping_hands(
    hands_list: List[Any],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, int]:
    """
    Identifica cuál es la Mano Base (soporte plano) y cuál la Mano Activa (puño con pulgar arriba).

    Returns:
        Tuple[base_hand_pts, active_hand_pts, base_idx, active_idx]
    """
    if not hands_list or len(hands_list) < 2:
        if hands_list and len(hands_list) == 1:
            pts = extract_landmarks_array(hands_list[0])
            is_act, _ = is_helping_active_hand(pts)
            if is_act:
                return None, pts, -1, 0
            return pts, None, 0, -1
        return None, None, -1, -1

    pts0 = extract_landmarks_array(hands_list[0])
    pts1 = extract_landmarks_array(hands_list[1])

    if pts0 is None or pts1 is None:
        return None, None, -1, -1

    # Criterio 1: Evaluación postural (pulgar arriba vs mano plana)
    is_act_0, _ = is_helping_active_hand(pts0)
    is_act_1, _ = is_helping_active_hand(pts1)

    if is_act_0 and not is_act_1:
        return pts1, pts0, 1, 0
    elif is_act_1 and not is_act_0:
        return pts0, pts1, 0, 1

    # Criterio 2: Altura vertical Y (la mano activa está apoyada SOBRE la base, por tanto menor Y en pantalla)
    y_mean_0 = (pts0[LM_WRIST, 1] + pts0[LM_MIDDLE_MCP, 1]) / 2.0
    y_mean_1 = (pts1[LM_WRIST, 1] + pts1[LM_MIDDLE_MCP, 1]) / 2.0

    if y_mean_0 > y_mean_1:
        # pts0 es más baja (mayor Y) -> base, pts1 es más alta (menor Y) -> activa
        return pts0, pts1, 0, 1
    else:
        return pts1, pts0, 1, 0
