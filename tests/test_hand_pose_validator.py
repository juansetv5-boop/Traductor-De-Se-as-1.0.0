"""
Test integral de validación postural (Queremas) en GhostTrainer.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import pytest

from src.ui.ghost_trainer import GhostTrainer, ReactiveGhostButton, GESTURE_REQUIREMENTS
from src.core.hand_pose_validator import (
    HandPoseValidator,
    classify_finger_state,
    compute_palm_normal,
    is_hand_fist,
)


class MockLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class MockHandLandmarks:
    def __init__(self, coords: np.ndarray):
        self.landmark = [MockLandmark(pt[0], pt[1], pt[2]) for pt in coords]


class MockResultsHands:
    def __init__(self, hands_list):
        self.multi_hand_landmarks = hands_list


def create_open_hand() -> np.ndarray:
    """Mano abierta con 5 dedos extendidos hacia arriba."""
    hand = np.zeros((21, 3), dtype=np.float32)
    hand[0] = [0.5, 0.8, 0.0]  # muñeca
    for i, mcp in enumerate([2, 5, 9, 13, 17]):
        hand[mcp] = [0.4 + i * 0.05, 0.6, 0.0]
    for i, pip in enumerate([3, 6, 10, 14, 18]):
        hand[pip] = [0.4 + i * 0.05, 0.45, 0.0]
    for i, dip in enumerate([3, 7, 11, 15, 19]):
        hand[dip] = [0.4 + i * 0.05, 0.35, 0.0]
    for i, tip in enumerate([4, 8, 12, 16, 20]):
        hand[tip] = [0.4 + i * 0.05, 0.25, 0.0]
    return hand


def create_fist_hand() -> np.ndarray:
    """Puño cerrado con los 4 dedos completamente flexionados hacia la palma."""
    hand = np.zeros((21, 3), dtype=np.float32)
    hand[0] = [0.5, 0.8, 0.0]  # muñeca
    # Pulgar (1,2,3,4) flexionado/descansando
    hand[1] = [0.45, 0.75, 0.0]
    hand[2] = [0.42, 0.70, 0.0]
    hand[3] = [0.40, 0.65, 0.0]
    hand[4] = [0.38, 0.60, 0.0]
    for i, mcp in enumerate([5, 9, 13, 17]):
        hand[mcp] = [0.4 + i * 0.05, 0.6, 0.0]
    for i, pip in enumerate([6, 10, 14, 18]):
        hand[pip] = [0.4 + i * 0.05, 0.52, 0.05]
    for i, dip in enumerate([7, 11, 15, 19]):
        hand[dip] = [0.4 + i * 0.05, 0.55, 0.03]
    for i, tip in enumerate([8, 12, 16, 20]):
        hand[tip] = [0.4 + i * 0.05, 0.58, 0.02]
    return hand


def create_hola_hand() -> np.ndarray:
    """Mano de 'hola': índice y medio extendidos, anular y meñique flexionados."""
    hand = create_fist_hand()
    # Extender índice (5, 6, 7, 8)
    hand[6] = [0.45, 0.45, 0.0]
    hand[7] = [0.45, 0.35, 0.0]
    hand[8] = [0.45, 0.25, 0.0]
    # Extender medio (9, 10, 11, 12)
    hand[10] = [0.50, 0.45, 0.0]
    hand[11] = [0.50, 0.35, 0.0]
    hand[12] = [0.50, 0.25, 0.0]
    return hand


def test_gesture_requirements_structure():
    assert "hola" in GESTURE_REQUIREMENTS
    assert "gracias" in GESTURE_REQUIREMENTS
    assert "por_favor" in GESTURE_REQUIREMENTS

    pf = GESTURE_REQUIREMENTS["por_favor"]
    assert pf["hand_shape"] == "FIST"
    assert pf["palm_orientation"] == "INWARD_CHEST"
    assert pf["fingers"]["index"] == "CURLED"
    assert pf["fingers"]["middle"] == "CURLED"
    assert pf["fingers"]["ring"] == "CURLED"
    assert pf["fingers"]["pinky"] == "CURLED"


def test_hand_pose_validator_por_favor():
    validator = HandPoseValidator()
    open_hand = create_open_hand()
    fist_hand = create_fist_hand()

    # Mano abierta en 'por_favor' DEBE fallar con advertencia
    ok, msg = validator.validate_pose("por_favor", open_hand)
    assert not ok
    assert msg == "Cierra el puño para 'Por favor'"

    # Puño cerrado en 'por_favor' DEBE ser válido
    ok, msg = validator.validate_pose("por_favor", fist_hand)
    assert ok
    assert msg is None


def test_ghost_trainer_por_favor_collision_feedback():
    trainer = GhostTrainer()
    trainer.current_gesture = "por favor"
    trainer._load_gesture_buttons("por favor")

    target_btn = trainer.buttons[0]
    target_pos = (target_btn.px_x, target_btn.px_y)

    # 1. Probar con mano abierta dentro del radio Hitbox
    open_lms = MockHandLandmarks(create_open_hand())
    res_hands = MockResultsHands([open_lms])

    trainer.update(
        cursor_pos=target_pos,
        delta_time=0.1,
        results_hands=res_hands,
    )

    # El botón NO debe activarse y debe marcar has_posture_error = True
    assert target_btn.hover_progress == 0.0
    assert target_btn.has_posture_error is True
    assert target_btn.posture_error_msg == "Cierra el puño para 'Por favor'"
    assert trainer.hand_posture_error is True
    assert trainer.hand_posture_error_message == "Cierra el puño para 'Por favor'"

    # 2. Probar con puño cerrado dentro del radio Hitbox
    fist_lms = MockHandLandmarks(create_fist_hand())
    res_hands_fist = MockResultsHands([fist_lms])

    trainer.update(
        cursor_pos=target_pos,
        delta_time=0.1,
        results_hands=res_hands_fist,
    )

    # Ahora cumple SIMULTÁNEAMENTE posición y postura: el botón debe acumular progreso
    assert target_btn.has_posture_error is False
    assert trainer.hand_posture_error is False
    assert target_btn.hover_progress > 0.0


def test_ghost_trainer_hola_feedback():
    trainer = GhostTrainer()
    trainer.current_gesture = "hola"
    trainer._load_gesture_buttons("hola")

    target_btn = trainer.buttons[0]
    target_pos = (target_btn.px_x, target_btn.px_y)

    # Probar con puño cerrado (inválido para 'hola')
    fist_lms = MockHandLandmarks(create_fist_hand())
    res_hands_fist = MockResultsHands([fist_lms])

    trainer.update(
        cursor_pos=target_pos,
        delta_time=0.1,
        results_hands=res_hands_fist,
    )

    assert target_btn.hover_progress == 0.0
    assert target_btn.has_posture_error is True
    assert trainer.hand_posture_error is True

    # Probar con postura canónica de 'hola'
    hola_lms = MockHandLandmarks(create_hola_hand())
    res_hands_hola = MockResultsHands([hola_lms])

    trainer.update(
        cursor_pos=target_pos,
        delta_time=0.1,
        results_hands=res_hands_hola,
    )

    assert target_btn.has_posture_error is False
    assert trainer.hand_posture_error is False
    assert target_btn.hover_progress > 0.0


def test_ghost_trainer_gracias_feedback():
    trainer = GhostTrainer()
    trainer.current_gesture = "gracias"
    trainer._load_gesture_buttons("gracias")

    target_btn = trainer.buttons[0]
    target_pos = (target_btn.px_x, target_btn.px_y)

    # 1. Puño cerrado en 'gracias' DEBE fallar
    fist_lms = MockHandLandmarks(create_fist_hand())
    res_hands_fist = MockResultsHands([fist_lms])

    trainer.update(
        cursor_pos=target_pos,
        delta_time=0.1,
        results_hands=res_hands_fist,
    )

    assert target_btn.hover_progress == 0.0
    assert target_btn.has_posture_error is True
    assert trainer.hand_posture_error is True

    # 2. Palma abierta en 'gracias' DEBE ser válida
    open_lms = MockHandLandmarks(create_open_hand())
    res_hands_open = MockResultsHands([open_lms])

    trainer.update(
        cursor_pos=target_pos,
        delta_time=0.1,
        results_hands=res_hands_open,
    )

    assert target_btn.has_posture_error is False
    assert trainer.hand_posture_error is False
    assert target_btn.hover_progress > 0.0


def test_resolve_active_hand_cursor_numpy_boolean_safe():
    """Verifica que resolve_active_hand_cursor soporte arrays numpy multidimensionales sin provocar ValueError."""
    from src.ui.ghost_trainer import CursorType
    trainer = GhostTrainer()

    # Arrays de coordenadas en numpy (1D o 2D)
    fallback_points = {
        "right_index": np.array([0.45, 0.55], dtype=np.float32),
        "left_index": np.array([0.25, 0.35], dtype=np.float32),
        "right_wrist": np.array([0.48, 0.70], dtype=np.float32),
        "left_wrist": np.array([0.22, 0.70], dtype=np.float32),
    }

    # Probar con CursorType.INDEX_TIP
    trainer.cursor_type = CursorType.INDEX_TIP
    pt, idx = trainer.resolve_active_hand_cursor(None, fallback_points)
    assert pt is not None
    assert idx == -1

    # Probar cuando right_index es None y fallback a left_index
    fallback_left_only = {
        "right_index": None,
        "left_index": np.array([0.25, 0.35], dtype=np.float32),
    }
    pt, idx = trainer.resolve_active_hand_cursor(None, fallback_left_only)
    assert pt is not None
    assert idx == -1

    # Probar con CursorType.WRIST
    trainer.cursor_type = CursorType.WRIST
    pt, idx = trainer.resolve_active_hand_cursor(None, fallback_points)
    assert pt is not None
    assert idx == -1

    # Probar con CursorType.AUTO
    trainer.cursor_type = CursorType.AUTO
    pt, idx = trainer.resolve_active_hand_cursor(None, fallback_points)
    assert pt is not None
    assert idx == -1




def test_alphabet_validation():
    from src.core.hand_pose_validator import validate_alphabet_letter
    
    # 1. Test geometric 'v' vs 'u'
    hand = create_fist_hand()
    # Extender índice y medio juntos ('u')
    hand[6:9] = [[0.45, 0.45, 0.0], [0.45, 0.35, 0.0], [0.45, 0.25, 0.0]]
    hand[10:13] = [[0.46, 0.45, 0.0], [0.46, 0.35, 0.0], [0.46, 0.25, 0.0]]
    
    mock_lms = MockHandLandmarks(hand)
    
    # U should pass
    ok, err_f, msg, _ = validate_alphabet_letter("u", mock_lms)
    assert ok is True
    
    # V should fail due to geometry (too close)
    ok, err_f, msg, _ = validate_alphabet_letter("v", mock_lms)
    assert ok is False
    assert err_f == "index"
    
    # Separar para 'v'
    hand[6:9] = [[0.40, 0.45, 0.0], [0.38, 0.35, 0.0], [0.35, 0.25, 0.0]]
    hand[10:13] = [[0.50, 0.45, 0.0], [0.52, 0.35, 0.0], [0.55, 0.25, 0.0]]
    mock_lms = MockHandLandmarks(hand)
    
    # V should pass now
    ok, err_f, msg, _ = validate_alphabet_letter("v", mock_lms)
    assert ok is True
    # U should fail
    ok, err_f, msg, _ = validate_alphabet_letter("u", mock_lms)
    assert ok is False


if __name__ == "__main__":
    test_gesture_requirements_structure()
    test_hand_pose_validator_por_favor()
    test_ghost_trainer_por_favor_collision_feedback()
    test_ghost_trainer_hola_feedback()
    test_ghost_trainer_gracias_feedback()
    test_resolve_active_hand_cursor_numpy_boolean_safe()
    test_alphabet_validation()
    print("TODAS LAS PRUEBAS INTEGRALES PASARON SATISFACTORIAMENTE!")
