"""
test_caminar_kinematics.py — Pruebas unitarias para la cinemática asimétrica del gesto 'caminar'.
"""
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock
import numpy as np
import pytest

sys.path.insert(0, ".")

from src.ui.ghost_trainer import GhostTrainer, FSMState
from src.core.hand_pose_validator import (
    is_walking_active_hand,
    is_walking_base_hand,
    identify_walking_hands,
)


class MockLandmark:
    def __init__(self, x, y, z=0.0, visibility=0.95):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


class MockHandLandmarks:
    def __init__(self, coords: np.ndarray):
        self.landmark = [MockLandmark(pt[0], pt[1], pt[2]) for pt in coords]


class MockResultsHands:
    def __init__(self, hands_list):
        self.multi_hand_landmarks = hands_list


def create_walking_base_hand(y_base: float = 0.70) -> np.ndarray:
    """Mano base plana horizontal como superficie / piso (y más baja en pantalla)."""
    hand = np.zeros((21, 3), dtype=np.float32)
    # Muñeca
    hand[0] = [0.30, y_base, 0.0]
    # MCPs extendidos hacia la derecha
    for i, mcp in enumerate([2, 5, 9, 13, 17]):
        hand[mcp] = [0.35 + i * 0.02, y_base - 0.02, 0.0]
    # PIPs
    for i, pip in enumerate([3, 6, 10, 14, 18]):
        hand[pip] = [0.45 + i * 0.02, y_base - 0.03, 0.0]
    # DIPs
    for i, dip in enumerate([3, 7, 11, 15, 19]):
        hand[dip] = [0.55 + i * 0.02, y_base - 0.03, 0.0]
    # TIPs extendidos horizontalmente
    for i, tip in enumerate([4, 8, 12, 16, 20]):
        hand[tip] = [0.65 + i * 0.02, y_base - 0.03, 0.0]
    return hand


def create_walking_active_hand(wrist_y: float = 0.45, tip_y: float = 0.65) -> np.ndarray:
    """
    Mano activa con piernas caminantes:
    - Muñeca arriba (wrist_y)
    - Índice y medio extendidos orientados HACIA ABAJO (tip_y > wrist_y)
    - Anular y meñique flexionados en la palma
    """
    hand = np.zeros((21, 3), dtype=np.float32)
    hand[0] = [0.40, wrist_y, 0.0]  # Muñeca arriba

    # MCPs
    hand[2] = [0.38, wrist_y + 0.05, 0.0]  # Pulgar MCP
    hand[5] = [0.39, wrist_y + 0.06, 0.0]  # Índice MCP
    hand[9] = [0.41, wrist_y + 0.06, 0.0]  # Medio MCP
    hand[13] = [0.43, wrist_y + 0.05, 0.0]  # Anular MCP
    hand[17] = [0.44, wrist_y + 0.04, 0.0]  # Meñique MCP

    # Pulgar flexionado hacia la palma
    hand[3] = [0.38, wrist_y + 0.08, 0.0]
    hand[4] = [0.39, wrist_y + 0.07, 0.0]

    # Índice extendido HACIA ABAJO (Y aumenta hacia abajo)
    hand[6] = [0.39, wrist_y + 0.11, 0.0]  # PIP
    hand[7] = [0.39, wrist_y + 0.15, 0.0]  # DIP
    hand[8] = [0.39, tip_y, 0.0]           # TIP

    # Medio extendido HACIA ABAJO
    hand[10] = [0.41, wrist_y + 0.11, 0.0]  # PIP
    hand[11] = [0.41, wrist_y + 0.15, 0.0]  # DIP
    hand[12] = [0.41, tip_y, 0.0]           # TIP

    # Anular flexionado en la palma
    hand[14] = [0.43, wrist_y + 0.08, 0.0]  # PIP
    hand[15] = [0.43, wrist_y + 0.06, 0.0]  # DIP
    hand[16] = [0.43, wrist_y + 0.05, 0.0]  # TIP replegado

    # Meñique flexionado en la palma
    hand[18] = [0.44, wrist_y + 0.07, 0.0]  # PIP
    hand[19] = [0.44, wrist_y + 0.05, 0.0]  # DIP
    hand[20] = [0.44, wrist_y + 0.04, 0.0]  # TIP replegado

    return hand


def test_walking_hands_postural_classification():
    base_hand = create_walking_base_hand()
    active_hand = create_walking_active_hand()

    # Validar Mano Activa
    ok_act, err_act = is_walking_active_hand(active_hand)
    assert ok_act is True
    assert err_act is None

    # Si los dedos índice y medio están extendidos pero apuntan hacia arriba en vez de hacia abajo
    up_hand = active_hand.copy()
    up_hand[6] = [0.39, 0.45, 0.0]
    up_hand[7] = [0.39, 0.38, 0.0]
    up_hand[8] = [0.39, 0.30, 0.0]
    up_hand[10] = [0.41, 0.45, 0.0]
    up_hand[11] = [0.41, 0.38, 0.0]
    up_hand[12] = [0.41, 0.30, 0.0]

    ok_up, err_up = is_walking_active_hand(up_hand)
    assert ok_up is False
    assert "abajo" in err_up.lower()

    # Validar Mano Base
    ok_base, err_base = is_walking_base_hand(base_hand)
    assert ok_base is True
    assert err_base is None


def test_identify_walking_hands_roles():
    base_coords = create_walking_base_hand(y_base=0.75)
    active_coords = create_walking_active_hand(wrist_y=0.40, tip_y=0.60)

    mock_base = MockHandLandmarks(base_coords)
    mock_active = MockHandLandmarks(active_coords)

    # Caso 1: Orden [Base, Activa]
    b_pts, a_pts, b_idx, a_idx = identify_walking_hands([mock_base, mock_active])
    assert b_idx == 0
    assert a_idx == 1
    assert b_pts is not None and a_pts is not None

    # Caso 2: Orden invertido [Activa, Base]
    b_pts_inv, a_pts_inv, b_idx_inv, a_idx_inv = identify_walking_hands([mock_active, mock_base])
    assert b_idx_inv == 1
    assert a_idx_inv == 0


def test_ghost_trainer_dynamic_waypoints_projection_on_base_hand():
    trainer = GhostTrainer()
    trainer.current_gesture = "caminar"
    trainer._load_gesture_buttons("caminar")

    assert len(trainer.buttons) == 3

    base_coords = create_walking_base_hand(y_base=0.68)
    active_coords = create_walking_active_hand(wrist_y=0.35, tip_y=0.55)

    mock_base = MockHandLandmarks(base_coords)
    mock_active = MockHandLandmarks(active_coords)
    res_hands = MockResultsHands([mock_base, mock_active])

    trainer.update(cursor_pos=None, delta_time=0.033, results_hands=res_hands)

    # WP_0 anclado en la muñeca / base de la palma (LM 0) de la mano base
    expected_wp0_x = int(base_coords[0, 0] * trainer.screen_w)
    expected_wp0_y = int(base_coords[0, 1] * trainer.screen_h)
    assert abs(trainer.buttons[0].px_x - expected_wp0_x) <= 3
    assert abs(trainer.buttons[0].px_y - expected_wp0_y) <= 3

    # WP_1 anclado en el centro / nudillos medios (LM 9) de la mano base
    expected_wp1_x = int(base_coords[9, 0] * trainer.screen_w)
    expected_wp1_y = int(base_coords[9, 1] * trainer.screen_h)
    assert abs(trainer.buttons[1].px_x - expected_wp1_x) <= 3
    assert abs(trainer.buttons[1].px_y - expected_wp1_y) <= 3

    # WP_2 anclado en la punta de los dedos (LM 12) de la mano base
    expected_wp2_x = int(base_coords[12, 0] * trainer.screen_w)
    expected_wp2_y = int(base_coords[12, 1] * trainer.screen_h)
    assert abs(trainer.buttons[2].px_x - expected_wp2_x) <= 3
    assert abs(trainer.buttons[2].px_y - expected_wp2_y) <= 3


def test_ghost_trainer_walking_sequence_completion_and_commit():
    trainer = GhostTrainer()
    trainer.current_gesture = "caminar"
    trainer._load_gesture_buttons("caminar")
    trainer.tts.speak = MagicMock()
    trainer.capture_pipeline.commit = MagicMock(return_value=True)

    # Mock de cuerpo completo con caderas visibles para satisfacer FULL_BODY_REQUIRED
    pose_lms = [MockLandmark(0.5, 0.5, 0.0, visibility=0.95) for _ in range(33)]
    mock_results = SimpleNamespace(
        pose_landmarks=SimpleNamespace(landmark=pose_lms)
    )

    base_coords = create_walking_base_hand(y_base=0.70)
    mock_base = MockHandLandmarks(base_coords)

    # Recorrer secuencialmente WP_0 -> WP_1 -> WP_2
    for step_idx in range(3):
        # Actualizar primero para proyectar waypoints sobre la mano base
        dummy_active = create_walking_active_hand(wrist_y=0.40, tip_y=0.55)
        res_hands = MockResultsHands([mock_base, MockHandLandmarks(dummy_active)])
        trainer.update(cursor_pos=None, delta_time=0.03, results=mock_results, results_hands=res_hands)

        target_btn = trainer.buttons[step_idx]
        target_norm_x = target_btn.px_x / trainer.screen_w
        target_norm_y = target_btn.px_y / trainer.screen_h

        # Posicionar las puntas caminantes (LM 8 y 12) sobre el waypoint objetivo
        # Desplazar la mano completa manteniendo la postura recta
        step_active_coords = create_walking_active_hand(wrist_y=target_norm_y - 0.20, tip_y=target_norm_y)
        dx = target_norm_x - 0.40  # centro de la mano activa en x
        step_active_coords[:, 0] += dx

        res_hands_step = MockResultsHands([mock_base, MockHandLandmarks(step_active_coords)])

        # Sostener hasta completar el botón
        for _ in range(35):
            trainer.update(cursor_pos=None, delta_time=0.033, results=mock_results, results_hands=res_hands_step)

    # Debe haberse completado el gesto y guardado con la categoría y seña correctas
    assert trainer.fsm_state == FSMState.SUCCESS
    trainer.capture_pipeline.commit.assert_called_once_with(
        gesture="caminar",
        category="verbos",
    )
    trainer.tts.speak.assert_called_once_with("Excelente, has completado caminar")
