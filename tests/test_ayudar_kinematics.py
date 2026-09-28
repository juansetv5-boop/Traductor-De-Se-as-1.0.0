"""
test_ayudar_kinematics.py — Pruebas unitarias para la cinemática acoplada y postural del gesto 'ayudar'.
"""
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock
import numpy as np
import pytest

sys.path.insert(0, ".")

from src.ui.ghost_trainer import GhostTrainer, FSMState
from src.core.hand_pose_validator import (
    is_helping_active_hand,
    is_helping_base_hand,
    are_helping_hands_coupled,
    identify_helping_hands,
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


def create_helping_base_hand(wrist_x: float = 0.50, wrist_y: float = 0.65) -> np.ndarray:
    """
    Mano base plana horizontal mirando hacia arriba (5 dedos extendidos).
    En coords de imagen: Y = wrist_y, puntas extendidas horizontal/ligeramente arriba.
    """
    hand = np.zeros((21, 3), dtype=np.float32)
    # Muñeca
    hand[0] = [wrist_x - 0.10, wrist_y, 0.0]

    # MCPs extendidos hacia la derecha
    for i, mcp in enumerate([2, 5, 9, 13, 17]):
        hand[mcp] = [wrist_x - 0.05 + i * 0.02, wrist_y - 0.02, 0.0]

    # PIPs
    for i, pip in enumerate([3, 6, 10, 14, 18]):
        hand[pip] = [wrist_x + 0.02 + i * 0.02, wrist_y - 0.02, 0.0]

    # DIPs
    for i, dip in enumerate([3, 7, 11, 15, 19]):
        hand[dip] = [wrist_x + 0.08 + i * 0.02, wrist_y - 0.02, 0.0]

    # TIPs extendidos horizontalmente
    for i, tip in enumerate([4, 8, 12, 16, 20]):
        hand[tip] = [wrist_x + 0.14 + i * 0.02, wrist_y - 0.02, 0.0]

    return hand


def create_helping_active_hand(wrist_x: float = 0.50, wrist_y: float = 0.58) -> np.ndarray:
    """
    Mano activa (puño con pulgar vertical hacia arriba / 'Bien'):
    - Muñeca situada en (wrist_x, wrist_y)
    - Dedos índice, medio, anular y meñique flexionados hacia la palma
    - Pulgar extendido verticalmente hacia arriba (Y menor que muñeca y MCPs)
    """
    hand = np.zeros((21, 3), dtype=np.float32)
    hand[0] = [wrist_x, wrist_y, 0.0]  # Muñeca

    # MCPs
    hand[2] = [wrist_x - 0.02, wrist_y - 0.03, 0.0]  # Pulgar MCP
    hand[5] = [wrist_x, wrist_y - 0.05, 0.0]         # Índice MCP
    hand[9] = [wrist_x + 0.02, wrist_y - 0.05, 0.0]  # Medio MCP
    hand[13] = [wrist_x + 0.04, wrist_y - 0.04, 0.0] # Anular MCP
    hand[17] = [wrist_x + 0.06, wrist_y - 0.03, 0.0] # Meñique MCP

    # Pulgar extendido verticalmente hacia arriba (menor Y)
    hand[1] = [wrist_x - 0.02, wrist_y - 0.02, 0.0]  # CMC
    hand[3] = [wrist_x - 0.02, wrist_y - 0.07, 0.0]  # IP
    hand[4] = [wrist_x - 0.02, wrist_y - 0.12, 0.0]  # TIP (arriba)

    # Índice flexionado (curled)
    hand[6] = [wrist_x, wrist_y - 0.08, 0.0]         # PIP
    hand[7] = [wrist_x + 0.01, wrist_y - 0.04, 0.0]  # DIP
    hand[8] = [wrist_x + 0.01, wrist_y - 0.03, 0.0]  # TIP flexionado

    # Medio flexionado (curled)
    hand[10] = [wrist_x + 0.02, wrist_y - 0.08, 0.0] # PIP
    hand[11] = [wrist_x + 0.02, wrist_y - 0.04, 0.0] # DIP
    hand[12] = [wrist_x + 0.02, wrist_y - 0.03, 0.0] # TIP flexionado

    # Anular flexionado (curled)
    hand[14] = [wrist_x + 0.04, wrist_y - 0.07, 0.0] # PIP
    hand[15] = [wrist_x + 0.04, wrist_y - 0.04, 0.0] # DIP
    hand[16] = [wrist_x + 0.04, wrist_y - 0.03, 0.0] # TIP flexionado

    # Meñique flexionado (curled)
    hand[18] = [wrist_x + 0.06, wrist_y - 0.06, 0.0] # PIP
    hand[19] = [wrist_x + 0.06, wrist_y - 0.04, 0.0] # DIP
    hand[20] = [wrist_x + 0.06, wrist_y - 0.03, 0.0] # TIP flexionado

    return hand


def test_helping_active_hand_postural_classification():
    """Valida la clasificación postural de la mano activa ('Bien')."""
    active_hand = create_helping_active_hand()
    ok, err = is_helping_active_hand(active_hand)
    assert ok is True
    assert err is None

    # Error si los dedos están abiertos
    open_hand = create_helping_base_hand()
    ok_open, err_open = is_helping_active_hand(open_hand)
    assert ok_open is False
    assert "puño" in err_open.lower() or "flexionados" in err_open.lower()

    # Error si el pulgar apunta hacia abajo
    down_thumb = active_hand.copy()
    down_thumb[4] = [down_thumb[0, 0] - 0.02, down_thumb[0, 1] + 0.08, 0.0]  # TIP abajo de muñeca
    ok_down, err_down = is_helping_active_hand(down_thumb)
    assert ok_down is False
    assert "pulgar" in err_down.lower()


def test_helping_base_hand_postural_classification():
    """Valida la clasificación postural de la mano base (palma plana horizontal)."""
    base_hand = create_helping_base_hand()
    ok, err = is_helping_base_hand(base_hand)
    assert ok is True
    assert err is None

    # Error si está en puño
    fist_hand = create_helping_active_hand()
    ok_fist, err_fist = is_helping_base_hand(fist_hand)
    assert ok_fist is False
    assert "base" in err_fist.lower() or "extiende" in err_fist.lower()


def test_helping_hands_coupling():
    """Valida el acoplamiento físico de la mano activa sobre la base."""
    base_hand = create_helping_base_hand(wrist_x=0.50, wrist_y=0.65)
    active_hand = create_helping_active_hand(wrist_x=0.50, wrist_y=0.58)

    # Acopladas correctamente
    ok, err = are_helping_hands_coupled(base_hand, active_hand)
    assert ok is True
    assert err is None

    # Separadas lateralmente
    far_active = create_helping_active_hand(wrist_x=0.90, wrist_y=0.58)
    ok_far, err_far = are_helping_hands_coupled(base_hand, far_active)
    assert ok_far is False
    assert "apoya" in err_far.lower()

    # Mano activa por debajo de la base
    below_active = create_helping_active_hand(wrist_x=0.50, wrist_y=0.85)
    ok_below, err_below = are_helping_hands_coupled(base_hand, below_active)
    assert ok_below is False
    assert "sobre" in err_below.lower()


def test_identify_helping_hands_roles():
    """Valida la identificación de roles de mano base vs mano activa."""
    base_hand = create_helping_base_hand(wrist_x=0.50, wrist_y=0.65)
    active_hand = create_helping_active_hand(wrist_x=0.50, wrist_y=0.58)

    mock_base = MockHandLandmarks(base_hand)
    mock_active = MockHandLandmarks(active_hand)

    # Orden [Base, Activa]
    b_pts, a_pts, b_idx, a_idx = identify_helping_hands([mock_base, mock_active])
    assert b_idx == 0
    assert a_idx == 1
    assert b_pts is not None and a_pts is not None

    # Orden [Activa, Base]
    b_pts2, a_pts2, b_idx2, a_idx2 = identify_helping_hands([mock_active, mock_base])
    assert b_idx2 == 1
    assert a_idx2 == 0


def test_ghost_trainer_ayudar_waypoints_and_completion():
    """Valida la trayectoria de 2 puntos de 'ayudar' y el guardado con commit en verbos/ayudar."""
    trainer = GhostTrainer()
    trainer.current_gesture = "ayudar"
    trainer._load_gesture_buttons("ayudar")

    assert trainer.current_gesture == "ayudar"
    assert len(trainer.buttons) == 2

    # Coordenadas esperadas: P0 exterior frontal, P1 esternón
    btn0 = trainer.buttons[0]
    btn1 = trainer.buttons[1]
    assert "1. Frontal" in btn0.label
    assert "2. Esternón" in btn1.label
    # En coordenadas de imagen, P1 (esternón, y + 1.2*fs) está más arriba que P0 (frontal, y + 1.8*fs)
    # Por lo tanto btn1.px_y < btn0.px_y (menor Y = más arriba)
    assert btn1.px_y < btn0.px_y

    # Simulación de paso de P0 a P1 con manos acopladas
    trainer.capture_pipeline.commit = MagicMock()
    trainer.capture_pipeline.start_recording = MagicMock()
    trainer.tts.speak = MagicMock()

    # Generar manos centradas en P0
    p0_norm_x = btn0.px_x / trainer.screen_w
    p0_norm_y = btn0.px_y / trainer.screen_h

    base_p0 = create_helping_base_hand(wrist_x=p0_norm_x, wrist_y=p0_norm_y + 0.05)
    active_p0 = create_helping_active_hand(wrist_x=p0_norm_x, wrist_y=p0_norm_y)

    mock_hands_p0 = MockResultsHands([
        MockHandLandmarks(base_p0),
        MockHandLandmarks(active_p0),
    ])

    dummy_frame = np.zeros((trainer.screen_h, trainer.screen_w, 3), dtype=np.uint8)

    # Hover en P0 hasta completar
    for _ in range(35):
        trainer.update(
            cursor_pos=(btn0.px_x, btn0.px_y),
            results_hands=mock_hands_p0,
            current_frame=dummy_frame,
            current_spatial=np.zeros(147, dtype=np.float32),
            delta_time=0.033,
        )

    assert btn0.is_completed is True
    assert trainer.fsm_state == FSMState.IN_SEQUENCE
    assert trainer.capture_pipeline.start_recording.called
    assert trainer.capture_pipeline.start_recording.call_args[1]["gesture"] == "ayudar"

    # Ahora mover manos a P1
    p1_norm_x = btn1.px_x / trainer.screen_w
    p1_norm_y = btn1.px_y / trainer.screen_h

    base_p1 = create_helping_base_hand(wrist_x=p1_norm_x, wrist_y=p1_norm_y + 0.05)
    active_p1 = create_helping_active_hand(wrist_x=p1_norm_x, wrist_y=p1_norm_y)

    mock_hands_p1 = MockResultsHands([
        MockHandLandmarks(base_p1),
        MockHandLandmarks(active_p1),
    ])

    for _ in range(25):
        trainer.update(
            cursor_pos=(btn1.px_x, btn1.px_y),
            results_hands=mock_hands_p1,
            current_frame=dummy_frame,
            current_spatial=np.zeros(147, dtype=np.float32),
            delta_time=0.033,
        )

    assert btn1.is_completed is True
    assert trainer.fsm_state == FSMState.SUCCESS
    assert trainer.capture_pipeline.commit.called
    commit_kwargs = trainer.capture_pipeline.commit.call_args[1]
    assert commit_kwargs["gesture"] == "ayudar"
    assert commit_kwargs["category"] == "verbos"
    assert trainer.tts.speak.called


def test_ghost_trainer_ayudar_thumb_drop_during_sequence_alerts():
    """Valida que si durante la traslación hacia el pecho se desfigura el pulgar o se desacoplan, se alerta."""
    trainer = GhostTrainer()
    trainer.current_gesture = "ayudar"
    trainer._load_gesture_buttons("ayudar")

    btn0 = trainer.buttons[0]
    btn1 = trainer.buttons[1]

    p0_norm_x = btn0.px_x / trainer.screen_w
    p0_norm_y = btn0.px_y / trainer.screen_h

    base_p0 = create_helping_base_hand(wrist_x=p0_norm_x, wrist_y=p0_norm_y + 0.05)
    active_p0 = create_helping_active_hand(wrist_x=p0_norm_x, wrist_y=p0_norm_y)

    mock_hands_p0 = MockResultsHands([
        MockHandLandmarks(base_p0),
        MockHandLandmarks(active_p0),
    ])

    dummy_frame = np.zeros((trainer.screen_h, trainer.screen_w, 3), dtype=np.uint8)

    # Completar P0 para entrar en FSMState.IN_SEQUENCE
    for _ in range(35):
        trainer.update(
            cursor_pos=(btn0.px_x, btn0.px_y),
            results_hands=mock_hands_p0,
            current_frame=dummy_frame,
            current_spatial=np.zeros(147, dtype=np.float32),
            delta_time=0.033,
        )

    assert trainer.fsm_state == FSMState.IN_SEQUENCE

    # En secuencia: la mano activa pierde el pulgar extendido (desfigurado)
    broken_active = active_p0.copy()
    broken_active[4] = [broken_active[0, 0] - 0.02, broken_active[0, 1] + 0.08, 0.0]  # pulgar abajo

    mock_hands_broken = MockResultsHands([
        MockHandLandmarks(base_p0),
        MockHandLandmarks(broken_active),
    ])

    trainer.update(
        cursor_pos=(btn1.px_x, btn1.px_y),
        results_hands=mock_hands_broken,
        current_frame=dummy_frame,
        current_spatial=np.zeros(147, dtype=np.float32),
        delta_time=0.033,
    )

    assert trainer.hand_posture_error is True
    assert "pulgar" in trainer.hand_posture_error_message.lower()

