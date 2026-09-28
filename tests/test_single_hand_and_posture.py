"""
test_single_hand_and_posture.py — Pruebas para selección unimanual y alertas de cuerpo completo.
"""

import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, ".")

from src.ui.ghost_trainer import FULL_BODY_REQUIRED, CursorType, GhostTrainer


class MockLandmark:
    def __init__(self, x: float, y: float, visibility: float = 0.9) -> None:
        self.x = x
        self.y = y
        self.visibility = visibility


class TestSingleHandAndPosture(unittest.TestCase):
    def setUp(self) -> None:
        self.trainer = GhostTrainer(cursor_type=CursorType.INDEX_TIP)

    def test_single_hand_targeting_closer_hand(self) -> None:
        """Verifica que si hay 2 manos, se seleccione como activa la más cercana al Ghost Button."""
        # Colocar Ghost Button activo en el centro
        btn = self.trainer.buttons[0]
        btn_x = btn.px_x if btn.px_x > 0 else int(btn.x_norm * self.trainer.screen_w)
        btn_y = btn.px_y if btn.px_y > 0 else int(btn.y_norm * self.trainer.screen_h)

        norm_btn_x = btn_x / self.trainer.screen_w
        norm_btn_y = btn_y / self.trainer.screen_h

        # Mano 0: Cerca del botón (delta 0.02)
        hand0 = SimpleNamespace(
            landmark=[MockLandmark(norm_btn_x + 0.02, norm_btn_y + 0.02) for _ in range(21)]
        )
        # Mano 1: Lejos del botón (delta 0.35)
        hand1 = SimpleNamespace(
            landmark=[MockLandmark(norm_btn_x + 0.35, norm_btn_y + 0.35) for _ in range(21)]
        )

        results_hands = SimpleNamespace(multi_hand_landmarks=[hand0, hand1])

        cursor_px, active_idx = self.trainer.resolve_active_hand_cursor(results_hands)
        self.assertEqual(active_idx, 0)

        # Invertir orden: Mano 0 lejos, Mano 1 cerca
        results_hands_inv = SimpleNamespace(multi_hand_landmarks=[hand1, hand0])
        cursor_px_inv, active_idx_inv = self.trainer.resolve_active_hand_cursor(results_hands_inv)
        self.assertEqual(active_idx_inv, 1)

    def test_full_body_posture_blocking_and_warning(self) -> None:
        """Verifica que las señas en FULL_BODY_REQUIRED activen advertencia y bloqueen la captura si las caderas no son visibles."""
        self.trainer.current_gesture = "caminar"
        self.trainer._load_gesture_buttons("caminar")

        # Pose sin caderas visibles (visibilidad baja < 0.65)
        pose_lms = [MockLandmark(0.5, 0.5, visibility=0.9) for _ in range(33)]
        pose_lms[23] = MockLandmark(0.45, 0.60, visibility=0.30)  # Cadera izq no visible
        pose_lms[24] = MockLandmark(0.55, 0.60, visibility=0.40)  # Cadera der no visible

        results = SimpleNamespace(
            pose_landmarks=SimpleNamespace(landmark=pose_lms)
        )

        # Ubicar cursor exactamente sobre el botón inicial para probar el bloqueo
        target_btn = self.trainer.buttons[0]
        cursor_pos = (target_btn.px_x, target_btn.px_y)

        # Actualizar FSM
        self.trainer.update(cursor_pos, delta_time=0.1, results=results)

        # Debe activarse la advertencia y el progreso debe permanecer bloqueado en 0.0
        self.assertTrue(self.trainer.posture_warning)
        self.assertEqual(target_btn.hover_progress, 0.0)
        self.assertFalse(self.trainer.capture_pipeline.is_recording)

        # Ahora proveer postura válida de cuerpo completo (visibilidad >= 0.65)
        pose_lms[23] = MockLandmark(0.45, 0.60, visibility=0.95)
        pose_lms[24] = MockLandmark(0.55, 0.60, visibility=0.95)

        self.trainer.update(cursor_pos, delta_time=0.1, results=results)

        # La advertencia debe apagarse y el botón debe empezar a llenarse
        self.assertFalse(self.trainer.posture_warning)
        self.assertGreater(target_btn.hover_progress, 0.0)


if __name__ == "__main__":
    unittest.main()
