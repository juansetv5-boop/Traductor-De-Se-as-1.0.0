"""
test_anthropometric_calibration.py — Validación de calibración antropométrica facial y
dimensionamiento dinámico en GhostTrainer.
"""

import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, ".")

from src.ui.ghost_trainer import GhostTrainer, check_palm_parallel_to_torso


class MockLandmark:
    def __init__(self, x: float, y: float, visibility: float = 0.9) -> None:
        self.x = x
        self.y = y
        self.visibility = visibility


class TestAnthropometricCalibration(unittest.TestCase):
    def setUp(self) -> None:
        self.trainer = GhostTrainer()

    def test_face_scale_from_ears(self) -> None:
        """Verifica el cálculo de FACE_SCALE basado en la distancia euclidiana entre orejas (LM 7 y 8)."""
        # Crear 33 landmarks mock
        lms = [MockLandmark(0.5, 0.5) for _ in range(33)]

        # Nariz (0)
        lms[0] = MockLandmark(0.50, 0.25)
        # Oreja Izquierda (7) y Derecha (8): distancia en X = 0.12 (pantalla 1920 -> 230 px)
        lms[7] = MockLandmark(0.44, 0.24)
        lms[8] = MockLandmark(0.56, 0.24)
        # Hombros (11 y 12)
        lms[11] = MockLandmark(0.40, 0.50)
        lms[12] = MockLandmark(0.60, 0.50)

        results = SimpleNamespace(
            pose_landmarks=SimpleNamespace(landmark=lms)
        )

        initial_scale = self.trainer.face_scale

        # Actualizar suficientes veces para que el filtro EMA y el clamp del 5% converjan
        for _ in range(500):
            self.trainer.update_anthropometric_calibration(results)

        expected_ear_dist = (0.56 - 0.44) * self.trainer.screen_w
        expected_measured = expected_ear_dist * 1.5
        self.assertAlmostEqual(self.trainer.face_scale, expected_measured, delta=5.0)

        # Validar dimensionamiento dinámico
        expected_r_visual = max(18, int(0.35 * self.trainer.face_scale))
        expected_r_hitbox = max(24, int(0.45 * self.trainer.face_scale))
        self.assertEqual(self.trainer.r_visual, expected_r_visual)
        self.assertEqual(self.trainer.r_hitbox, expected_r_hitbox)

    def test_fallback_when_face_is_lost(self) -> None:
        """Verifica que si la cara se pierde, se mantenga el último FACE_SCALE válido registrado."""
        valid_scale = 180.0
        self.trainer.last_valid_face_scale = valid_scale
        self.trainer.face_scale = valid_scale

        # Resultados vacíos (cara no detectada)
        empty_results = SimpleNamespace(pose_landmarks=None)
        self.trainer.update_anthropometric_calibration(empty_results)

        self.assertEqual(self.trainer.face_scale, valid_scale)

    def test_anatomical_positioning_hola_trajectory(self) -> None:
        """Verifica que HOLA tenga P0 en la sien y P1 en desplazamiento oblicuo con palma abierta."""
        self.trainer.current_gesture = "Hola"
        self.trainer._reposition_buttons_anatomically()

        self.assertEqual(len(self.trainer.buttons), 2)
        btn0 = self.trainer.buttons[0]
        btn1 = self.trainer.buttons[1]

        # P0: Nose.x + 1.2 * fs, Nose.y - 0.4 * fs
        nx, ny = self.trainer.smoothed_nose
        fs = self.trainer.face_scale

        expected_p0_x = int(nx + 1.20 * fs)
        expected_p0_y = int(ny - 0.40 * fs)
        self.assertAlmostEqual(btn0.px_x, expected_p0_x, delta=3)
        self.assertAlmostEqual(btn0.px_y, expected_p0_y, delta=3)

        # P1: Nose.x + 1.8 * fs, Nose.y - 0.5 * fs
        expected_p1_x = int(nx + 1.80 * fs)
        expected_p1_y = int(ny - 0.50 * fs)
        self.assertAlmostEqual(btn1.px_x, expected_p1_x, delta=3)
        self.assertAlmostEqual(btn1.px_y, expected_p1_y, delta=3)

        # Hitbox requiere palma abierta
        self.assertTrue(btn0.require_open_palm)
        self.assertTrue(btn1.require_open_palm)

    def test_anatomical_positioning_gracias_projection(self) -> None:
        """Verifica que GRACIAS inicie en mentón (Nose.y + 0.45 * fs) y proyecte a 45° con palma hacia arriba."""
        self.trainer.current_gesture = "Gracias"
        self.trainer._reposition_buttons_anatomically()

        self.assertEqual(len(self.trainer.buttons), 2)
        btn0 = self.trainer.buttons[0]
        btn1 = self.trainer.buttons[1]

        nx, ny = self.trainer.smoothed_nose
        fs = self.trainer.face_scale

        # P0: Nose.x, Nose.y + 0.45 * fs
        expected_p0_x = int(nx)
        expected_p0_y = int(ny + 0.45 * fs)
        self.assertAlmostEqual(btn0.px_x, expected_p0_x, delta=3)
        self.assertAlmostEqual(btn0.px_y, expected_p0_y, delta=3)

        # P1: Nose.x + 0.5 * fs, Nose.y + 1.5 * fs
        expected_p1_x = int(nx + 0.50 * fs)
        expected_p1_y = int(ny + 1.50 * fs)
        self.assertAlmostEqual(btn1.px_x, expected_p1_x, delta=3)
        self.assertAlmostEqual(btn1.px_y, expected_p1_y, delta=3)

        # P1 exige rotación de palma hacia arriba
        self.assertTrue(btn1.require_palm_up)

    def test_anatomical_positioning_buenos_dias_biphasic(self) -> None:
        """Verifica que BUENOS DIAS sea bifásico: FASE 1 mentón y FASE 2 bimanual simultánea en pecho."""
        self.trainer.current_gesture = "Buenos dias"
        self.trainer._reposition_buttons_anatomically()

        # Debe generar 6 botones (2 de FASE 1 + 4 de FASE 2)
        self.assertEqual(len(self.trainer.buttons), 6)

        btn_fase1_0 = self.trainer.buttons[0]
        btn_fase1_1 = self.trainer.buttons[1]
        self.assertFalse(btn_fase1_0.is_bimanual)
        self.assertFalse(btn_fase1_1.is_bimanual)

        # FASE 2: Pecho Izq (btn 2) y Pecho Der (btn 3)
        btn_pecho_izq = self.trainer.buttons[2]
        btn_pecho_der = self.trainer.buttons[3]
        self.assertTrue(btn_pecho_izq.is_bimanual)
        self.assertTrue(btn_pecho_der.is_bimanual)
        self.assertEqual(btn_pecho_izq.pair_id, 3)
        self.assertEqual(btn_pecho_der.pair_id, 2)
        self.assertEqual(btn_pecho_izq.side, "left")
        self.assertEqual(btn_pecho_der.side, "right")

        # FASE 2: Arcos Izq (btn 4) y Der (btn 5)
        btn_arco_izq = self.trainer.buttons[4]
        btn_arco_der = self.trainer.buttons[5]
        self.assertTrue(btn_arco_izq.is_bimanual)
        self.assertTrue(btn_arco_der.is_bimanual)
        self.assertTrue(btn_arco_izq.require_open_palm)
        self.assertTrue(btn_arco_der.require_open_palm)

    def test_por_favor_circular_orbit_6_waypoints(self) -> None:
        """Verifica que 'por_favor' genere 6 waypoints en órbita circular horaria con R = 0.6 * FACE_SCALE."""
        for g_name in ["Por favor", "por_favor"]:
            self.trainer.current_gesture = g_name
            self.trainer._reposition_buttons_anatomically()

            self.assertEqual(len(self.trainer.buttons), 6, f"Debe haber exactamente 6 botones para {g_name}")

            fs = self.trainer.face_scale
            expected_radius = 0.60 * fs
            self.assertAlmostEqual(self.trainer.por_favor_radius, expected_radius, delta=1e-3)

            centro_x, centro_y = self.trainer.por_favor_center

            # Verificar cada uno de los 6 waypoints en sentido horario iniciando arriba
            for i in range(6):
                angle = - (i * (2.0 * np.pi / 6.0)) + (np.pi / 2.0)
                expected_x = int(centro_x + expected_radius * np.cos(angle))
                expected_y = int(centro_y - expected_radius * np.sin(angle))

                btn = self.trainer.buttons[i]
                self.assertAlmostEqual(btn.px_x, expected_x, delta=2, msg=f"Waypoint {i} X difiere")
                self.assertAlmostEqual(btn.px_y, expected_y, delta=2, msg=f"Waypoint {i} Y difiere")

    def test_por_favor_biomechanical_palm_orientation(self) -> None:
        """Verifica la validación biomecánica de la normal de la palma paralela al torso."""
        # Mano plana frontal
        flat_lms = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(21)]
        flat_lms[0] = SimpleNamespace(x=0.5, y=0.5, z=0.0)      # Muñeca
        flat_lms[5] = SimpleNamespace(x=0.5, y=0.4, z=0.0)      # Base índice
        flat_lms[17] = SimpleNamespace(x=0.55, y=0.42, z=0.0)   # Base meñique
        hand_flat = SimpleNamespace(landmark=flat_lms)

        is_parallel, score = check_palm_parallel_to_torso(hand_flat)
        self.assertTrue(is_parallel)
        self.assertGreaterEqual(score, 0.80)

        # Mano de canto (karate-chop / perpendicular al pecho)
        edge_lms = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(21)]
        edge_lms[0] = SimpleNamespace(x=0.5, y=0.5, z=0.0)      # Muñeca
        edge_lms[5] = SimpleNamespace(x=0.5, y=0.4, z=0.0)      # Base índice
        edge_lms[17] = SimpleNamespace(x=0.5, y=0.4, z=0.10)    # Base meñique desplazada en Z
        hand_edge = SimpleNamespace(landmark=edge_lms)

        is_parallel_edge, score_edge = check_palm_parallel_to_torso(hand_edge)
        self.assertFalse(is_parallel_edge)
        self.assertLess(score_edge, 0.38)

    def test_por_favor_radial_deviation_abort(self) -> None:
        """Verifica que una desviación radial mayor a 1.5 * R aborte la toma con feedback."""
        from src.ui.ghost_trainer import FSMState
        self.trainer.current_gesture = "por_favor"
        self.trainer._load_gesture_buttons("por_favor")
        self.trainer.fsm_state = FSMState.IN_SEQUENCE

        centro_x, centro_y = self.trainer.por_favor_center
        r = self.trainer.por_favor_radius

        # Mover cursor a 1.8 * R del centro
        far_cursor = (int(centro_x + 1.8 * r), int(centro_y))
        self.trainer.update(far_cursor, 0.03)

        self.assertEqual(self.trainer.fsm_state, FSMState.TIMEOUT_RESET)
        self.assertGreater(self.trainer.capture_pipeline.red_feedback_alpha, 0.0)

    def test_por_favor_reverse_rotation_abort(self) -> None:
        """Verifica que una inversión de giro sostenida en sentido antihorario aborte la toma."""
        import math
        from src.ui.ghost_trainer import FSMState
        self.trainer.current_gesture = "por_favor"
        self.trainer._load_gesture_buttons("por_favor")
        self.trainer.fsm_state = FSMState.IN_SEQUENCE

        centro_x, centro_y = self.trainer.por_favor_center
        r = self.trainer.por_favor_radius

        # Simular giro antihorario (ángulo decreciente en coordenadas polares horarias)
        # Iniciar en ángulo 0 (derecha) y rotar hacia arriba en sentido antihorario
        for step in range(15):
            angle = - (step * 0.12)  # Moviéndose antihorario
            cx = int(centro_x + r * math.cos(angle))
            cy = int(centro_y + r * math.sin(angle))
            self.trainer.update((cx, cy), 0.03)
            if self.trainer.fsm_state == FSMState.TIMEOUT_RESET:
                break

        self.assertEqual(self.trainer.fsm_state, FSMState.TIMEOUT_RESET)
        self.assertGreater(self.trainer.capture_pipeline.red_feedback_alpha, 0.0)

    def test_por_favor_completion_commit_and_tts(self) -> None:
        """Verifica que completar los 6 waypoints de por_favor guarde con etiqueta 'por_favor' y audio exacto."""
        from unittest.mock import MagicMock
        from src.ui.ghost_trainer import FSMState

        self.trainer.current_gesture = "por_favor"
        self.trainer._load_gesture_buttons("por_favor")
        self.trainer.tts.speak = MagicMock()
        self.trainer.capture_pipeline.commit = MagicMock(return_value=True)

        # Simular recorrido secuencial de los 6 waypoints
        for i in range(6):
            btn = self.trainer.buttons[i]
            target_pos = (btn.px_x, btn.px_y)
            # Dwell time suficiente para activar
            for _ in range(25):
                self.trainer.update(target_pos, 0.03)

        self.assertEqual(self.trainer.fsm_state, FSMState.SUCCESS)
        self.trainer.capture_pipeline.commit.assert_called_once_with(
            gesture="por_favor",
            category="verbos",
        )
        self.trainer.tts.speak.assert_called_once_with("Excelente, por favor registrado")


if __name__ == "__main__":
    unittest.main()
