"""
test_ghost_pipeline.py — Pruebas unitarias y de integración para GhostCapturePipeline.
Valida:
    1. Llenado y delimitación estricta de buffer fijo deque(maxlen=60).
    2. Comportamiento del trigger de inicio con Hold.
    3. Rutina de descarte y feedback visual.
    4. Rutina de guardado asíncrono (MP4 a 30 FPS + tensor [60, features] en NPZ para PyTorch).
    5. Contador persistente en disco.
"""

import os
import shutil
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, ".")

from src.data_collection.ghost_capture_pipeline import GhostCapturePipeline


class TestGhostCapturePipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = "dataset_test"
        self.pipeline = GhostCapturePipeline(base_dir=self.test_dir, target_frames=60, target_fps=30)

    def tearDown(self) -> None:
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_buffer_fixed_size_deque(self) -> None:
        """Verifica que el buffer temporal mantenga tamaño máximo fijo de 60 frames."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_spatial = np.zeros(147, dtype=np.float32)

        self.pipeline.start_recording("Hola")
        self.assertTrue(self.pipeline.is_recording)

        # Insertar 80 frames
        for _ in range(80):
            self.pipeline.record_frame(dummy_frame, dummy_spatial)

        # Deque debe estar topado en 60
        self.assertEqual(len(self.pipeline.frame_buffer), 60)
        self.assertEqual(len(self.pipeline.spatial_buffer), 60)

    def test_discard_clears_buffer_and_triggers_feedback(self) -> None:
        """Verifica que discard limpie el buffer y active el feedback visual rojo."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_spatial = np.zeros(147, dtype=np.float32)

        self.pipeline.start_recording("Gracias")
        for _ in range(15):
            self.pipeline.record_frame(dummy_frame, dummy_spatial)

        self.assertEqual(len(self.pipeline.frame_buffer), 15)
        self.pipeline.discard("Timeout de prueba")

        self.assertFalse(self.pipeline.is_recording)
        self.assertEqual(len(self.pipeline.frame_buffer), 0)
        self.assertGreater(self.pipeline.red_feedback_alpha, 0.0)

    def test_commit_creates_mp4_and_npz_tensor(self) -> None:
        """Verifica que commit guarde tanto el video MP4 como el archivo NPZ con tensor [60, 441]."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_spatial = np.random.randn(147).astype(np.float32)

        self.pipeline.start_recording("Por favor")
        for i in range(45):  # Menos de 60 para validar el padding a 60
            self.pipeline.record_frame(dummy_frame, dummy_spatial + i * 0.01)

        gesture = "Por favor"
        committed = self.pipeline.commit(gesture)
        self.assertTrue(committed)
        self.assertFalse(self.pipeline.is_recording)

        # Esperar a que el hilo secundario complete la persistencia
        time.sleep(1.0)

        # Validar directorio de video
        video_dir = os.path.join(self.test_dir, "raw_videos", "verbos", gesture)
        self.assertTrue(os.path.exists(video_dir))
        mp4_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
        self.assertEqual(len(mp4_files), 1)

        # Validar directorio de tensor
        tensor_dir = os.path.join(self.test_dir, "processed_tensors", "verbos", gesture)
        self.assertTrue(os.path.exists(tensor_dir))
        npz_files = [f for f in os.listdir(tensor_dir) if f.endswith(".npz")]
        self.assertEqual(len(npz_files), 1)

        # Cargar tensor .npz y validar forma de [60, features]
        npz_path = os.path.join(tensor_dir, npz_files[0])
        data = np.load(npz_path)
        self.assertIn("tensor", data)
        tensor = data["tensor"]
        self.assertEqual(tensor.shape, (60, 441))  # 60 frames x 441 features (pos + vel + acc)
        self.assertIn("spatial", data)
        self.assertEqual(data["spatial"].shape, (60, 147))  # 60 frames x 147 features espaciales

        # Validar contador de muestras
        count = self.pipeline.get_sample_count("verbos", gesture)
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
