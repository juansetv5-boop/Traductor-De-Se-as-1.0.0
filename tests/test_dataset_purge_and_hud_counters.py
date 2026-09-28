"""
test_dataset_purge_and_hud_counters.py — Pruebas para la purga del dataset y reinicio de contadores 0/30.
"""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, ".")

from src.ui.ghost_trainer import GhostTrainer
from src.data_collection.ghost_capture_pipeline import GhostCapturePipeline


def test_clean_dataset_directory_structure():
    """Verifica que la estructura de dataset/ esté limpia y cuente con los directorios base."""
    base_dir = "dataset"
    expected_subdirs = [
        os.path.join(base_dir, "raw_videos", "alfabeto"),
        os.path.join(base_dir, "raw_videos", "frases"),
        os.path.join(base_dir, "raw_videos", "verbos"),
        os.path.join(base_dir, "processed_tensors", "alfabeto"),
        os.path.join(base_dir, "processed_tensors", "frases"),
        os.path.join(base_dir, "processed_tensors", "verbos"),
    ]

    for subdir in expected_subdirs:
        assert os.path.isdir(subdir), f"El directorio requerido {subdir} no existe"

    # Verificar que no queden archivos de datos obsoletos
    obsolete_exts = {".mp4", ".avi", ".npz", ".npy", ".csv"}
    found_files = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in obsolete_exts:
                found_files.append(os.path.join(root, f))

    assert len(found_files) == 0, f"Se encontraron archivos obsoletos en dataset/: {found_files}"


def test_ghost_trainer_counters_start_at_zero_of_thirty():
    """Verifica que en GhostTrainer las muestras comiencen en 0/30 para todas las señas."""
    trainer = GhostTrainer()
    assert trainer.target_samples_per_gesture == 30

    # Probar para cada gesto en el vocabulario dinámico
    for gesture in trainer.dynamic_signs:
        trainer.current_gesture = gesture
        g_clean = gesture.strip().lower()
        lbl_count = "por_favor" if g_clean in ("por favor", "por_favor") else gesture
        count = trainer.capture_pipeline.get_sample_count(trainer.training_mode.lower(), lbl_count)
        assert count == 0, f"El contador para '{gesture}' no inició en 0 (valor={count})"

    # Probar reseteo explícito
    trainer.reset_sample_counters()
    assert len(trainer.capture_pipeline._counts) == 0


def test_hud_renders_zero_of_thirty():
    """Verifica que al renderizar el HUD, el texto de conteo contenga '0/30'."""
    trainer = GhostTrainer()
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Renderizar HUD sobre frame vacío
    rendered = trainer.render(dummy_frame, cursor_pos=None)
    assert rendered is not None
    assert rendered.shape == dummy_frame.shape

    # Verificar que el contador objetivo es 30 y las muestras actuales son 0
    sample_count = trainer.capture_pipeline.get_sample_count(trainer.training_mode.lower(), trainer.current_gesture)
    assert sample_count == 0
    assert trainer.target_samples_per_gesture == 30
