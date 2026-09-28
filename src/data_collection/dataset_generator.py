"""
dataset_generator.py — Procesamiento de secuencias de video a tensores PyTorch.
Extrae ventanas temporales deslizantes (T=40), calcula diferenciales cinemáticos (441 features)
y realiza partición estricta por sujetos (Subject-wise Split) para evitar Data Leakage.
"""

from __future__ import annotations

import argparse
import os
import sys

# Permitir ejecucion directa como script agregando la raiz al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from src.config_dynamic import (
    DATA_DIR,
    DATASET_FILE,
    INPUT_FEATURES_PER_FRAME,
    PILOT_VOCABULARY,
    RAW_SPATIAL_FEATURES,
    STRIDE_STEP,
    TEMPORAL_WINDOW_SIZE,
    VIDEOS_DIR,
)
from src.data_collection.feature_extractor import HolisticFeatureExtractor


class DatasetGenerator:
    """
    Convierte videos crudos en tensores empaquetados para entrenamiento de TCN.
    """

    def __init__(
        self,
        videos_dir: str = VIDEOS_DIR,
        output_file: str = DATASET_FILE,
        window_size: int = TEMPORAL_WINDOW_SIZE,
        stride: int = STRIDE_STEP,
    ) -> None:
        self.videos_dir = videos_dir
        self.output_file = output_file
        self.window_size = window_size
        self.stride = stride
        self.extractor = HolisticFeatureExtractor()
        self.label_map = {name: idx for idx, name in enumerate(PILOT_VOCABULARY)}

    def process_video_to_spatial_sequence(self, video_path: str) -> np.ndarray:
        """Lee un archivo de video y extrae la matriz de posiciones espaciales (N_frames, 147)."""
        cap = cv2.VideoCapture(video_path)
        spatial_frames: List[np.ndarray] = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = self.extractor.process_frame(frame)
            spatial_vec, _ = self.extractor.extract_spatial_vector(results)
            spatial_frames.append(spatial_vec)

        cap.release()
        if not spatial_frames:
            return np.empty((0, RAW_SPATIAL_FEATURES), dtype=np.float32)

        return np.vstack(spatial_frames).astype(np.float32)

    def extract_windows_from_sequence(
        self, spatial_seq: np.ndarray, label_idx: int
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Aplica ventana deslizante de tamaño T=40 y calcula cinemática completa."""
        windows: List[np.ndarray] = []
        labels: List[int] = []

        num_frames = len(spatial_seq)
        if num_frames < self.window_size:
            # Si el video es ligeramente más corto, hacer padding replicando el último frame
            pad_len = self.window_size - num_frames
            padding = np.repeat(spatial_seq[-1:], pad_len, axis=0)
            spatial_seq = np.vstack([spatial_seq, padding])
            num_frames = len(spatial_seq)

        for start_idx in range(0, num_frames - self.window_size + 1, self.stride):
            sub_seq = spatial_seq[start_idx : start_idx + self.window_size]  # (40, 147)
            kinematics = self.extractor.compute_temporal_kinematics(sub_seq)  # (40, 441)
            # Reorganizar a formato Conv1D: (Channels, Time) = (441, 40)
            conv1d_input = np.transpose(kinematics, (1, 0))
            windows.append(conv1d_input)
            labels.append(label_idx)

        return windows, labels

    def build_dataset(self, val_ratio: float = 0.2) -> Dict[str, torch.Tensor]:
        """
        Escanea el directorio de videos, agrupa por sujetos y divide en Train / Validation.
        """
        if not os.path.exists(self.videos_dir):
            raise FileNotFoundError(f"Directorio de videos no encontrado: {self.videos_dir}")

        subjects = sorted([s for s in os.listdir(self.videos_dir) if os.path.isdir(os.path.join(self.videos_dir, s))])
        print(f"[INFO] Sujetos detectados: {subjects}")

        if not subjects:
            raise ValueError(f"No se encontraron carpetas de sujetos en {self.videos_dir}")

        # Partición Subject-wise: Si hay >= 2 sujetos, asignamos al menos 1 sujeto a validación
        if len(subjects) >= 2:
            num_val_subjects = max(1, int(len(subjects) * val_ratio))
            train_subjects = subjects[:-num_val_subjects]
            val_subjects = subjects[-num_val_subjects:]
        else:
            # Fallback si solo hay 1 sujeto grabado por el momento
            train_subjects = subjects
            val_subjects = subjects
            print("[AVISO] Solo 1 sujeto detectado; el conjunto de validación usará muestras del mismo sujeto.")

        print(f"[INFO] Sujetos de Train: {train_subjects}")
        print(f"[INFO] Sujetos de Val: {val_subjects}")

        def collect_from_subjects(subject_list: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
            all_x: List[np.ndarray] = []
            all_y: List[int] = []

            for subj in subject_list:
                subj_path = os.path.join(self.videos_dir, subj)
                for gesture_name in os.listdir(subj_path):
                    if gesture_name not in self.label_map:
                        continue
                    label_idx = self.label_map[gesture_name]
                    gesture_dir = os.path.join(subj_path, gesture_name)
                    video_files = [f for f in os.listdir(gesture_dir) if f.endswith(".mp4")]

                    for vf in video_files:
                        vpath = os.path.join(gesture_dir, vf)
                        seq = self.process_video_to_spatial_sequence(vpath)
                        if len(seq) == 0:
                            continue
                        wins, lbls = self.extract_windows_from_sequence(seq, label_idx)
                        all_x.extend(wins)
                        all_y.extend(lbls)

            if not all_x:
                return torch.empty((0, INPUT_FEATURES_PER_FRAME, self.window_size)), torch.empty((0,), dtype=torch.long)

            tensor_x = torch.tensor(np.array(all_x), dtype=torch.float32)
            tensor_y = torch.tensor(np.array(all_y), dtype=torch.long)
            return tensor_x, tensor_y

        print("[PROCESANDO] Extrayendo keypoints de videos de Train...")
        train_x, train_y = collect_from_subjects(train_subjects)
        print(f"[PROCESANDO] Extrayendo keypoints de videos de Val...")
        val_x, val_y = collect_from_subjects(val_subjects)

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        dataset_payload = {
            "train_x": train_x,
            "train_y": train_y,
            "val_x": val_x,
            "val_y": val_y,
            "classes": PILOT_VOCABULARY,
            "window_size": self.window_size,
            "input_features": INPUT_FEATURES_PER_FRAME,
        }

        torch.save(dataset_payload, self.output_file)
        print(f"[EXITO] Dataset guardado en: {self.output_file}")
        print(f"  - Train shape: {train_x.shape}, Labels: {train_y.shape}")
        print(f"  - Val shape:   {val_x.shape}, Labels: {val_y.shape}")
        return dataset_payload

    @staticmethod
    def generate_synthetic_benchmark_dataset(
        output_file: str = DATASET_FILE,
        num_samples_per_class: int = 40,
    ) -> None:
        """
        Genera un dataset sintético cinemático válido para validar la arquitectura TCN,
        entrenar el modelo piloto y medir los benchmarks de latencia sub-5ms en GPU de inmediato.
        """
        print("\n[GENERANDO DATASET SINTETICO BENCHMARK]...")
        num_classes = len(PILOT_VOCABULARY)
        total_samples = num_classes * num_samples_per_class

        # Creamos trayectorias temporales con perfiles sinusoidales específicos por clase
        t_steps = np.linspace(0, np.pi, TEMPORAL_WINDOW_SIZE)
        train_x_list = []
        train_y_list = []
        val_x_list = []
        val_y_list = []

        for class_idx in range(num_classes):
            for s in range(num_samples_per_class):
                # Generar una trayectoria base suave
                freq = class_idx + 1
                base_curve = np.sin(freq * t_steps) * 0.5
                noise = np.random.normal(0, 0.02, size=(RAW_SPATIAL_FEATURES, TEMPORAL_WINDOW_SIZE))

                # Matriz espacial (147, 40)
                spatial = np.tile(base_curve, (RAW_SPATIAL_FEATURES, 1)) + noise
                # Diferenciales: velocidad y aceleración
                vel = np.zeros_like(spatial)
                vel[:, 1:] = spatial[:, 1:] - spatial[:, :-1]
                acc = np.zeros_like(vel)
                acc[:, 1:] = vel[:, 1:] - vel[:, :-1]

                # Full input tensor: (441, 40)
                full_sample = np.vstack([spatial, vel, acc]).astype(np.float32)

                if s < int(num_samples_per_class * 0.8):
                    train_x_list.append(full_sample)
                    train_y_list.append(class_idx)
                else:
                    val_x_list.append(full_sample)
                    val_y_list.append(class_idx)

        payload = {
            "train_x": torch.tensor(np.array(train_x_list), dtype=torch.float32),
            "train_y": torch.tensor(np.array(train_y_list), dtype=torch.long),
            "val_x": torch.tensor(np.array(val_x_list), dtype=torch.float32),
            "val_y": torch.tensor(np.array(val_y_list), dtype=torch.long),
            "classes": PILOT_VOCABULARY,
            "window_size": TEMPORAL_WINDOW_SIZE,
            "input_features": INPUT_FEATURES_PER_FRAME,
        }

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        torch.save(payload, output_file)
        print(f"[EXITO] Dataset sintético generado en {output_file}")
        print(f"  - Train: {payload['train_x'].shape}, Val: {payload['val_x'].shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generador de Dataset para Traductor Dinámico LSC")
    parser.add_argument("--synthetic", action="store_true", help="Generar dataset sintético para pruebas de arquitectura")
    args = parser.parse_args()

    if args.synthetic:
        DatasetGenerator.generate_synthetic_benchmark_dataset()
    else:
        generator = DatasetGenerator()
        generator.build_dataset()


if __name__ == "__main__":
    main()
