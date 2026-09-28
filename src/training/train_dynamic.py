"""
train_dynamic.py — Pipeline de entrenamiento en PyTorch para el modelo TCN de LSC.
Incluye DataLoader, optimizador AdamW, Cosine Annealing, cálculo de métricas (Accuracy, F1-Score),
guardado de pesos y exportación de TorchScript (JIT Trace).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, TensorDataset

from src.config_dynamic import (
    BATCH_SIZE,
    DATASET_FILE,
    LEARNING_RATE,
    MODEL_WEIGHTS_PATH,
    MODELS_DIR,
    NUM_CLASSES,
    NUM_EPOCHS,
    PILOT_VOCABULARY,
    TEMPORAL_WINDOW_SIZE,
    TRACED_MODEL_PATH,
    WEIGHT_DECAY,
)
from src.models.tcn_model import LSCDynamicTCNClassifier, build_tcn_model


class FocalLoss(nn.Module):
    """Focal Loss para manejar desbalances y enfocarse en ejemplos cinemáticos difíciles."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none", weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class DynamicTrainer:
    """Orquesta el ciclo de vida de entrenamiento, validación y exportación de la TCN."""

    def __init__(
        self,
        dataset_path: str = DATASET_FILE,
        epochs: int = NUM_EPOCHS,
        batch_size: int = BATCH_SIZE,
        lr: float = LEARNING_RATE,
        device: str | None = None,
    ) -> None:
        self.dataset_path = dataset_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[INFO] Dispositivo de entrenamiento: {self.device}")
        os.makedirs(MODELS_DIR, exist_ok=True)

    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """Carga el dataset de tensores y devuelve los DataLoaders de Train y Val."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"No se encontró el dataset en '{self.dataset_path}'. "
                f"Ejecute primero dataset_generator.py (--synthetic o con videos reales)."
            )

        payload = torch.load(self.dataset_path)
        train_x = payload["train_x"]
        train_y = payload["train_y"]
        val_x = payload["val_x"]
        val_y = payload["val_y"]

        print(f"[DATOS] Cargadas {len(train_x)} muestras de Train y {len(val_x)} de Validación.")

        train_ds = TensorDataset(train_x, train_y)
        val_ds = TensorDataset(val_x, val_y)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader

    def train(self) -> LSCDynamicTCNClassifier:
        """Ejecuta el ciclo de entrenamiento completo."""
        train_loader, val_loader = self.load_data()
        model = build_tcn_model(self.device)

        criterion = FocalLoss(gamma=1.5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-5)

        best_val_loss = float("inf")
        best_val_f1 = 0.0

        print("\n" + "=" * 75)
        print("  INICIANDO ENTRENAMIENTO DE RED TCN (LSC)")
        print("=" * 75)

        for epoch in range(1, self.epochs + 1):
            model.train()
            train_loss = 0.0
            correct_train = 0
            total_train = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

                train_loss += loss.item() * len(batch_y)
                preds = outputs.argmax(dim=-1)
                correct_train += (preds == batch_y).sum().item()
                total_train += len(batch_y)

            train_loss /= total_train
            train_acc = (correct_train / total_train) * 100.0

            # Validación
            model.eval()
            val_loss = 0.0
            val_preds_list = []
            val_targets_list = []

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item() * len(batch_y)

                    preds = outputs.argmax(dim=-1)
                    val_preds_list.extend(preds.cpu().numpy())
                    val_targets_list.extend(batch_y.cpu().numpy())

            val_loss /= len(val_loader.dataset)
            val_preds = np.array(val_preds_list)
            val_targets = np.array(val_targets_list)
            val_acc = (val_preds == val_targets).mean() * 100.0
            val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)

            scheduler.step()

            if epoch % 5 == 0 or epoch == self.epochs or val_loss < best_val_loss:
                print(
                    f"Epoch [{epoch:02d}/{self.epochs:02d}] | "
                    f"Train Loss: {train_loss:.4f} (Acc: {train_acc:.1f}%) | "
                    f"Val Loss: {val_loss:.4f} (Acc: {val_acc:.1f}%, F1: {val_f1:.3f})"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_f1 = val_f1
                torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)

        print("\n" + "=" * 75)
        print(f"[ENTRENAMIENTO COMPLETADO] Mejor Val Loss: {best_val_loss:.4f} | Mejor F1: {best_val_f1:.4f}")
        print(f"[GUARDADO] Pesos guardados en: {MODEL_WEIGHTS_PATH}")

        # Cargar los mejores pesos para exportación
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=self.device))
        self.export_traced_model(model)
        return model

    def export_traced_model(self, model: LSCDynamicTCNClassifier) -> None:
        """Exporta el modelo optimizado con TorchScript JIT para inferencia ultra-rápida."""
        model.eval()
        dummy_input = torch.randn(1, 441, TEMPORAL_WINDOW_SIZE, device=self.device)
        try:
            traced_model = torch.jit.trace(model, dummy_input)
            traced_model.save(TRACED_MODEL_PATH)
            print(f"[EXPORTACION] Modelo TorchScript guardado en: {TRACED_MODEL_PATH}")
        except Exception as e:
            print(f"[AVISO] No se pudo exportar a TorchScript: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenador de TCN para LSC")
    parser.add_argument("--epochs", type=int, default=25, help="Número de épocas")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Tamaño de lote")
    args = parser.parse_args()

    trainer = DynamicTrainer(epochs=args.epochs, batch_size=args.batch_size)
    trainer.train()


if __name__ == "__main__":
    main()
