"""
tcn_model.py — Red Convolucional Temporal Causal (TCN) optimizada para inferencia en GPU.
Diseñada para modelar secuencias cinemáticas temporales continuas (LSC) en tiempo real con latencia sub-2ms.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config_dynamic import (
    INPUT_FEATURES_PER_FRAME,
    NUM_CLASSES,
    TCN_CHANNEL_SIZES,
    TCN_DROPOUT,
    TCN_KERNEL_SIZE,
    TEMPORAL_WINDOW_SIZE,
)


class Chomp1d(nn.Module):
    """
    Recorta el exceso de padding del extremo derecho para garantizar convoluciones causales estrictas.
    Ningún fotograma futuro puede influir en la predicción del fotograma presente o pasado.
    """

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Bloque residual de TCN con convoluciones dilatadas causales, normalización y dropout.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.bn1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.bn2,
            self.relu2,
            self.dropout2,
        )

        # Proyección 1x1 si las dimensiones de entrada y salida no coinciden
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Pila jerárquica de bloques temporales con factores de dilatación exponenciales (1, 2, 4, 8, ...).
    """

    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class LSCDynamicTCNClassifier(nn.Module):
    """
    Clasificador completo de Lengua de Señas Colombiana basado en TCN.
    Recibe tensor (Batch, Features=441, Time=40) y entrega logits de clase (Batch, NumClasses).
    """

    def __init__(
        self,
        input_features: int = INPUT_FEATURES_PER_FRAME,
        num_classes: int = NUM_CLASSES,
        num_channels: List[int] = TCN_CHANNEL_SIZES,
        kernel_size: int = TCN_KERNEL_SIZE,
        dropout: float = TCN_DROPOUT,
    ) -> None:
        super().__init__()
        self.tcn = TemporalConvNet(
            num_inputs=input_features,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        last_channel = num_channels[-1]
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(last_channel, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Paso hacia adelante:
            x: Tensor de entrada (Batch, Features=441, Time=40)
        Retorna:
            Logits sin normalizar (Batch, NumClasses)
        """
        # Salida TCN: (Batch, LastChannel, Time)
        features = self.tcn(x)
        # Pooling sobre la dimensión temporal: (Batch, LastChannel, 1)
        pooled = self.pooling(features).squeeze(-1)
        # Clasificador lineal: (Batch, NumClasses)
        logits = self.head(pooled)
        return logits

    @torch.no_grad()
    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Calcula probabilidades Softmax para inferencia en tiempo real."""
        self.eval()
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


def build_tcn_model(device: torch.device) -> LSCDynamicTCNClassifier:
    """Instancia y transfiere el modelo TCN al dispositivo (CUDA/CPU)."""
    model = LSCDynamicTCNClassifier(
        input_features=INPUT_FEATURES_PER_FRAME,
        num_classes=NUM_CLASSES,
        num_channels=TCN_CHANNEL_SIZES,
        kernel_size=TCN_KERNEL_SIZE,
        dropout=TCN_DROPOUT,
    )
    return model.to(device)


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TEST TCN] Dispositivo seleccionado: {dev}")
    model = build_tcn_model(dev)

    # Tensor de prueba: 2 secuencias de 441 características y 40 fotogramas
    dummy_input = torch.randn(2, INPUT_FEATURES_PER_FRAME, TEMPORAL_WINDOW_SIZE, device=dev)
    output = model(dummy_input)
    probs = model.predict_probabilities(dummy_input)

    print(f"  - Entrada:  {dummy_input.shape}")
    print(f"  - Logits:   {output.shape}")
    print(f"  - Probs:    {probs.shape}")
    print("  - Parametros totales:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("[TEST TCN] Modelo compilado y validado exitosamente.")
