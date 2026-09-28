"""
benchmark_latency.py — Perfilado riguroso de latencia y presupuesto computacional (sub-5ms).
Evalúa tiempos de inferencia en GPU (CUDA) y CPU sobre el modelo TCN para secuencias de T=40 frames.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from src.config_dynamic import (
    INPUT_FEATURES_PER_FRAME,
    TEMPORAL_WINDOW_SIZE,
    TRACED_MODEL_PATH,
)
from src.models.tcn_model import build_tcn_model


def benchmark_model(num_runs: int = 500) -> None:
    print("\n" + "=" * 75)
    print("  BENCHMARK Y PERFILADO DE LATENCIA — RED TCN LSC")
    print("=" * 75)

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
        print(f"[HARDWARE] GPU Detectada: {torch.cuda.get_device_name(0)}")
        print(f"[HARDWARE] VRAM Total: {torch.cuda.get_device_properties(0).total_memory / (1024**2):.1f} MB")
    else:
        print("[HARDWARE] Aceleración CUDA no detectada en este entorno de PyTorch. Ejecutando en CPU.")

    for dev_name in devices:
        dev = torch.device(dev_name)
        model = build_tcn_model(dev)
        model.eval()

        # Batch de 1 ventana temporal (inferencia en vivo por frame)
        dummy_input = torch.randn(1, INPUT_FEATURES_PER_FRAME, TEMPORAL_WINDOW_SIZE, device=dev)

        # Calentamiento (Warmup)
        for _ in range(50):
            _ = model(dummy_input)

        if dev_name == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            if dev_name == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)  # ms

        mean_ms = np.mean(times)
        p50_ms = np.percentile(times, 50)
        p95_ms = np.percentile(times, 95)
        p99_ms = np.percentile(times, 99)

        print(f"\n--- Resultados en Dispositivo: [{dev_name.upper()}] ({num_runs} iteraciones) ---")
        print(f"  * Latencia Media:        {mean_ms:.3f} ms")
        print(f"  * Latencia p50 (Mediana): {p50_ms:.3f} ms")
        print(f"  * Latencia p95:           {p95_ms:.3f} ms")
        print(f"  * Latencia p99:           {p99_ms:.3f} ms")
        print(f"  * Inferencia / seg (FPS): {1000.0 / mean_ms:.1f} FPS")

        if mean_ms < 5.0:
            print(f"  [RESULTADO] Cumple con el presupuesto objetivo de latencia sub-5ms.")
        else:
            print(f"  [AVISO] Supera el objetivo de 5ms.")


if __name__ == "__main__":
    benchmark_model()
