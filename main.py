"""
main.py — Orquestador principal del proyecto Traductor Alpha de Lenguaje de Señas.
Soporta tanto el subsistema de Dactilología Estática (RandomForest)
como el nuevo subsistema Dinámico Continuo LSC (MediaPipe Holistic + PyTorch TCN GPU + Apex TTS).
"""

from __future__ import annotations

import os
import sys


def print_banner() -> None:
    """Imprime el encabezado principal en consola."""
    print("=" * 80)
    print("      TRADUCTOR ALPHA — LENGUA DE SEÑAS COLOMBIANA (LSC)")
    print("   [Estático: Dactilología RF]  |  [Dinámico: TCN GPU + Apex TTS]")
    print("=" * 80)


def show_menu() -> None:
    """Muestra las opciones del menú principal."""
    print("\n--- MODO 1: DACTILOLOGÍA ESTÁTICA (RandomForest) ---")
    print("  [1] Verificación de Keypoints 3D Mano (keypoint_extractor)")
    print("  [2] Recolección de Datos Estáticos a CSV (data_collector)")
    print("  [3] Entrenamiento Clasificador RandomForest (model_trainer)")
    print("  [4] Traductor Estático en Vivo + TTS (inference)")

    print("\n--- MODO 2: GESTOS DINÁMICOS CONTINUOS LSC (GPU TCN + APEX TTS) ---")
    print("  [5] Recolección de Videos LSC por Sujetos (video_collector)")
    print("  [6] Generador de Dataset Temporal (T=40, dX, d^2X) (dataset_generator)")
    print("  [7] Entrenamiento de Red TCN en GPU (train_dynamic)")
    print("  [8] Traductor Dinámico en Vivo + Apex Trigger + Voz (live_engine)")
    print("  [9] Benchmark de Latencia y Presupuesto Sub-5ms (benchmark_latency)")
    print("  [10] GhostPath Visual Trainer (Trayectoria Fantasma LSC) (ghost_trainer)")

    print("\n  [0] Salir")
    print("-" * 80)


def main() -> None:
    """Ejecuta el menú principal del sistema."""
    print_banner()

    while True:
        show_menu()
        choice: str = input("Ingrese una opción [0-10]: ").strip()

        # ─── Modo Estático ───────────────────────────────────────────────────
        if choice == "1":
            print("\n[INICIANDO] Verificación de Keypoints 3D...")
            from src.keypoint_extractor import _run_verification
            _run_verification()

        elif choice == "2":
            print("\n[INICIANDO] Recolector de Datos Estáticos...")
            from src.data_collector import main as run_collector
            run_collector()

        elif choice == "3":
            print("\n[INICIANDO] Entrenador RandomForest...")
            from src.model_trainer import main as run_trainer
            run_trainer()

        elif choice == "4":
            print("\n[INICIANDO] Inferencia Estática en Tiempo Real...")
            from src.inference import main as run_inference
            run_inference()

        # ─── Modo Dinámico LSC ───────────────────────────────────────────────
        elif choice == "5":
            print("\n[INICIANDO] Recolección Dinámica por Sujetos...")
            from src.data_collection.video_collector import main as run_vid_collector
            run_vid_collector()

        elif choice == "6":
            print("\n[INICIANDO] Generador de Dataset Temporal...")
            from src.data_collection.dataset_generator import main as run_dataset_gen
            run_dataset_gen()

        elif choice == "7":
            print("\n[INICIANDO] Entrenamiento TCN GPU...")
            from src.training.train_dynamic import main as run_train_dynamic
            run_train_dynamic()

        elif choice == "8":
            print("\n[INICIANDO] Motor Dinámico LSC en Vivo con Apex TTS...")
            from src.live_inference.live_engine import main as run_live_dynamic
            run_live_dynamic()

        elif choice == "9":
            print("\n[INICIANDO] Benchmark de Latencia...")
            from tests.benchmark_latency import benchmark_model
            benchmark_model()

        elif choice == "10":
            print("\n[INICIANDO] GhostPath Visual Trainer (Trayectoria Fantasma LSC)...")
            from src.ui.ghost_trainer import main as run_ghost_trainer
            run_ghost_trainer()

        elif choice == "0":
            print("\n¡Gracias por utilizar el Traductor Alpha LSC!")
            sys.exit(0)

        else:
            print("\n[Opción no válida] Ingrese un número entre 0 y 10.")


if __name__ == "__main__":
    main()
