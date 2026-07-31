"""
main.py — Orquestador principal del proyecto Traductor Alpha de Lenguaje de Señas.

Ofrece un menú interactivo en consola para ejecutar secuencialmente cualquier fase del pipeline:
    [1] Verificación de Captura y Normalización de Keypoints
    [2] Recolección de Muestras a CSV
    [3] Entrenamiento y Evaluación del Modelo ML
    [4] Traducción en Tiempo Real con Voz (TTS)
"""

from __future__ import annotations

import os
import sys


def print_banner() -> None:
    """Imprime el encabezado principal en consola."""
    print("=" * 75)
    print("  TRADUCTOR ALPHA — LENGUAJE DE SEÑAS (DACTILOLOGÍA ESTÁTICA) A TEXTO Y VOZ")
    print("=" * 75)


def show_menu() -> None:
    """Muestra las opciones del menú principal."""
    print("\nSeleccione la fase que desea ejecutar:\n")
    print("  [1] Fase 1: Verificación de Extracción y Normalización 3D (keypoint_extractor)")
    print("  [2] Fase 2: Recolección de Datos de Señas a CSV (data_collector)")
    print("  [3] Fase 3: Entrenamiento y Evaluación del Clasificador ML (model_trainer)")
    print("  [4] Fase 4: Traducción en Tiempo Real + Lectura por Voz TTS (inference)")
    print("  [0] Salir")
    print("-" * 75)


def main() -> None:
    """Ejecuta el menú principal del sistema."""
    print_banner()

    while True:
        show_menu()
        choice: str = input("Ingrese una opción [0-4]: ").strip()

        if choice == "1":
            print("\n[INICIANDO] Fase 1: Extractor de Keypoints 3D...")
            from src.keypoint_extractor import _run_verification
            _run_verification()

        elif choice == "2":
            print("\n[INICIANDO] Fase 2: Recolector de Datos...")
            from src.data_collector import main as run_collector
            run_collector()

        elif choice == "3":
            print("\n[INICIANDO] Fase 3: Entrenador del Modelo ML...")
            from src.model_trainer import main as run_trainer
            run_trainer()

        elif choice == "4":
            print("\n[INICIANDO] Fase 4: Inferencia en Tiempo Real + TTS...")
            from src.inference import main as run_inference
            run_inference()

        elif choice == "0":
            print("\n¡Gracias por utilizar el Traductor Alpha de Lenguaje de Señas!")
            sys.exit(0)

        else:
            print("\n[Opción no válida] Ingrese un número entre 0 y 4.")


if __name__ == "__main__":
    main()
