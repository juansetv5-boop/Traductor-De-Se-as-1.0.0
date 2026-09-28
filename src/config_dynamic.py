"""
config_dynamic.py — Constantes y configuración global para el Traductor Dinámico Alpha LSC.
Centraliza parámetros de cinemática temporal, arquitectura de red TCN, triggers de Apex y TTS asíncrono.
"""

from __future__ import annotations

import os
from typing import Final, List

# ─── Cámara y Video ──────────────────────────────────────────────────────────
CAMERA_INDEX: Final[int] = 0
CAMERA_WIDTH: Final[int] = 640
CAMERA_HEIGHT: Final[int] = 480
TARGET_FPS: Final[int] = 30

# ─── Ventana Temporal y Cinemática ───────────────────────────────────────────
# T = 40 fotogramas a 30 FPS (~1.33 segundos), tamaño óptimo para capturar
# preparación, stroke (ápice) y retracción de señas dinámicas en LSC.
TEMPORAL_WINDOW_SIZE: Final[int] = 40
STRIDE_STEP: Final[int] = 1

# ─── Vocabulario Piloto LSC ──────────────────────────────────────────────────
# 5 señas esenciales de alta frecuencia + clase Neutro (reposo)
PILOT_VOCABULARY: Final[List[str]] = [
    "Neutro",        # Clase 0: Reposo / transición neutra
    "Hola",          # Clase 1: Saludo con oscilación o trayectoria abierta
    "Gracias",       # Clase 2: Toque de mentón/pecho hacia el frente
    "Por favor",     # Clase 3: Movimiento circular sobre el pecho
    "Ayuda",         # Clase 4: Bimanual (palma sobre puño hacia adelante/arriba)
    "Buenos dias",   # Clase 5: Compuesto (frente/pecho hacia apertura)
]

NUM_CLASSES: Final[int] = len(PILOT_VOCABULARY)

# ─── MediaPipe Holistic Landmarks ────────────────────────────────────────────
# Pose superior: Nariz (0), Hombro Izq (11), Hombro Der (12),
# Codo Izq (13), Codo Der (14), Muñeca Izq (15), Muñeca Der (16)
POSE_LANDMARK_INDICES: Final[List[int]] = [0, 11, 12, 13, 14, 15, 16]
NUM_POSE_POINTS: Final[int] = len(POSE_LANDMARK_INDICES)  # 7 puntos

NUM_HAND_LANDMARKS: Final[int] = 21  # 21 landmarks por mano
NUM_COORDS: Final[int] = 3           # X, Y, Z

# Total puntos 3D por frame: 7 (pose) + 21 (mano izq) + 21 (mano der) = 49 puntos
TOTAL_POINTS: Final[int] = NUM_POSE_POINTS + (2 * NUM_HAND_LANDMARKS)
RAW_SPATIAL_FEATURES: Final[int] = TOTAL_POINTS * NUM_COORDS  # 49 * 3 = 147

# Entrada por frame: Posición normalizada (147) + Velocidad (147) + Aceleración (147)
INPUT_FEATURES_PER_FRAME: Final[int] = RAW_SPATIAL_FEATURES * 3  # 441 características

# ─── Rutas del Pipeline ──────────────────────────────────────────────────────
DATA_DIR: Final[str] = "data"
VIDEOS_DIR: Final[str] = os.path.join(DATA_DIR, "dynamic_videos")
DATASET_FILE: Final[str] = os.path.join(DATA_DIR, "dynamic_dataset.pt")
MODELS_DIR: Final[str] = "models"
MODEL_WEIGHTS_PATH: Final[str] = os.path.join(MODELS_DIR, "tcn_lsc_model.pth")
TRACED_MODEL_PATH: Final[str] = os.path.join(MODELS_DIR, "tcn_lsc_traced.pt")

# ─── Arquitectura TCN y Entrenamiento ────────────────────────────────────────
TCN_CHANNEL_SIZES: Final[List[int]] = [64, 128, 128, 64]
TCN_KERNEL_SIZE: Final[int] = 3
TCN_DROPOUT: Final[float] = 0.2
LEARNING_RATE: Final[float] = 1e-3
BATCH_SIZE: Final[int] = 16
NUM_EPOCHS: Final[int] = 40
WEIGHT_DECAY: Final[float] = 1e-4

# ─── Motor de Trigger y Apex Cinemático ──────────────────────────────────────
CONFIDENCE_THRESHOLD: Final[float] = 0.85      # Probabilidad mínima requerida
MIN_STROKE_VELOCITY: Final[float] = 0.04       # Velocidad mínima alcanzada durante el stroke
APEX_DECELERATION_RATIO: Final[float] = 0.40   # Desaceleración post-ápice (relativa al pico)
TRIGGER_COOLDOWN_FRAMES: Final[int] = 35       # Cooldown anti-rebote tras emitir una seña (~1.1 s)

# ─── TTS ──────────────────────────────────────────────────────────────────────
TTS_RATE: Final[int] = 160                     # Palabras por minuto
TTS_VOLUME: Final[float] = 1.0
