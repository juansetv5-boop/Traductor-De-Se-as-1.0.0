"""
config.py — Constantes globales del proyecto Traductor Alpha.
Centraliza toda la configuración para evitar valores mágicos dispersos en el código.
"""

from typing import Final

# ─── Cámara ───────────────────────────────────────────────────────────────────
CAMERA_INDEX: Final[int] = 0                  # Índice del dispositivo de captura
CAMERA_WIDTH: Final[int] = 640                # Ancho del frame en píxeles
CAMERA_HEIGHT: Final[int] = 480               # Alto del frame en píxeles

# ─── MediaPipe Hands ──────────────────────────────────────────────────────────
MP_MAX_HANDS: Final[int] = 1                  # Máximo de manos a detectar
MP_MIN_DETECTION_CONF: Final[float] = 0.7     # Confianza mínima para detección
MP_MIN_TRACKING_CONF: Final[float] = 0.5      # Confianza mínima para tracking

# ─── Extracción de Keypoints ──────────────────────────────────────────────────
NUM_LANDMARKS: Final[int] = 21                # Puntos clave por mano (MediaPipe)
NUM_COORDS: Final[int] = 3                    # Coordenadas por landmark (X, Y, Z)
NUM_FEATURES: Final[int] = NUM_LANDMARKS * NUM_COORDS  # 63 características totales

# ─── Alfabeto (Dactilología Estática) ─────────────────────────────────────────
# Se omiten J y Z porque requieren movimiento dinámico que un vector
# estático de 63 features no puede representar de forma fiable.
STATIC_ALPHABET: Final[list[str]] = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y",
]

# ─── Debounce (Estabilidad de Predicción) ─────────────────────────────────────
DEBOUNCE_FRAMES: Final[int] = 10              # Frames consecutivos con la misma
                                               # predicción para confirmar una letra

# ─── Rutas de Archivos ────────────────────────────────────────────────────────
DATA_CSV_PATH: Final[str] = "data/keypoints.csv"
MODEL_PKL_PATH: Final[str] = "models/sign_classifier.pkl"

# ─── Recolección de Datos ─────────────────────────────────────────────────────
SAMPLES_PER_KEY: Final[int] = 1               # Muestras capturadas por pulsación de tecla
TARGET_SAMPLES_PER_CLASS: Final[int] = 350    # Meta de muestras por letra


# ─── TTS ──────────────────────────────────────────────────────────────────────
TTS_RATE: Final[int] = 150                    # Velocidad de habla (palabras por minuto)
