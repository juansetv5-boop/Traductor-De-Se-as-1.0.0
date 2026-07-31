"""
keypoint_extractor.py — Extracción y normalización de keypoints 3D con MediaPipe Hands.

Pipeline de procesamiento:
    Frame BGR → MediaPipe Hands → 21 Landmarks (X, Y, Z)
    → Centrado (restar muñeca) → Escalado (dividir por max absoluto)
    → Vector normalizado de 63 floats en rango [-1, 1].

Garantías matemáticas:
    - Invarianza a la posición de la mano en el frame (centrado respecto a muñeca).
    - Invarianza a la distancia mano-cámara (escalado por magnitud máxima).
    - Vector siempre unidimensional de exactamente NUM_FEATURES (63) elementos.
"""

from __future__ import annotations

import sys
import time
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from numpy.typing import NDArray

# Importar constantes desde la raíz del proyecto
sys.path.insert(0, ".")
from config import (
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    MP_MAX_HANDS,
    MP_MIN_DETECTION_CONF,
    MP_MIN_TRACKING_CONF,
    NUM_COORDS,
    NUM_FEATURES,
    NUM_LANDMARKS,
)


class KeypointExtractor:
    """
    Extractor de keypoints 3D de la mano usando MediaPipe Hands.

    Responsabilidades:
        1. Inicializar y gestionar el ciclo de vida del modelo MediaPipe.
        2. Procesar frames BGR y devolver resultados crudos de MediaPipe.
        3. Convertir landmarks a un vector numpy normalizado de 63 valores.
        4. Dibujar landmarks sobre el frame para visualización (opcional).
    """

    def __init__(
        self,
        max_hands: int = MP_MAX_HANDS,
        min_detection_confidence: float = MP_MIN_DETECTION_CONF,
        min_tracking_confidence: float = MP_MIN_TRACKING_CONF,
    ) -> None:
        """Inicializa el detector MediaPipe Hands con la configuración dada."""
        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles

        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_frame(self, frame_bgr: NDArray[np.uint8]) -> object:
        """
        Procesa un frame BGR y devuelve el objeto de resultados de MediaPipe.

        MediaPipe requiere entrada en RGB. La conversión se hace internamente
        para mantener la interfaz simple (el caller siempre pasa BGR de OpenCV).

        Args:
            frame_bgr: Frame en formato BGR (numpy array H×W×3, uint8).

        Returns:
            Objeto results de MediaPipe con atributo multi_hand_landmarks.
        """
        frame_rgb: NDArray[np.uint8] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # writeable=False optimiza el rendimiento al evitar copias internas
        frame_rgb.flags.writeable = False
        results = self._hands.process(frame_rgb)
        frame_rgb.flags.writeable = True
        return results

    def extract(
        self,
        frame_bgr: NDArray[np.uint8],
        auto_flip_left_hand: bool = True,
    ) -> Optional[NDArray[np.float64]]:
        """
        Extrae y normaliza los keypoints 3D de la primera mano detectada.

        Pipeline completo:
            1. Procesar frame con MediaPipe → obtener landmarks crudos y handedness.
            2. Convertir 21 landmarks a array (21, 3).
            3. Invariancia Zurda/Diestra: Si la mano es izquierda y auto_flip_left_hand es True,
               se invierte el eje X para mapearla al espacio vectorial de mano derecha.
            4. Centrar: restar coordenadas de Landmark 0 (muñeca) a todos los puntos.
            5. Escalar: dividir por el valor absoluto máximo del vector centrado.
            6. Aplanar a vector 1D de 63 elementos.

        Args:
            frame_bgr: Frame en formato BGR (numpy array H×W×3, uint8).
            auto_flip_left_hand: Invertir eje X para manos izquierdas.

        Returns:
            Vector numpy de 63 floats normalizados en [-1, 1], o None si no
            se detectó ninguna mano.
        """
        results = self.process_frame(frame_bgr)

        # Sin manos detectadas → retornar None explícitamente
        if not results.multi_hand_landmarks:
            return None

        # Tomar solo la primera mano detectada
        hand_landmarks = results.multi_hand_landmarks[0]

        # Paso 1: Extraer las 21 coordenadas (X, Y, Z) en un array (21, 3)
        raw_coords: NDArray[np.float64] = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float64,
        )

        # Verificación de integridad dimensional
        assert raw_coords.shape == (NUM_LANDMARKS, NUM_COORDS), (
            f"Forma inesperada: {raw_coords.shape}, esperada ({NUM_LANDMARKS}, {NUM_COORDS})"
        )

        # Paso 2: Centrado — restar coordenadas de la muñeca (Landmark 0)
        wrist: NDArray[np.float64] = raw_coords[0]
        centered: NDArray[np.float64] = raw_coords - wrist

        # Paso 3: Invariancia Zurda/Diestra (Flipping eje X para manos izquierdas)
        if auto_flip_left_hand and results.multi_handedness:
            handedness_label: str = results.multi_handedness[0].classification[0].label
            if handedness_label == "Left":
                centered[:, 0] = -centered[:, 0]


        # Paso 3: Escalado — dividir por la magnitud absoluta máxima
        max_abs: np.float64 = np.max(np.abs(centered))

        if max_abs < 1e-9:
            # Todos los puntos colapsaron en la muñeca → vector degenerado
            # Retornar zeros en lugar de producir NaN/Inf por división
            return np.zeros(NUM_FEATURES, dtype=np.float64)

        normalized: NDArray[np.float64] = centered / max_abs

        # Paso 4: Aplanar a vector unidimensional de 63 elementos
        flat: NDArray[np.float64] = normalized.flatten()

        assert flat.shape == (NUM_FEATURES,), (
            f"Vector final con forma {flat.shape}, esperada ({NUM_FEATURES},)"
        )

        return flat

    def draw_landmarks(
        self,
        frame_bgr: NDArray[np.uint8],
        results: object,
    ) -> NDArray[np.uint8]:
        """
        Dibuja los landmarks y conexiones de MediaPipe sobre el frame.

        No modifica el frame original; trabaja sobre una copia.

        Args:
            frame_bgr: Frame BGR original.
            results: Objeto de resultados de MediaPipe (de process_frame).

        Returns:
            Copia del frame con landmarks dibujados.
        """
        annotated: NDArray[np.uint8] = frame_bgr.copy()

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                # Dibujar conexiones entre landmarks
                self._mp_drawing.draw_landmarks(
                    image=annotated,
                    landmark_list=hand_lms,
                    connections=self._mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self._mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=self._mp_drawing_styles.get_default_hand_connections_style(),
                )

        return annotated

    def release(self) -> None:
        """Libera los recursos internos de MediaPipe."""
        self._hands.close()


# ──────────────────────────────────────────────────────────────────────────────
# Script de verificación — ejecutar con: python -m src.keypoint_extractor
# ──────────────────────────────────────────────────────────────────────────────
def _run_verification() -> None:
    """
    Abre la webcam, extrae keypoints normalizados y los muestra en consola.

    Verificaciones que realiza:
        - Apertura correcta de la cámara.
        - Detección de mano y extracción de 63 valores.
        - Rango de valores en [-1, 1] (normalización correcta).
        - Primeros 3 valores siempre [0.0, 0.0, 0.0] (centrado en muñeca).
        - FPS del pipeline completo.
    """
    extractor = KeypointExtractor()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara. Verifica CAMERA_INDEX en config.py.")
        sys.exit(1)

    print("=" * 70)
    print(" VERIFICACIÓN — KeypointExtractor (Fase 1)")
    print(" Controles: [Q] Salir")
    print("=" * 70)

    prev_time: float = time.time()

    try:
        while True:
            ret: bool
            frame: NDArray[np.uint8]
            ret, frame = cap.read()

            if not ret:
                print("[ERROR] No se pudo leer frame de la cámara.")
                break

            # Voltear horizontalmente para efecto espejo (más intuitivo)
            frame = cv2.flip(frame, 1)

            # Procesar y obtener resultados crudos para dibujar
            results = extractor.process_frame(frame)

            # Extraer vector normalizado
            keypoints: Optional[NDArray[np.float64]] = extractor.extract(frame)

            # Dibujar landmarks sobre el frame
            annotated: NDArray[np.uint8] = extractor.draw_landmarks(frame, results)

            # Calcular FPS
            current_time: float = time.time()
            fps: float = 1.0 / max(current_time - prev_time, 1e-9)
            prev_time = current_time

            # ── HUD sobre el frame ──
            # Fondo semitransparente para el texto
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (CAMERA_WIDTH, 90), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

            cv2.putText(
                annotated, f"FPS: {fps:.1f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )

            if keypoints is not None:
                # Verificar invariantes
                wrist_zero: bool = np.allclose(keypoints[:3], 0.0, atol=1e-9)
                in_range: bool = bool(np.all(np.abs(keypoints) <= 1.0 + 1e-9))

                status_color = (0, 255, 0) if (wrist_zero and in_range) else (0, 0, 255)
                status_text = "OK" if (wrist_zero and in_range) else "FALLO"

                cv2.putText(
                    annotated,
                    f"Vector: {keypoints.shape[0]} features | Norm: {status_text}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2,
                )
                cv2.putText(
                    annotated,
                    f"Rango: [{keypoints.min():.3f}, {keypoints.max():.3f}]",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
                )

                # Imprimir en consola cada ~1 segundo (evitar spam)
                if int(current_time * 2) % 2 == 0 and int(prev_time * 2) % 2 != 0:
                    print(f"\n[FPS: {fps:.1f}] Vector ({keypoints.shape[0]} valores):")
                    print(f"  Muñeca (L0): [{keypoints[0]:.6f}, {keypoints[1]:.6f}, {keypoints[2]:.6f}]")
                    print(f"  Rango:       [{keypoints.min():.6f}, {keypoints.max():.6f}]")
                    print(f"  Muñeca=0:    {wrist_zero} | En [-1,1]: {in_range}")
            else:
                cv2.putText(
                    annotated, "No se detecta mano",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2,
                )

            cv2.imshow("Traductor Alpha - Verificacion Fase 1", annotated)

            # Salir con 'Q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.release()
        print("\n[INFO] Recursos liberados correctamente.")


if __name__ == "__main__":
    _run_verification()
