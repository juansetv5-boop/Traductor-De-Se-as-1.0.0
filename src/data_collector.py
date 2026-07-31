"""
data_collector.py — Módulo de recolección de muestras de keypoints a CSV.

Permite capturar muestras de keypoints 3D (63 features normalizados) en tiempo real
y guardarlas en `data/keypoints.csv` asociadas a su etiqueta de letra (A-Z).

Características:
    - Captura en ráfaga (mantenimiento de tecla o ESPACIO).
    - Append automático persistente a CSV con cabeceras f0..f62, label.
    - Control de duplicados/muestras válidas (solo con detección activa de mano).
    - Doble confirmación para borrado de dataset o de clase individual.
    - HUD visual con barra de progreso, contador por letra, total general y badge de grabación.
"""

from __future__ import annotations

import csv
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

# Importar módulos y constantes del proyecto
sys.path.insert(0, ".")
from config import (
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    DATA_CSV_PATH,
    NUM_FEATURES,
    STATIC_ALPHABET,
    TARGET_SAMPLES_PER_CLASS,
)
from src.keypoint_extractor import KeypointExtractor


class DataCollector:
    """
    Gestor de recolección de datos y persistencia en CSV para el dataset de señas.
    """

    def __init__(self, csv_path: str = DATA_CSV_PATH) -> None:
        """
        Inicializa el recolector de datos.

        Args:
            csv_path: Ruta del archivo CSV donde se almacenarán las muestras.
        """
        self.csv_path: str = csv_path
        self._ensure_csv_file()
        self.counts: Dict[str, int] = self.load_counts()

    def _ensure_csv_file(self) -> None:
        """Crea el archivo CSV con cabeceras si no existe."""
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                headers: List[str] = [f"f{i}" for i in range(NUM_FEATURES)] + ["label"]
                writer.writerow(headers)

    def load_counts(self) -> Dict[str, int]:
        """
        Lee el CSV existente y calcula el conteo actual de muestras por cada letra.

        Returns:
            Diccionario {letra: cantidad_de_muestras}.
        """
        counts: Dict[str, int] = {char: 0 for char in STATIC_ALPHABET}
        if not os.path.exists(self.csv_path):
            return counts

        with open(self.csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader, None)  # Omitir cabecera
            if headers is None:
                return counts

            for row in reader:
                if len(row) == NUM_FEATURES + 1:
                    label: str = row[-1].upper()
                    if label in counts:
                        counts[label] += 1

        return counts

    def append_sample(self, keypoints: NDArray[np.float64], label: str) -> bool:
        """
        Añade un vector de 63 keypoints al CSV con su etiqueta.

        Args:
            keypoints: Vector numpy de 63 floats normalizados.
            label: Etiqueta de la letra (ej. "A").

        Returns:
            True si se guardó con éxito, False en caso contrario.
        """
        if keypoints.shape != (NUM_FEATURES,):
            return False

        label_clean: str = label.upper()
        if label_clean not in STATIC_ALPHABET:
            return False

        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            row: List[object] = list(keypoints) + [label_clean]
            writer.writerow(row)

        self.counts[label_clean] += 1
        return True

    def clear_dataset(self) -> None:
        """Elimina todo el contenido del CSV y reinicia los contadores."""
        with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            headers: List[str] = [f"f{i}" for i in range(NUM_FEATURES)] + ["label"]
            writer.writerow(headers)

        self.counts = {char: 0 for char in STATIC_ALPHABET}

    def clear_class(self, label: str) -> None:
        """
        Elimina todas las muestras pertenencientes a una letra específica del CSV.

        Args:
            label: Etiqueta de la letra a limpiar (ej. "A").
        """
        if not os.path.exists(self.csv_path):
            return

        label_clean: str = label.upper()
        rows_to_keep: List[List[str]] = []

        with open(self.csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            if headers:
                rows_to_keep.append(headers)

            for row in reader:
                if len(row) == NUM_FEATURES + 1 and row[-1].upper() != label_clean:
                    rows_to_keep.append(row)

        with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows_to_keep)

        self.counts[label_clean] = 0

    @property
    def total_samples(self) -> int:
        """Devuelve el total acumulado de muestras en el dataset."""
        return sum(self.counts.values())


def draw_hud(
    frame: NDArray[np.uint8],
    active_label: str,
    collector: DataCollector,
    is_recording: bool,
    has_hand: bool,
    confirm_mode: Optional[str] = None,
) -> NDArray[np.uint8]:
    """
    Renderiza la interfaz gráfica (HUD) sobre el frame de la cámara.

    Args:
        frame: Frame BGR original.
        active_label: Letra seleccionada actualmente.
        collector: Instancia de DataCollector con los datos del dataset.
        is_recording: True si se está grabando activamente en este frame.
        has_hand: True si MediaPipe detecta una mano en el frame.
        confirm_mode: "ALL" si se requiere confirmación de borrado total,
                      "CLASS" para borrado de clase, o None.

    Returns:
        Copia del frame con la interfaz renderizada.
    """
    annotated: NDArray[np.uint8] = frame.copy()
    h, w, _ = annotated.shape

    # 1. Overlay superior semitransparente (Header HUD)
    header_h: int = 110
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), (15, 15, 20), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    # 2. Información de la Letra Activa y Estado
    count: int = collector.counts.get(active_label, 0)
    target: int = TARGET_SAMPLES_PER_CLASS
    pct: float = min(1.0, count / max(1, target))

    # Título y Letra Activa
    cv2.putText(
        annotated,
        f"LETRA ACTIVA: [{active_label}]",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
    )

    # Muestras acumuladas por letra
    count_color: Tuple[int, int, int] = (0, 255, 0) if count >= target else (0, 215, 255)
    cv2.putText(
        annotated,
        f"Muestras: {count} / {target}",
        (270, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        count_color,
        2,
    )

    # Total General del Dataset
    cv2.putText(
        annotated,
        f"Total CSV: {collector.total_samples}",
        (480, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
    )

    # 3. Barra de Progreso Visual por Letra
    bar_x, bar_y, bar_w, bar_h = 15, 45, w - 30, 20
    cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)

    fill_w: int = int(bar_w * pct)
    bar_color: Tuple[int, int, int] = (0, 255, 0) if count >= target else (0, 165, 255)
    if fill_w > 0:
        cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
    cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)

    # Porcentaje numérico sobre la barra
    pct_text: str = f"{int(pct * 100)}%"
    cv2.putText(
        annotated,
        pct_text,
        (bar_x + bar_w // 2 - 15, bar_y + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    # 4. Indicador de Grabación / Estado
    if is_recording and has_hand:
        # Badge "● GRABANDO" en rojo parpadeante/sólido
        cv2.circle(annotated, (30, 85), 8, (0, 0, 255), -1)
        cv2.putText(
            annotated,
            "GRABANDO MUESTRAS...",
            (48, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
        )
    elif is_recording and not has_hand:
        cv2.putText(
            annotated,
            "¡MANTEN LA MANO EN PANTALLA PARA GRABAR!",
            (15, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2,
        )
    else:
        cv2.putText(
            annotated,
            "Manten presionada la letra o ESPACIO para grabar",
            (15, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            1,
        )

    # 5. Overlay Inferior: Leyenda de Controles
    footer_h: int = 40
    overlay_foot = annotated.copy()
    cv2.rectangle(overlay_foot, (0, h - footer_h), (w, h), (15, 15, 20), -1)
    cv2.addWeighted(overlay_foot, 0.75, annotated, 0.25, 0, annotated)

    cv2.putText(
        annotated,
        "[A-Z]: Seleccionar/Grabar | [< >]: Cambiar Letra | [SUPR]: Borrar Todo | [BS]: Borrar Letra | [Q]: Salir",
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (220, 220, 220),
        1,
    )

    # 6. Alerta Visual de Doble Confirmación (si aplica)
    if confirm_mode is not None:
        dialog_w, dialog_h = 560, 140
        dx: int = (w - dialog_w) // 2
        dy: int = (h - dialog_h) // 2

        # Fondo del diálogo de advertencia
        cv2.rectangle(annotated, (dx, dy), (dx + dialog_w, dy + dialog_h), (0, 0, 140), -1)
        cv2.rectangle(annotated, (dx, dy), (dx + dialog_w, dy + dialog_h), (255, 255, 255), 2)

        if confirm_mode == "ALL":
            cv2.putText(
                annotated,
                "¡ADVERTENCIA: BORRAR TODO EL DATASET!",
                (dx + 30, dy + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                annotated,
                "¿Esta seguro de eliminar TODAS las muestras?",
                (dx + 30, dy + 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 220, 255),
                1,
            )
        else:  # CLASS
            cv2.putText(
                annotated,
                f"¡ADVERTENCIA: BORRAR MUESTRAS DE '{active_label}'!",
                (dx + 30, dy + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                annotated,
                f"¿Eliminar las {count} muestras de la letra '{active_label}'?",
                (dx + 30, dy + 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 220, 255),
                1,
            )

        cv2.putText(
            annotated,
            "Presione [Y] para CONFIRMAR  |  [N] o [ESC] para CANCELAR",
            (dx + 30, dy + 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )

    return annotated


def main() -> None:
    """Función principal de ejecución del DataCollector."""
    extractor = KeypointExtractor()
    collector = DataCollector()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la camara en indice {CAMERA_INDEX}.")
        sys.exit(1)

    alphabet: List[str] = STATIC_ALPHABET
    current_idx: int = 0
    confirm_mode: Optional[str] = None  # None, "ALL", "CLASS"

    print("=" * 70)
    print(" RECOLECTOR DE DATOS — Traductor Alpha (Fase 2)")
    print(" Manten presionada cualquier letra (A-Z) o ESPACIO para grabar.")
    print(" Usa las flechas IZQ/DER para cambiar de letra.")
    print(" Presiona [SUPR] para borrar todo el dataset (requiere confirmacion).")
    print(" Presiona [BACKSPACE] para borrar la letra activa (requiere confirmacion).")
    print(" Presiona [Q] para salir.")
    print("=" * 70)

    try:
        while True:
            ret: bool
            frame: NDArray[np.uint8]
            ret, frame = cap.read()

            if not ret:
                print("[ERROR] Fallo al capturar frame de la camara.")
                break

            # Efecto espejo
            frame = cv2.flip(frame, 1)

            # Procesar con MediaPipe
            results = extractor.process_frame(frame)
            keypoints: Optional[NDArray[np.float64]] = extractor.extract(frame)
            annotated: NDArray[np.uint8] = extractor.draw_landmarks(frame, results)

            has_hand: bool = keypoints is not None
            active_label: str = alphabet[current_idx]

            # Leer entrada de teclado (espera 1 ms)
            key: int = cv2.waitKey(1) & 0xFF

            is_recording: bool = False

            # ── Modo de Doble Confirmación Activo ──
            if confirm_mode is not None:
                if key in (ord("y"), ord("Y")):
                    if confirm_mode == "ALL":
                        collector.clear_dataset()
                        print("\n[INFO] Dataset borrado completamente.")
                    elif confirm_mode == "CLASS":
                        collector.clear_class(active_label)
                        print(f"\n[INFO] Muestras de letra '{active_label}' borradas.")
                    confirm_mode = None
                elif key in (ord("n"), ord("N"), 27):  # N o ESC
                    confirm_mode = None
                    print("\n[INFO] Operacion de borrado cancelada.")

            # ── Modo Normal de Captura y Navegación ──
            else:
                if key != 255:
                    key_char: str = chr(key).upper()

                    # 1. Presionar letra directa A-Z
                    if key_char in alphabet:
                        if key_char != active_label:
                            current_idx = alphabet.index(key_char)
                            active_label = key_char
                        is_recording = True

                    # 2. Mantener ESPACIO para grabar letra activa actual
                    elif key == 32:  # Barra espaciadora
                        is_recording = True

                    # 3. Navegación con Flechas (Izquierda = 81/2422400, Derecha = 83) o teclas A/D alternativas
                    elif key == 81 or key == 2 or key == ord(","):  # Flecha Izquierda
                        current_idx = (current_idx - 1) % len(alphabet)
                    elif key == 83 or key == 3 or key == ord("."):  # Flecha Derecha
                        current_idx = (current_idx + 1) % len(alphabet)

                    # 4. Solicitud de Borrado Total: Tecla DELETE (SUPR = 255 en ASCII basico, 80/46 en OpenCV)
                    elif key in (46, 127):  # Delete / Supr
                        confirm_mode = "ALL"

                    # 5. Solicitud de Borrado de Clase: BACKSPACE (8)
                    elif key == 8:  # Backspace
                        confirm_mode = "CLASS"

                    # 6. Salir con 'Q'
                    elif key in (ord("q"), ord("Q")):
                        break

                # Guardar muestra en ráfaga si está grabando y hay mano válida
                if is_recording and has_hand:
                    saved: bool = collector.append_sample(keypoints, active_label)
                    if saved:
                        curr_count: int = collector.counts[active_label]
                        print(
                            f"\r[GRABANDO] Letra '{active_label}': {curr_count}/{TARGET_SAMPLES_PER_CLASS} muestras",
                            end="",
                            flush=True,
                        )

            # Renderizar HUD completo
            hud_frame: NDArray[np.uint8] = draw_hud(
                annotated,
                active_label,
                collector,
                is_recording,
                has_hand,
                confirm_mode,
            )

            cv2.imshow("Traductor Alpha - Recolector de Datos (Fase 2)", hud_frame)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.release()
        print(f"\n\n[INFO] Sesión finalizada. Total muestras en CSV: {collector.total_samples}")


if __name__ == "__main__":
    main()
