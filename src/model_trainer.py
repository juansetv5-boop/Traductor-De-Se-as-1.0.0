"""
model_trainer.py — Módulo de entrenamiento del clasificador de señas (Random Forest).

Lee el dataset de keypoints normalizados desde `data/keypoints.csv`, valida las clases
disponibles, entrena un modelo `RandomForestClassifier` (scikit-learn), genera métricas,
grafica y guarda la matriz de confusión, y exporta el binario `.pkl` junto con `metrics.json`.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# Importar constantes globales
sys.path.insert(0, ".")
from config import (
    DATA_CSV_PATH,
    MODEL_PKL_PATH,
    NUM_FEATURES,
    STATIC_ALPHABET,
)


class ModelTrainer:
    """
    Entrenador del clasificador de dactilología estática mediante RandomForest.
    """

    def __init__(
        self,
        csv_path: str = DATA_CSV_PATH,
        model_path: str = MODEL_PKL_PATH,
    ) -> None:
        """
        Inicializa el entrenador con las rutas de dataset y destino del modelo.

        Args:
            csv_path: Ruta al archivo CSV con las muestras.
            model_path: Ruta donde se guardará el modelo entrenado (.pkl).
        """
        self.csv_path: str = csv_path
        self.model_path: str = model_path
        self.models_dir: str = os.path.dirname(self.model_path)
        os.makedirs(self.models_dir, exist_ok=True)

    def load_dataset(self) -> Tuple[NDArray[np.float64], NDArray[np.str_]]:
        """
        Carga las 63 características (X) y las etiquetas (y) desde el CSV.

        Returns:
            Tupla (X, y) donde X es matriz (N, 63) y y es vector (N,).

        Raises:
            FileNotFoundError: Si no existe el CSV o no contiene muestras.
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"[ERROR] No se encontro el archivo de dataset en '{self.csv_path}'. "
                f"Ejecute primero 'python -m src.data_collector' para recolectar datos."
            )

        X_list: List[List[float]] = []
        y_list: List[str] = []

        with open(self.csv_path, mode="r", encoding="utf-8") as f:
            lines = f.readlines()

        if len(lines) <= 1:
            raise ValueError(f"[ERROR] El archivo '{self.csv_path}' esta vacio (sin muestras).")

        # Omitir cabecera (primera línea)
        for line in lines[1:]:
            parts: List[str] = line.strip().split(",")
            if len(parts) == NUM_FEATURES + 1:
                features: List[float] = [float(val) for val in parts[:NUM_FEATURES]]
                label: str = parts[-1].strip().upper()
                X_list.append(features)
                y_list.append(label)

        X: NDArray[np.float64] = np.array(X_list, dtype=np.float64)
        y: NDArray[np.str_] = np.array(y_list, dtype=str)

        return X, y

    def validate_alphabet_coverage(self, y: NDArray[np.str_]) -> List[str]:
        """
        Verifica si el dataset cubre todo el abecedario estático configurado.

        Si faltan letras, muestra una alerta ANSI destacada en consola.

        Args:
            y: Vector de etiquetas presentes en el dataset.

        Returns:
            Lista de clases (letras) únicas presentes en el dataset.
        """
        present_classes: List[str] = sorted(list(set(y.tolist())))
        missing_classes: List[str] = [char for char in STATIC_ALPHABET if char not in present_classes]

        print("\n" + "=" * 70)
        print(" DIAGNÓSTICO DEL DATASET")
        print("=" * 70)
        print(f" Total de muestras leídas: {len(y)}")
        print(f" Clases detectadas ({len(present_classes)}/{len(STATIC_ALPHABET)}): {', '.join(present_classes)}")

        if missing_classes:
            # Resaltado en consola con códigos ANSI (Amarillo/Rojo)
            color_yellow = "\033[93m"
            color_red = "\033[91m"
            color_reset = "\033[0m"

            print(f"\n{color_yellow}⚠️  ADVERTENCIA DE COBERTURA INCOMPLETA ⚠️{color_reset}")
            print(
                f"{color_red}El dataset NO contiene muestras para las siguientes {len(missing_classes)} letras:{color_reset}"
            )
            print(f" {color_red}Faltan: {', '.join(missing_classes)}{color_reset}")
            print(
                f"{color_yellow}→ El entrenamiento continuara solo con las {len(present_classes)} clases disponibles para prototipado.{color_reset}\n"
            )
        else:
            print("\033[92m[OK] Cobertura completa del alfabeto estático (24 letras presenciales).\033[0m\n")

        return present_classes

    def train_and_evaluate(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.str_],
    ) -> Tuple[RandomForestClassifier, float, Dict[str, object]]:
        """
        Realiza el split train/test, entrena el RandomForest y calcula las métricas.

        Args:
            X: Matriz de características (N, 63).
            y: Vector de etiquetas (N,).

        Returns:
            Tupla (modelo_entrenado, accuracy_global, reporte_dict).
        """
        # Comprobar si hay muestras suficientes para stratify
        class_counts = {cls: np.sum(y == cls) for cls in set(y)}
        can_stratify = all(count >= 2 for count in class_counts.values())

        stratify_param = y if can_stratify else None
        if not can_stratify:
            print("[AVISO] Algunas clases tienen menos de 2 muestras. Se realiza split sin estratificación.")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=stratify_param,
        )

        print(f" Muestras de entrenamiento: {len(X_train)}")
        print(f" Muestras de prueba:        {len(X_test)}")

        # Entrenar RandomForest Classifier
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=42,
            n_jobs=-1,  # Usar todos los núcleos CPU disponibles
        )
        clf.fit(X_train, y_train)

        # Evaluar
        y_pred = clf.predict(X_test)
        accuracy: float = float(clf.score(X_test, y_test))

        report_dict: Dict[str, object] = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        )
        report_text: str = classification_report(
            y_test,
            y_pred,
            zero_division=0,
        )

        print("\n" + "=" * 70)
        print(" RESULTADOS DE EVALUACIÓN ML (RandomForest)")
        print("=" * 70)
        print(f" Accuracy Global (Test 20%): {accuracy * 100:.2f}%\n")
        print(" Reporte de Clasificación Detallado:")
        print(report_text)

        # Generar y guardar Matriz de Confusión
        labels = sorted(list(set(y.tolist())))
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        self._plot_and_save_confusion_matrix(cm, labels)

        return clf, accuracy, report_dict

    def _plot_and_save_confusion_matrix(self, cm: NDArray[np.int64], labels: List[str]) -> None:
        """
        Renderiza la matriz de confusión con matplotlib, la guarda como PNG y la muestra.

        Args:
            cm: Matriz de confusión calculada.
            labels: Lista de nombres de clases.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap="Blues", ax=ax, xticks_rotation="horizontal", values_format="d")

        plt.title("Matriz de Confusión — Clasificador de Señas (RandomForest)", fontsize=13, fontweight="bold")
        plt.tight_layout()

        # Guardar imagen en carpeta models/
        cm_path: str = os.path.join(self.models_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=300)
        print(f"[INFO] Matriz de confusión guardada en '{cm_path}'.")

        # Mostrar ventana gráfica emergente
        plt.show(block=False)
        plt.pause(2)  # Mostrar durante 2 segundos sin bloquear completamente la ejecución
        plt.close(fig)

    def save_artifacts(
        self,
        clf: RandomForestClassifier,
        accuracy: float,
        present_classes: List[str],
        total_samples: int,
        report_dict: Dict[str, object],
    ) -> None:
        """
        Exporta el binario .pkl del modelo y el informe estructurado metrics.json.

        Args:
            clf: Modelo de scikit-learn entrenado.
            accuracy: Precisión global del modelo.
            present_classes: Lista de clases incluidas en el entrenamiento.
            total_samples: Número de filas procesadas.
            report_dict: Diccionario del classification_report.
        """
        # 1. Exportar modelo .pkl
        joblib.dump(clf, self.model_path)
        print(f"\n[ÉXITO] Modelo exportado correctamente a '{self.model_path}'.")

        # 2. Exportar métricas a metrics.json
        metrics_path: str = os.path.join(self.models_dir, "metrics.json")
        metrics_data = {
            "fecha_entrenamiento": datetime.now().isoformat(),
            "accuracy_global": round(accuracy, 4),
            "accuracy_porcentaje": f"{accuracy * 100:.2f}%",
            "muestras_totales": total_samples,
            "clases_entrenadas": present_classes,
            "num_clases": len(present_classes),
            "reporte_detallado": report_dict,
        }

        with open(metrics_path, mode="w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=4, ensure_ascii=False)

        print(f"[ÉXITO] Archivo de métricas guardado en '{metrics_path}'.")


def main() -> None:
    """Función principal de ejecución del módulo de entrenamiento."""
    print("=" * 70)
    print(" ENTRENAMIENTO DEL MODELO — Traductor Alpha (Fase 3)")
    print(" Clasificador: RandomForestClassifier (100 árboles)")
    print("=" * 70)

    trainer = ModelTrainer()

    try:
        # 1. Cargar datos
        X, y = trainer.load_dataset()

        # 2. Validar abecedario
        present_classes = trainer.validate_alphabet_coverage(y)

        # 3. Entrenar y evaluar
        clf, accuracy, report_dict = trainer.train_and_evaluate(X, y)

        # 4. Guardar binario y json de métricas
        trainer.save_artifacts(
            clf=clf,
            accuracy=accuracy,
            present_classes=present_classes,
            total_samples=len(y),
            report_dict=report_dict,
        )

        print("\n" + "=" * 70)
        print(" FASE 3 COMPLETADA CON ÉXITO — Listo para inferencia (Fase 4)")
        print("=" * 70)

    except Exception as e:
        print(f"\n[ERROR CRÍTICO] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
