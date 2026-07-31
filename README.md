# Traductor Alpha de Lenguaje de Señas (Dactilología Estática) a Texto y Voz (TTS)

Sistema modular y de alto rendimiento en Python para la traducción en tiempo real del alfabeto dactilológico de la Lengua de Señas (A-Z) a texto y voz sintetizada offline.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Realtime-green.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-RandomForest-blueviolet.svg)

---

## 🌟 Características Principales

- **Sin CNNs Pesadas (Alto Rendimiento):** Procesa la mano mediante **MediaPipe Hands** (21 puntos clave 3D) convirtiendo cada fotograma en un vector vectorial de **63 características** ($21 \text{ puntos} \times [X, Y, Z]$). Ejecución a $\ge 30\text{ FPS}$ en CPU estándar sin requerir GPU.
- **Normalización Matemáticamente Robusta:**
  - **Centrado (Traducción):** Resta las coordenadas 3D de la muñeca (`Landmark 0`) a todos los puntos.
  - **Escalado (Invariancia a Distancia):** Divide entre la magnitud absoluta máxima del vector centrado, logrando invarianza total ante variaciones de distancia o escala de la mano frente a la cámara.
- **Confirmación por Debounce:** Buffer de estabilidad de 15 fotogramas consecutivos (~0.5s) que evita parpadeos o falsos positivos entre gestos físicamente similares (ej. M, N, S).
- **Lector de Voz (TTS) Asíncrono Non-Blocking:** Síntesis de voz con `pyttsx3` en un hilo secundario (`threading.Thread`) mediante colas de tareas (`queue.Queue`), garantizando $0\%$ de caídas en los FPS de OpenCV.
- **Temporizador de Inactividad (3 Segundos):** Al retirar la mano o mantener la misma seña inmóvil por 3 segundos, el sistema pronuncia automáticamente el texto acumulado y limpia la pantalla.

---

## 📁 Estructura del Proyecto

```
Vision-Artificial-1.0.0/
├── config.py                  # Constantes globales del proyecto
├── main.py                    # Menú interactivo / Lector principal
├── requirements.txt           # Dependencias pip
├── README.md                  # Documentación principal
├── CONTEXT.md                 # Contexto de arquitectura y decisiones de diseño
├── .gitignore                 # Exclusiones de Git
│
├── src/
│   ├── __init__.py            # Inicializador de paquete
│   ├── keypoint_extractor.py  # Extracción y normalización de keypoints 3D
│   ├── data_collector.py      # Recolección de dataset en ráfaga a CSV
│   ├── model_trainer.py       # Entrenamiento del RandomForest y métricas
│   └── inference.py           # Inferencia en vivo + Debounce + TTS 3s
│
├── data/
│   └── keypoints.csv          # Dataset generado (63 features + label)
│
└── models/
    ├── sign_classifier.pkl    # Modelo entrenado (.pkl)
    ├── confusion_matrix.png   # Gráfica de la matriz de confusión
    └── metrics.json           # Informe de precisión y cobertura
```

---

## 🚀 Instalación

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/juansetv5-boop/Traductor-De-Se-as-1.0.0.git
   cd Traductor-De-Se-as-1.0.0
   ```

2. **Crear y activar un entorno virtual (Recomendado):**
   ```bash
   python -m venv .venv
   # En Windows:
   .venv\Scripts\activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Guía de Uso Paso a Paso

Puedes ejecutar todo el sistema a través del menú interactivo centralizado:

```bash
python main.py
```

O ejecutar cada módulo individualmente:

### Paso 1: Verificación de Captura y Normalización
Valida que la cámara y MediaPipe estén extrayendo correctamente los 63 valores en rango $[-1, 1]$:
```bash
python -m src.keypoint_extractor
```

### Paso 2: Recolección de Datos a CSV
Manten presionada la tecla de cualquier letra (A-Z) o `ESPACIO` para grabar muestras en ráfaga a 30 FPS hasta alcanzar la meta sugerida de 350 muestras por clase:
```bash
python -m src.data_collector
```
- `< >` o `, .`: Cambiar de letra activa.
- `SUPR` / `DELETE`: Borrar todo el dataset (con doble confirmación `Y/N`).
- `BACKSPACE`: Borrar muestras de la letra activa (con doble confirmación `Y/N`).

### Paso 3: Entrenamiento del Modelo ML
Entrena el clasificador `RandomForestClassifier` (100 árboles), valida la cobertura del alfabeto, genera el informe de precisión y grafica la matriz de confusión:
```bash
python -m src.model_trainer
```

### Paso 4: Inferencia y Traducción en Tiempo Real con Voz
Inicia la ventana de lectura en tiempo real con superposición de HUD, debounce y sintetizador de voz automático:
```bash
python -m src.inference
```
- `ESPACIO`: Insertar espacio entre palabras.
- `BACKSPACE` / `C`: Borrar último carácter.
- `Q` / `ESC` / `}`: Salir de la aplicación.

---

## 📐 Fundamento Matemático de Normalización

Dado un conjunto de 21 coordenadas 3D detectadas por MediaPipe $\mathbf{P} = \{\mathbf{p}_0, \mathbf{p}_1, \dots, \mathbf{p}_{20}\}$ donde cada $\mathbf{p}_i = (x_i, y_i, z_i)$:

1. **Traslación (Centrado en la muñeca):**
   $$\mathbf{p}'_i = \mathbf{p}_i - \mathbf{p}_0 \quad \forall i \in \{0, \dots, 20\}$$
   donde $\mathbf{p}'_0 = (0, 0, 0)$.

2. **Escalado (Invariancia de escala y distancia):**
   $$S = \max_{i, c} |\mathbf{p}'_{i, c}| \quad \text{donde } c \in \{x, y, z\}$$
   $$\mathbf{\hat{p}}_i = \frac{\mathbf{p}'_i}{S}$$

3. **Vectorización:**
   $$\mathbf{V} = [\mathbf{\hat{p}}_0, \mathbf{\hat{p}}_1, \dots, \mathbf{\hat{p}}_{20}] \in \mathbb{R}^{63}$$

---

## 📜 Licencia

Desarrollado bajo licencia MIT. Libre para uso educativo e investigación.
