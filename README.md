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

### Modo Dinámico: Gestos Continuos LSC en GPU (TCN + Apex Trigger)

Para señas dinámicas completas que involucran trayectoria temporal, manos y cuerpo (ej. *Hola, Gracias, Por favor, Ayuda, Buenos días*):

#### Paso 5: Recolección de Secuencias de Video Continuas
Graba clips de video respetando el protocolo anti-data leakage (organización por sujetos):
```bash
python src/data_collection/video_collector.py
```

#### Paso 6: Generación de Dataset Temporal y Cinemático
Extrae ventanas temporales de $T=40$ frames y calcula derivadas cinemáticas ($dX, d^2X$):
```bash
# Con dataset real de videos:
python src/data_collection/dataset_generator.py

# O generar dataset sintético de prueba inmediata:
python src/data_collection/dataset_generator.py --synthetic
```

#### Paso 7: Entrenamiento de la Red TCN (PyTorch / GPU)
Entrena la red convolucional temporal causal con Focal Loss y exportación a TorchScript:
```bash
python src/training/train_dynamic.py --epochs 25
```

#### Paso 8: Traducción Continua en Vivo con Apex TTS
Inicia el motor de inferencia en tiempo real con detección del ápice cinemático y síntesis de voz sin retraso perceptual:
```bash
python src/live_inference/live_engine.py
```

#### Paso 9: Perfilado de Latencia Sub-5ms
Mide la latencia de inferencia en GPU (CUDA) y CPU:
```bash
python tests/benchmark_latency.py
```

---

## 📐 Fundamento Matemático de Cinemática Temporal

Para un gesto dinámico continuo, se analiza una ventana de tiempo $T=40$ fotogramas a 30 FPS (~1.33 s):

1. **Vector Espacial Normalizado por Frame:**
   - Pose superior (7 puntos: hombros, codos, muñecas, nariz) centrada respecto al punto medio entre hombros y escalada por el ancho bi-acromial.
   - Manos (21 puntos cada una) normalizadas en modo wrist-relative y escaladas por la dimensión de la palma.
   - Vector espacial resultante: $\mathbf{P}_t \in \mathbb{R}^{147}$.

2. **Diferenciales Temporales (Velocidad y Aceleración):**
   - Velocidad vectorial: $\mathbf{V}_t = \mathbf{P}_t - \mathbf{P}_{t-1} \in \mathbb{R}^{147}$
   - Aceleración vectorial: $\mathbf{A}_t = \mathbf{V}_t - \mathbf{V}_{t-1} \in \mathbb{R}^{147}$
   - Tensor de características por frame: $\mathbf{F}_t = [\mathbf{P}_t, \mathbf{V}_t, \mathbf{A}_t] \in \mathbb{R}^{441}$.
   - Entrada para Convolución Temporal 1D: $\mathbf{X} \in \mathbb{R}^{\text{Batch} \times 441 \times 40}$.

3. **Detección del Apex (Disparador TTS sin retardo ciego):**
   - La velocidad de la muñeca $v(t) = \|\mathbf{p}_{\text{wrist}}(t) - \mathbf{p}_{\text{wrist}}(t-1)\|$ se monitorea en tiempo real.
   - El evento de traducción se dispara exactamente en el mínimo local de velocidad post-pico ($v(t) \le \alpha \cdot v_{\max}$ con $\alpha=0.40$), sincronizando la síntesis de voz en el momento natural en que el usuario finaliza la seña.

---

## 📜 Licencia

Desarrollado bajo licencia MIT. Libre para uso educativo e investigación.

