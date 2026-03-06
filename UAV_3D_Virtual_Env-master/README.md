# Sistema de Vuelo Autónomo (UAV 3D Virtual Env)

Este proyecto está basado en el repositorio [UAV_3d_virtual_env](https://github.com/rafaelcostafrf/UAV_3d_virtual_env). A partir de esta base, he realizado todas las modificaciones, mejoras e implementaciones posteriores de capacidades de vuelo autónomo mediante IA.

### Sobre el Repositorio Base
El repositorio original proporcionaba un entorno de simulación visual en 3D para drones (UAVs), diseñado específicamente para abordar problemas comunes en aprendizaje automático (machine learning) y visión artificial. Al utilizar el motor de juegos **Panda3D**, ofrecía una alternativa de alto rendimiento frente al desarrollo de simulaciones desde cero en Python.

El sistema base incluía:
- **Dinámica completa de un cuadricóptero funcional**.
- **Cámaras integradas (on-board y off-board)** con integración directa con OpenCV.
- **Escenarios y modelos de UAV totalmente personalizables**, permitiendo simular diversas misiones.
- **Detección de colisiones** y fundamentos de interacción con el entorno.

Esta base sólida permitió probar algoritmos en un entorno simulado antes de su implementación en plataformas reales, reduciendo significativamente los costes y riesgos de desarrollo.



# Cómo Empezar (Instalación y Ejecución)

Este proyecto requiere Python 3.9+ (se recomienda 3.10 o superior).

### 1. Activar el Entorno Virtual
El proyecto ya incluye un entorno virtual en la carpeta `.v`. Para activarlo:

```powershell
# En Windows (PowerShell)
.\.v\Scripts\activate
```

### 2. Instalar el Proyecto (Si es necesario)
Si es la primera vez que configuras el proyecto o has borrado el entorno, instala todas las dependencias:

```powershell
# Habilitar rutas largas en Windows (ejecutar como ADMIN si da error de rutas largas)
# New-ItemProperty -Path "HKLM:\System\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

# Instalación completa
pip install -e .[all]
```

### 3. Ejecutar el Programa Principal
Una vez activado el entorno, lanza la simulación 3D:

```powershell
python scripts/run_simulation.py
```

---

### Estructura del Proyecto
El proyecto está organizado siguiendo los estándares modernos de Python e incorpora diversas utilidades para ML y RL:

- **`assets/`**: Recursos estáticos (modelos 3D y texturas).
- **`config/`**: Archivos de configuración y calibración de cámaras.
- **`data/`**: Datasets generados y recolectados (ej. `depth_dataset`).
- **`docs/`**: Documentación adicional y recursos del proyecto.
- **`examples/`**: Scripts de ejemplo y pruebas de concepto simples.
- **`experiments/`**: Resultados de experimentos, registros detallados y comparativas.
- **`models/`**: Modelos entrenados (ej. modelos de estimación de profundidad y controladores RL).
- **`scripts/`**: Scripts principales de ejecución para simulación, entrenamiento, evaluación y recolección de datos.
- **`src/`**: Carpeta de código fuente principal estructurada en módulos:
  - **`agents/`**: Implementaciones de agentes y utilidades de configuración de RL.
  - **`dataset/`**: Clases y utilidades para la carga y manejo de datos.
  - **`envs/`**: Entornos Gymnasium (Quadrotor y Panda3D).
  - **`models/`**: Definición de arquitecturas de redes neuronales (ej. U-Net de profundidad).
  - **`simulation/`**: Control de motores físicos, setup del mundo 3D y cámaras.
  - **`utils/`**: Utilitarios generales para el flujo de trabajo.
  - **`vision/`**: Módulos de procesamiento de visión artificial.
- **`tests/`**: Suite de pruebas (pytest) para verificar la integridad del sistema.
- **`weights/`**: Pesos de respaldo o archivos pre-entrenados adicionales.

---

# Ejecutar Pruebas y Scripts Adicionales
Para verificar la integridad o realizar entrenamientos:

```bash
# Ejecutar los tests
pytest tests/

# Entrenar un agente con Stable-Baselines3
python scripts/train_sb3.py

# Evaluar un agente ya entrenado
python scripts/evaluate_sb3.py
```

---

## Monocular Depth Estimation

### Características
- **Depth Buffer Extraction**: Extrae ground truth depth del simulador Panda3D
- **Dataset Collection**: Generación automatizada de datasets RGB-Depth con almacenamiento HDF5
- **Lightweight U-Net**: Modelo de ~1.2M parámetros optimizado para PCs personales
- **Métricas Estándar**: RMSE, AbsRel, δ1/δ2/δ3 para evaluación
- **Pipeline Completo**: Colección → Entrenamiento → Evaluación

### Uso Rápido

#### 1. Colectar Dataset de Profundidad
```bash
python scripts/collect_depth_dataset.py \
    --num-samples 5000 \
    --output-dir ./data/depth_dataset \
    --camera high_freq
```

#### 2. Entrenar Modelo de Profundidad
```bash
python scripts/train_depth_model.py \
    --dataset-dir ./data/depth_dataset \
    --output-dir ./models/depth_v1 \
    --epochs 50 \
    --batch-size 16
```

#### 3. Evaluar Modelo
```bash
python scripts/evaluate_depth_model.py \
    --model-path ./models/depth_v1/best_model.pth \
    --dataset-dir ./data/depth_dataset \
    --split test
```

#### 4. Usar en Código
```python
from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv

env = Panda3DQuadrotorEnv(use_camera=True, use_depth=True)
obs, info = env.reset()
# obs['depth_high_freq']: (64, 64, 1) float32
```

Ver `PROJECT_HISTORY.md` para documentación detallada.

---

# Guía de Uso del Simulador

### Controles de Cámara (Ventana 3D)
1. **C**: Cambia entre las diferentes cámaras (on-board/externas).
2. **WASD**: Cambia el ángulo de la cámara externa.
3. **QE**: Cambia la distancia (zoom) de la cámara externa.
4. **R**: Resetea la posición de la cámara.

### Calibración de Cámaras
El software detecta automáticamente si faltan los archivos de calibración en `./config/camera_calibration_*.npz` y, si es necesario, ejecutará el algoritmo de calibración tomando capturas aleatorias del patrón de tablero de ajedrez en el mundo 3D.

### Nota sobre el Controlador
El quadrotor se controla mediante una red neuronal entrenada con PPO. Puedes configurar el comportamiento en `scripts/run_simulation.py`:
- **REAL_CTRL = True**: Usa los estados reales del simulador.
- **REAL_CTRL = False**: Usa la simulación de sensores (acelerómetro, giroscopio, GPS).
- **HOVER = True**: El dron despega y se mantiene estable en el sitio.
- **HOVER = False**: El dron comienza en un estado inicial aleatorio.
