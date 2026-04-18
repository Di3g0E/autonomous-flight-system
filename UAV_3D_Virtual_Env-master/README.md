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

Este proyecto requiere Python 3.13+. Se recomienda el uso de [uv](https://github.com/astral-sh/uv) para una gestión de dependencias mucho más rápida.

```powershell
# Situarse en la raiz del proyecto
cd UAV_3D_Virtual_Env-master
```

### 1. Configuración del Entorno (con `uv`)
Si es la primera vez que configuras el proyecto o quieres recrear el entorno `.venv`:

```powershell
# Crear el entorno virtual .venv (si no existe) e instalar dependencias
uv venv .venv --python 3.13
.venv\Scripts\activate
uv pip install -e .[all]
```

### 2. Método Tradicional (Venv + Pip) — CPU
Si no utilizas `uv`, puedes seguir el método estándar:

```powershell
# Habilitar rutas largas en Windows (ejecutar como ADMIN si da error de rutas largas)
# New-ItemProperty -Path "HKLM:\System\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

# Crear entorno virtual
python -m venv .venv

# Activar
.\.venv\Scripts\activate

# Instalación completa
pip install -e .[all]
```

### 3. Instalación con GPU (CUDA) — Recomendado para Entrenamiento

Para entrenamientos largos, es recomendable usar la GPU. Requiere una GPU NVIDIA con drivers CUDA instalados (verificar con `nvidia-smi`).

```powershell
# 1. Crear entorno virtual dedicado para GPU
python -m venv .vgpu

# 2. Activar
.vgpu\Scripts\activate

# 3. Instalar PyTorch con CUDA PRIMERO (antes que cualquier otra dependencia)
#    Adaptar cu128 a tu version de CUDA (ver "CUDA Version" en nvidia-smi)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 4. Instalar resto de dependencias
pip install -r requirements-gpu.txt

# 5. Instalar el proyecto en modo editable
pip install -e .[all]

# 6. Verificar que CUDA funciona
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> **Importante**: El paso 3 debe ejecutarse antes del 4 y 5 para evitar que pip sobreescriba PyTorch CUDA con la versión CPU.

Probado con: Python 3.13, NVIDIA RTX 3050 (4GB VRAM), Driver 572.16, CUDA 12.8.

### 4. Ejecutar el Programa Principal
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

# Entrenar un agente con Stable-Baselines3 (hover control)
python scripts/train_sb3.py

# Entrenar seguimiento visual con curriculum v2 (lemniscata)
python scripts/train_lemniscate_v2.py

# Evaluar modelo de seguimiento v2
python tests/test_lemniscate_v2.py

# Validar posiciones de spawn
python tests/test_spawn_positions.py

# Evaluar un agente ya entrenado
python scripts/evaluate_sb3.py

# Entrenar hover tracking v3.1 (fine-tune desde v3/900k)
python scripts/train_hover_track_v3_1.py --no-display

# Evaluar checkpoints v3.1
python tests/evaluate_hover_track_v3_1.py --no-display

# Test espiral + SAC v3.1 con target móvil
python tests/test_spiral_track_v3_1.py --no-display

# Entrenar hover tracking v4 (fine-tune desde v3.1/400k, target móvil)
python scripts/train_hover_track_v4.py \
    --base-checkpoint ./models/hover_track_v3_1/checkpoints/model_400000_steps.zip

# Evaluar checkpoints v4
python tests/evaluate_hover_track_v4.py --no-display
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

## Seguimiento Visual y Modo Filming

El entorno incluye un sistema de **seguimiento visual** donde el dron debe seguir una esfera magenta que traza una trayectoria en lemniscata (∞).

### Características Principales
- **Target magenta** con detección HSV (H: 140–170), eliminando falsos positivos en la escena urbana
- **Sistema de recompensa v2**: 6 componentes densos (survival, stability, centering, scale, discovery, not_visible) diseñados para que el seguimiento activo sea 11× más rentable que el hover pasivo
- **Filming mode**: descarta toda la recompensa del entorno base, dejando solo la señal visual como guía
- **Trayectoria lemniscata**: curva simétrica en ∞ con escala y velocidad configurables
- **Inicialización constrained**: estados iniciales acotados (near-hover) con domain randomization progresiva

### Sistema de Recompensa v2

| Componente | Rango | Descripción |
|---|---|---|
| `R_survival` | +0.05 | Incentivo constante por mantenerse en vuelo |
| `R_stability` | 0 → +1.0 | Gaussianas multiplicadas sobre velocidad angular y tilt |
| `R_centering` | 0 → +3.0 | Centrado del target en la imagen FPV |
| `R_scale` | 0 → +2.0 | Gaussiana asimétrica sobre fracción de imagen (ideal 25%) |
| `R_discovery` | +3.0 | Bonus repetible cada vez que el target reaparece |
| `R_not_visible` | -0.5/step | Penalización pasiva sin target visible |

### Parámetros Clave
| Parámetro | Default | Descripción |
|---|---|---|
| `use_new_reward` | False | Activa el sistema de recompensa v2 |
| `initial_target_distance` | 2.0 | Distancia fija dron-target en Fase A |
| `constrained_init` | False | Inicialización near-hover acotada |
| `init_pos_range` | 0.5 | Rango de posición inicial (m) |
| `init_vel_range` | 0.25 | Rango de velocidad inicial (m/s) |
| `init_ang_range` | 0.1 | Rango de ángulos iniciales (rad) |
| `lemniscate_scale` | 5.0 | Semianchura de la trayectoria en ∞ |

### Uso
```python
from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv

# Modo v2 con recompensa multicomponente
env = Panda3DQuadrotorEnv(
    use_camera=True,
    use_target=True,
    target_mode='moving',
    filming_mode=True,
    use_new_reward=True,
    initial_target_distance=2.0,
    constrained_init=True,
    lemniscate_scale=5.0,
)
```

### Entrenamiento con Curriculum Adaptativo
```bash
# Entrenar con curriculum de 3 fases (hover → lemniscata lenta → rápida)
python scripts/train_lemniscate_v2.py
```

El script `train_lemniscate_v2.py` implementa:
- **Fase A** (0–30%): Target fijo a 2m, aprender hover + centrado
- **Fase B** (30–70%): Lemniscata lenta (0.02→0.16 m/s)
- **Fase C** (70–100%): Lemniscata a velocidad completa (0.16→0.30 m/s)
- Transfer learning desde `models/goal_controller/best_model.zip`
- Domain randomization basada en rendimiento (no temporal)

---

## Hover Tracking (v3) — Cámara Vertical + SAC

Sistema de seguimiento donde el dron se posiciona sobre la esfera y la mantiene centrada en una cámara que apunta hacia abajo, a la distancia calibrada de **1.394m**.

### Características Principales
- **Cámara vertical** (`pitch=-90°`): observa la esfera desde arriba
- **Observación centroide** (19-D flat): estado(13) + centroid_x, centroid_y, fraction, visible, delta_cx, delta_cy
- **Sin CNN**: features extraídas por detección HSV (buffer ~21 MB vs ~1.2 GB)
- **SAC** con entropía auto-ajustada (off-policy, sample-efficient)
- **Espiral de búsqueda**: modelo RL pre-entrenado que se activa cuando la esfera se pierde durante 0.2s

### Sistema de Recompensa v3 (3 componentes, rango [-1, +4])

| Componente | Rango | Descripción |
|---|---|---|
| `R_stability` | 0 → +1.0 | Vuelo estable (baja velocidad angular, bajo tilt) |
| `R_centering` | 0 → +2.0 | Esfera centrada en la imagen |
| `R_scale` | 0 → +1.0 | Fracción de imagen cercana al 25% (gaussiana asimétrica) |
| `R_invisible` | -1.0 | Penalización cuando la esfera no es visible |

### Uso
```bash
# Entrenar hover tracking (SAC, ~7h para 200k steps)
python scripts/train_hover_track.py --timesteps 200000

# Evaluar con espiral de búsqueda
python tests/test_hover_track.py --model-path ./models/hover_track/best_model.zip

# Evaluar con target en movimiento
python tests/test_hover_track.py --target-mode moving --target-speed 0.1

# Entrenar modelo de espiral (prerequisito)
python scripts/train_spiral_follow.py --timesteps 500000
```

### Resultados (200k steps)
| Métrica | Inicio | Final |
|---|---|---|
| Reward | 152 | 1,574 (+10.3×) |
| Visibilidad | 62% | 91% |
| Centering | 0.43 | 0.27 (-37%) |
| Estabilidad | 0.59 | 0.99 |
| Episodios completos | ~20% | 100% |

### Hover Track v2 — Entrenamiento Robusto con Curriculum

Reentrenamiento del modelo SAC con target descentrado y condiciones post-espiral para mejorar la transición búsqueda→estabilización:

```bash
# Entrenar v2 (500k steps, ~60h, guarda checkpoints cada 50k)
python scripts/train_hover_track_v2.py --timesteps 500000 --no-display
```

Tres fases de curriculum progresivo: target offset ±0.1→1.0m, velocidad lateral ±0.10→0.35 m/s, tilt ±0.05→0.15 rad. El modelo original v1 no se modifica.

### Hover Track v3 — Fase 0 de Estabilización + Centering Apretado

Curriculum de 4 fases con Phase 0 de estabilización pura (sin target) para aprender control motor básico antes del tracking visual. Gaussiana de centering más estrecha (`4.0×exp(-6d²)`) y episodios de 30s.

```bash
# Entrenar v3 (1.5M steps, ~15h con GPU)
python scripts/train_hover_track_v3.py --timesteps 1500000 --no-display

# Evaluar checkpoints v3
python tests/evaluate_hover_track_v3.py --no-display
```

### Hover Track v3.1 — Reward Multiplicativa (Estabilidad como Gate)

Fine-tune del modelo v3 (checkpoint 900k) que corrige la inestabilidad observada en los tiers medium y hard. El problema era que la reward aditiva v3 permitía obtener reward alto de centering (max 4.0) sin necesidad de ser estable (max 1.0).

**Solución**: la reward de tracking ahora se multiplica por la estabilidad:

```
total = R_stability × (R_centering + R_center_vel + R_scale + 0.5) + R_vel_damp + R_smooth + R_invisible
```

Nuevos componentes:
- **R_smooth**: `-0.3 × ||Δaction||²` — penaliza cambios bruscos de motores
- **R_vel_damp**: `0.5 × exp(-4v²)` — recompensa velocidad lineal baja durante tracking

```bash
# Fine-tune v3.1 desde checkpoint 900k (500k steps, ~7.5h con GPU)
python scripts/train_hover_track_v3_1.py --no-display

# Evaluar checkpoints v3.1
python tests/evaluate_hover_track_v3_1.py --no-display

# Test integración espiral+SAC con target móvil (lemniscata)
python tests/test_spiral_track_v3_1.py --no-display
```

**Resultados de evaluación (checkpoint 400k — mejor modelo)**:
| Tier | Supervivencia | Jerk | Estabilidad |
|---|---|---|---|
| Easy (off=0.3m) | 100% | 0.134 | 0.9965 |
| Medium (off=0.6m) | 100% | 0.138 | 0.9879 |
| Hard (off=1.0m) | 80% | 0.097 | 0.9864 |

### Hover Track v4 — Fine-Tune con Objetivo Móvil

Fine-tune del modelo v3.1 (checkpoint 400k) con target en movimiento (lemniscata). Primer entrenamiento del proyecto con objetivo dinámico para el controlador SAC. El pipeline se simplifica eliminando los estados BRAKE y HANDOFF — el curriculum Phase C cubre esas condiciones implícitamente.

**Pipeline simplificado**:
```
SEARCH (PPO espiral) ←→ TRACK (SAC v4)
  k=20 steps sin target         target visible
```

**Curriculum de 3 fases**:
| Fase | Progreso | Velocidad target | Offset | Vel. init |
|---|---|---|---|---|
| A | 0%–30% | 0.05→0.15 m/s | 0.2→0.4m | 0.10→0.15 m/s |
| B | 30%–65% | 0.15→0.25 m/s | 0.4→0.7m | 0.15→0.30 m/s |
| C | 65%–100% | 0.25→0.40 m/s | 0.7→1.2m | 0.30→0.60 m/s |

```bash
# Fine-tune v4 (750k steps) usando el mejor checkpoint v3.1
python scripts/train_hover_track_v4.py \
    --base-checkpoint ./models/hover_track_v3_1/checkpoints/model_400000_steps.zip

# Evaluar checkpoints v4 con tiers de velocidad
python tests/evaluate_hover_track_v4.py --no-display
```

---

## Pipeline Espiral-a-Hover (Búsqueda y Estabilización)

Pipeline completo de 5 estados para buscar un target perdido con espiral y estabilizarse sobre él:

```
STABILIZE (PD) → SEARCH (PPO espiral) → BRAKE (PD) → HANDOFF (PD→SAC) → TRACK (SAC)
```

```bash
# Grabar vídeo del pipeline completo (4 vistas + quad_view)
python tests/test_spiral_to_hover_video.py --duration 20 --offset 1.5
```

El controlador garantiza que el SAC nunca actúa con el target invisible y que el dron frena antes de la transición.

---

## Modelos Entrenados

| Modelo | Ruta | Algoritmo | Propósito |
|---|---|---|---|
| Depth U-Net | `models/depth_final/best_model.pth` | Supervisado | Estimar profundidad monocular desde imagen RGB 32×32 |
| Goal Controller | `models/goal_controller/best_model.zip` | PPO | Controlador base: alcanzar posición objetivo en 3D |
| Spiral Follow | `models/spiral_follow/best_model.zip` | PPO | Seguir espiral de Arquímedes para búsqueda de target perdido |
| Hover Track v1 | `models/hover_track/best_model.zip` | SAC | Hover estático sobre target (condiciones ideales, target centrado) |
| Hover Track v2 | `models/hover_track_v2/best_model.zip` | SAC | Hover robusto: target descentrado, recuperación post-espiral |
| Hover Track v3 | `models/hover_track_v3/best_model.zip` | SAC | Phase 0 estabilización + centering apretado (4 fases) |
| Hover Track v3.1 | `models/hover_track_v3_1/checkpoints/model_400000_steps.zip` | SAC | Fine-tune de v3: reward multiplicativa. **Mejor checkpoint: 400k** (93.3% surv, jerk=0.123) |
| Hover Track v4 | `models/hover_track_v4/best_model.zip` | SAC | Fine-tune de v3.1 (400k) con objetivo móvil. 750k steps, ~22h. **Resultado: catastrophic forgetting en Phase C** — pendiente v4.1 |
| Lemniscate v2 | `models/lemniscate_v2/interrupted_model.zip` | PPO | Seguir trayectoria en ∞ (interrumpido, parcial) |

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
- **REAL_CTRL = False**: Usa la simulación de sensores (acelerómetro, giróscopo, GPS).
- **HOVER = True**: El dron despega y se mantiene estable en el sitio.
- **HOVER = False**: El dron comienza en un estado inicial aleatorio.

Para el modo de seguimiento visual, usa `filming_mode=True` en `Panda3DQuadrotorEnv` — el dron aprenderá a seguir la esfera magenta basándose exclusivamente en la imagen de la cámara FPV.
