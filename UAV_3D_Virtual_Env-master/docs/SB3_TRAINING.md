# Entrenamiento con Stable-Baselines3

## Descripción General

Este directorio contiene scripts para entrenar y evaluar agentes de Reinforcement Learning usando **Stable-Baselines3** (SB3), reemplazando el controlador manual por un enfoque estándar de RL.

## Objetivo

Verificar que el dron puede aprender a mantenerse estable (hover control) usando algoritmos modernos de RL como PPO y SAC.

## Requisitos

```bash
pip install stable-baselines3[extra]
pip install tensorboard
```

## Scripts Disponibles

### 1. `quick_train_sb3.py` - Verificación Rápida

**Propósito**: Entrenamiento rápido para verificar que el sistema funciona.

**Uso**:
```bash
python quick_train_sb3.py
```

**Características**:
- Entrenamiento corto (50,000 pasos)
- Comparación con política aleatoria (baseline)
- Evaluación automática
- Guarda modelo en `./models/quick_test/`

**Tiempo estimado**: 5-10 minutos

### 2. `train_sb3.py` - Entrenamiento Completo

**Propósito**: Entrenamiento completo con configuración profesional.

**Uso básico**:
```bash
# Entrenar con PPO (recomendado)
python train_sb3.py --algorithm ppo --timesteps 500000

# Entrenar con SAC (alternativa)
python train_sb3.py --algorithm sac --timesteps 500000
```

**Opciones avanzadas**:
```bash
python train_sb3.py \
    --algorithm ppo \
    --timesteps 1000000 \
    --n-envs 8 \
    --learning-rate 3e-4 \
    --save-dir ./models/my_experiment \
    --log-dir ./logs/my_experiment
```

**Características**:
- Entrenamiento paralelo (múltiples entornos)
- Callbacks para evaluación y checkpoints
- Logging con TensorBoard
- Guarda mejor modelo automáticamente

**Tiempo estimado**: 30-60 minutos (depende de timesteps y hardware)

### 3. `evaluate_sb3.py` - Evaluación de Modelos

**Propósito**: Evaluar modelos entrenados.

**Uso básico**:
```bash
# Evaluar un modelo
python evaluate_sb3.py ./models/sb3_ppo/best_model.zip --episodes 20

# Visualizar episodio detallado
python evaluate_sb3.py ./models/sb3_ppo/best_model.zip --visualize

# Comparar múltiples modelos
python evaluate_sb3.py --compare \
    ./models/sb3_ppo/best_model.zip \
    ./models/sb3_sac/best_model.zip \
    --episodes 20
```

**Características**:
- Métricas detalladas (reward, success rate, episode length)
- Visualización paso a paso
- Comparación de modelos
- Análisis de trayectorias

## Flujo de Trabajo Recomendado

### Paso 1: Verificación Rápida
```bash
# Verificar que todo funciona
python quick_train_sb3.py
```

**Resultado esperado**:
- El agente mejora respecto a la política aleatoria
- Se guarda un modelo en `./models/quick_test/`

### Paso 2: Entrenamiento Completo
```bash
# Entrenar con PPO (recomendado para empezar)
python train_sb3.py --algorithm ppo --timesteps 500000 --n-envs 4
```

**Durante el entrenamiento**:
- Se guardan checkpoints cada 50,000 pasos
- Se evalúa el modelo cada 10,000 pasos
- Se guarda el mejor modelo automáticamente

**Monitorear progreso**:
```bash
# En otra terminal
tensorboard --logdir ./logs/sb3_ppo
```

Abrir navegador en: `http://localhost:6006`

### Paso 3: Evaluación
```bash
# Evaluar el mejor modelo
python evaluate_sb3.py ./models/sb3_ppo/best_model.zip --episodes 20

# Ver episodio detallado
python evaluate_sb3.py ./models/sb3_ppo/best_model.zip --visualize
```

### Paso 4: Comparación (Opcional)
```bash
# Entrenar con SAC para comparar
python train_sb3.py --algorithm sac --timesteps 500000

# Comparar PPO vs SAC
python evaluate_sb3.py --compare \
    ./models/sb3_ppo/best_model.zip \
    ./models/sb3_sac/best_model.zip \
    --episodes 20
```

## Configuración de Algoritmos

### PPO (Proximal Policy Optimization)

**Ventajas**:
- Estable y robusto
- Funciona bien out-of-the-box
- Buen para entornos continuos

**Configuración por defecto**:
```python
learning_rate = 3e-4
n_steps = 2048
batch_size = 64
n_epochs = 10
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
```

**Cuándo usar**: Primera opción para la mayoría de casos.

### SAC (Soft Actor-Critic)

**Ventajas**:
- Sample-efficient (aprende más rápido)
- Off-policy (puede reutilizar experiencias)
- Bueno para control continuo

**Configuración por defecto**:
```python
learning_rate = 3e-4
buffer_size = 100000
batch_size = 256
tau = 0.005
gamma = 0.99
```

**Cuándo usar**: Si PPO es lento o necesitas más sample-efficiency.

## Estructura de Directorios

```
.
├── models/
│   ├── sb3_ppo/
│   │   ├── best_model.zip          # Mejor modelo (automático)
│   │   ├── ppo_quadrotor_50000_steps.zip
│   │   ├── ppo_quadrotor_100000_steps.zip
│   │   └── ppo_quadrotor_final.zip
│   ├── sb3_sac/
│   │   └── ...
│   └── quick_test/
│       └── ppo_quick_test.zip
│
├── logs/
│   ├── sb3_ppo/
│   │   └── PPO_1/                  # TensorBoard logs
│   └── sb3_sac/
│       └── SAC_1/
│
├── train_sb3.py
├── evaluate_sb3.py
└── quick_train_sb3.py
```

## Métricas de Éxito

### Hover Control (Tarea Básica)

**Objetivo**: Mantener el dron cerca del origen (0, 0, 0)

**Métricas**:
- **Reward promedio**: > -50 (bueno), > -20 (excelente)
- **Success rate**: > 50% (bueno), > 80% (excelente)
- **Distancia final**: < 0.5m (bueno), < 0.2m (excelente)

### Seguimiento Visual (Tarea de Filming)

**Objetivo**: Mantener la esfera magenta en la fracción ideal de la imagen de la cámara FPV

**Métricas**:
- **Distancia media al target**: < 2m (bueno), < 1m (excelente)
- **Fracción de imagen del target**: ~25% ideal (configurable con `ideal_fraction`)
- **Visibilidad del target**: > 80% de los steps (bueno)
- **Recompensa visual**: Positiva de forma consistente

**Baseline (política aleatoria)**:
- Reward promedio: ~ -400 a -600
- Success rate: 0%
- Distancia final: > 2m

## Troubleshooting

### Problema: "ModuleNotFoundError: No module named 'stable_baselines3'"

**Solución**:
```bash
pip install stable-baselines3[extra]
```

### Problema: Entrenamiento muy lento

**Soluciones**:
1. Reducir `n_envs` (menos entornos paralelos)
2. Reducir `n_steps` (actualizaciones más frecuentes)
3. Usar GPU si está disponible (automático)

### Problema: El agente no aprende

**Posibles causas**:
1. **Timesteps insuficientes**: Aumentar a 1M+
2. **Learning rate inadecuado**: Probar 1e-4 o 1e-3
3. **Algoritmo inadecuado**: Probar SAC en lugar de PPO

**Debug**:
```bash
# Ver curvas de aprendizaje en TensorBoard
tensorboard --logdir ./logs/sb3_ppo
```

### Problema: "CUDA out of memory"

**Solución**:
```bash
# Forzar uso de CPU
export CUDA_VISIBLE_DEVICES=""
python train_sb3.py ...
```

## Comparación con Controlador Manual

| Aspecto | Controlador Manual | SB3 (RL) |
|---------|-------------------|----------|
| **Desarrollo** | Requiere conocimiento de control | Automático |
| **Adaptabilidad** | Fijo, requiere reajuste manual | Se adapta automáticamente |
| **Rendimiento** | Depende del diseño | Puede superar controladores manuales |
| **Tiempo de setup** | Días/semanas | Horas (entrenamiento) |
| **Generalización** | Limitada | Buena (si se entrena bien) |
| **Interpretabilidad** | Alta | Baja (caja negra) |

## Entrenamiento Avanzado: Seguimiento Visual v2 con Curriculum

### 4. `train_lemniscate_v2.py` - Seguimiento con Curriculum Adaptativo

**Propósito**: Entrenar un agente PPO para seguir una trayectoria en lemniscata (∞) usando exclusivamente la cámara FPV, con un sistema de recompensa de 6 componentes y curriculum de 3 fases.

**Uso**:
```bash
python scripts/train_lemniscate_v2.py
```

**Características**:
- **Curriculum de 3 fases**:
  - Fase A (0–30%): Target fijo a 2m — aprender hover + centrado
  - Fase B (30–70%): Lemniscata lenta (0.02→0.16 m/s)
  - Fase C (70–100%): Velocidad completa (0.16→0.30 m/s)
- **Transfer learning**: Carga CNN pre-entrenada de `models/goal_controller/best_model.zip`, congela el feature extractor en Fase A, lo descongela en Fase B con lr=1e-5
- **Domain randomization progresiva**: Vinculada al rendimiento del agente (no al progreso temporal)
- **VecNormalize**: Normalización de rewards para estabilidad
- **Entropy bumps**: Re-exploración forzada al cambiar de fase

**Configuración PPO v2**:
```python
learning_rate = 1e-4 → 1e-5  # Decay lineal
n_steps = 4096
batch_size = 64               # 640 gradient steps/update
n_epochs = 10
clip_range = 0.15             # Compromiso freeze/fine-tuning
ent_coef = 0.01 → 0.003      # Schedule + bumps por fase
max_grad_norm = 0.5
```

**Recompensa v2 (6 componentes)**:
| Componente | Rango | Señal |
|---|---|---|
| R_survival | +0.05 | Incentivo por mantenerse en vuelo |
| R_stability | 0 → +1.0 | Penaliza velocidad angular y tilt |
| R_centering | 0 → +3.0 | Target centrado en imagen |
| R_scale | 0 → +2.0 | Fracción de imagen ideal (25%) |
| R_discovery | +3.0 | Bonus al re-encontrar target perdido |
| R_not_visible | -0.5/step | Penalización sin target visible |

**Outputs**:
```
models/lemniscate_v2/
├── best_model.zip          # Mejor modelo
├── training_log.csv        # Métricas por episodio (16 columnas)
├── vecnormalize.pkl        # Estadísticas de normalización
└── recordings/             # Vídeos periódicos
```

### 5. `test_lemniscate_v2.py` - Evaluación de Modelo v2

**Propósito**: Evaluar un modelo v2 entrenado con telemetría detallada.

**Uso**:
```bash
python tests/test_lemniscate_v2.py
```

**Características**:
- Ejecuta N episodios de evaluación (default 5)
- Graba vídeo side-by-side (FPV + aérea)
- CSV per-step con los 6 componentes de reward
- JSON de resumen con métricas agregadas

### 6. `test_spawn_positions.py` - Validación de Inicializaciones

**Propósito**: Verificar visualmente que las posiciones de spawn dron-target son correctas.

**Uso**:
```bash
python tests/test_spawn_positions.py
```

**Outputs**: `experiments/spawn_test/spawn_positions.mp4` + `spawn_summary.png`

## Métricas de Éxito: Seguimiento v2

**Objetivo**: Mantener el target magenta centrado y a la fracción de imagen ideal (~25%)

**Métricas**:
- **Ratio tracking/hover**: > 5× (verificado en 11×)
- **Visibilidad del target**: > 75% de los steps (threshold Fase B)
- **R_centering medio**: > 2.0/3.0 (threshold Fase B)
- **Episodios de 1000 steps**: > 80% sin truncaciones

## Entrenamiento Avanzado: Hover Tracking v3 con SAC

### 7. `train_hover_track.py` - Hover Tracking con Cámara Vertical

**Propósito**: Entrenar un agente SAC para mantenerse sobre una esfera magenta, observándola con una cámara vertical (pitch=-90°) y manteniéndola centrada a la distancia calibrada (1.394m).

**Uso**:
```bash
python scripts/train_hover_track.py --timesteps 200000
```

**Características**:
- **Observación centroide** (19-D flat): sin CNN, features extraídas por HSV
- **SAC**: off-policy, entropía auto-ajustada, replay buffer ~21 MB
- **Reward de 3 componentes** (rango [-1, +4]): stability, centering, scale
- **Init constrained**: near-hover (pos ±0.2m, vel ±0.1 m/s)

**Configuración SAC**:
```python
learning_rate = 3e-4
buffer_size = 300_000    # ~21 MB (vs ~1.2 GB con CNN)
learning_starts = 5_000  # 10 episodios de diversidad
batch_size = 256
gamma = 0.995            # horizonte ~200 steps
train_freq = 4           # agrupa gradient steps
gradient_steps = 4
ent_coef = 'auto'        # entropía auto-ajustada
net_arch = [128, 64]     # ~27k params por red
```

**Resultados (200k steps)**:
| Métrica | Inicio | Final | Cambio |
|---|---|---|---|
| Reward | 152 | 1,574 | +10.3× |
| Visibilidad | 62% | 91% | +30% |
| Centering dist | 0.43 | 0.27 | -37% |
| r_stability | 0.59 | 0.99 | +68% |
| Episode length | 98 | 500 (max) | Episodios completos |

### 8. `train_spiral_follow.py` - Modelo de Búsqueda en Espiral

**Propósito**: Entrenar un agente PPO para seguir una espiral de Arquímedes (búsqueda cuando el target se pierde).

**Uso**:
```bash
python scripts/train_spiral_follow.py --timesteps 500000
```

**Características**:
- **Observación** (18-D): 13 estado + 5 referencia de espiral
- **PPO MlpPolicy** con net_arch=[64, 32]
- **Curriculum de 2 fases**: ω_scale 0.3→1.0
- **Modelo usado como fallback** por el `SpiralSearchController`

## Entrenamiento Avanzado: Hover Tracking v3 con Phase 0 de Estabilización

### 9. `train_hover_track_v3.py` — Curriculum de 4 Fases

**Propósito**: Entrenar un agente SAC con una Phase 0 de estabilización pura (sin target) antes del tracking visual, y una gaussiana de centering más apretada para forzar precisión.

**Uso**:
```bash
python scripts/train_hover_track_v3.py --timesteps 1500000 --no-display
```

**Características principales**:
- **Phase 0** (0–8%): El target se mueve fuera del FOV; solo `R_survival + 2·R_stability + 2·R_vel_cancel`. El agente aprende a estabilizarse antes de intentar servoing visual.
- **Fases A/B/C** (8–100%): R_centering más estrecha `4.0×exp(-6d²)` (sigma efectivo 0.41 vs 0.58 de v2). Episodios de 30s.
- **R_center_vel**: Bonus por reducir distancia al centro (derivada negativa, clamped a [0,1]).
- **GPU training**: `.vgpu` env con PyTorch+CUDA 12.8 — ~30-50% más rápido que CPU.

**Configuración SAC v3**:
```python
learning_rate = 3e-4
buffer_size = 500_000
batch_size = 256
net_arch = [256, 128]    # ~195k parámetros por red
gamma = 0.995
train_freq = 4
ent_coef = 'auto'
```

**Resultados (1.5M steps)**:

| Checkpoint | Surv% | Reward | Vis% | Cent. | Easy | Med | Hard |
|---|---|---|---|---|---|---|---|
| 750k | 66.7 | 2,174 | 74.0 | 0.616 | 100% | 80% | 20% |
| 850k | 80.0 | 2,920 | 91.1 | 0.467 | 80% | 80% | 80% |
| **900k** | **80.0** | **2,974** | 83.7 | 0.479 | **100%** | **100%** | 40% |
| 1.5M | 86.7 | 2,102 | 75.9 | 0.658 | 80% | 80% | 100% |

**Checkpoint recomendado: 900k** — excelente en easy/medium, base para fine-tune v3.1.

---

## Entrenamiento Avanzado: Hover Tracking v3.1 — Fine-Tune con Reward Multiplicativa

### 10. `train_hover_track_v3_1.py` — Fine-Tune desde v3/900k

**Propósito**: Fine-tune del mejor checkpoint v3 (900k) corrigiendo el desequilibrio de la reward aditiva (R_centering dominaba 4:1 sobre R_stability). La nueva reward hace que el tracking solo se cobre si el vuelo es estable.

**Diagnóstico del problema**:
```
R v3 (aditiva):     total = R_stab(1.0) + R_cent(4.0) + R_vel(1.0) + R_scale(1.0) + R_inv(-1.0)
Inestable+centrado: 0.2   + 4.0         + 0.5         + 0.5         -  0           = 5.2  ← casi igual
Estable+centrado:   1.0   + 4.0         + 0.5         + 0.5         -  0           = 6.0
```

**Solución (reward v3.1 multiplicativa)**:
```python
R_tracking = R_centering + R_center_vel + R_scale           # 0 → 6.0
total = R_stability × (R_tracking + 0.5) + R_vel_damp + R_smooth + R_invisible
```

| Escenario | R v3 | R v3.1 | Diferencia |
|---|---|---|---|
| Estable + centrado (stab=0.95, cent=4.0) | 6.95 | 6.68 | -4% |
| Inestable + centrado (stab=0.20, cent=4.0) | 6.20 | 1.80 | **-71%** |
| Estable sin centrar (stab=0.95, cent=0.5) | 1.75 | 1.74 | ~igual |

**Nuevos componentes**:

```python
R_smooth  = -0.3 × ||action_t - action_{t-1}||²  # [-0.3, 0]  — suavidad de motores
R_vel_damp = 0.5 × exp(-4 × ||vel||²)             # [0, 0.5]   — baja velocidad lineal durante tracking
```

**Uso**:
```bash
# Con el checkpoint recomendado (900k de v3)
python scripts/train_hover_track_v3_1.py --no-display

# Para seleccionar base manualmente
python scripts/train_hover_track_v3_1.py \
    --base-checkpoint ./models/hover_track_v3/checkpoints/model_900000_steps.zip
```

**Configuración del fine-tune**:
```python
learning_rate = 1e-4           # Reducido (3e-4 en v3) — actualizaciones conservadoras
buffer_size = 500_000
# ⚠️ Replay buffer vaciado: transiciones v3 tienen rewards incompatibles con v3.1
# (magnitudes y gradientes distintos, corrupen los value estimates del critic)
timesteps = 500_000
curriculum = ['B', 'C']       # Sin Phase 0 ni A — el agente ya sabe volar
```

**Resultados de la evaluación formal** (30 episodios × checkpoint, target estático):

| Checkpoint | Surv% | R. medio | Vis% | Cent. | Jerk | R_stab | Easy | Med | Hard |
|---|---|---|---|---|---|---|---|---|---|
| 100k | 40.0 | 2,044 | 54.8 | 0.701 | 0.068 | 0.947 | 60% | 50% | 10% |
| 200k | 66.7 | 4,396 | 74.6 | 0.489 | 0.124 | 0.959 | 100% | 60% | 40% |
| 300k | 53.3 | 4,114 | 77.6 | 0.538 | 0.131 | 0.950 | 60% | 60% | 40% |
| **400k** | **93.3** | 6,112 | **92.6** | 0.452 | **0.123** | **0.990** | **100%** | **100%** | **80%** |
| 500k | 83.3 | **6,577** | 86.7 | **0.396** | 0.201 | 0.990 | 100% | 80% | 70% |

**Checkpoint ganador: 400k** — mayor supervivencia, menor jerk, mejor equilibrio entre tiers.

**Outputs**:
```
models/hover_track_v3_1/
├── best_model.zip
├── training_log.csv          # métricas por episodio (incluye r_vel_damp, r_smooth, jerk)
└── checkpoints/
    ├── model_50000_steps.zip
    ├── model_100000_steps.zip
    ├── ...
    └── model_500000_steps.zip

experiments/hover_track_v3_1/
├── checkpoint_comparison.json  # datos completos por checkpoint y tier
├── checkpoint_episodes.csv
├── checkpoint_global.png       # 8 paneles comparativos
└── checkpoint_tiers.png        # boxplots de jerk por tier
```

---

### 11. `tests/evaluate_hover_track_v3_1.py` — Evaluador Multi-Checkpoint

**Propósito**: Evaluación cuantitativa controlada de todos los checkpoints del fine-tune v3.1 sobre 3 tiers de dificultad, con métricas extendidas (jerk, vel_damp, smooth).

**Uso**:
```bash
python tests/evaluate_hover_track_v3_1.py --no-display
python tests/evaluate_hover_track_v3_1.py --model-dir ./models/hover_track_v3_1 --episodes 15
```

**Tiers de evaluación** (target estático):
| Tier | Offset init | Vel init | Descripción |
|---|---|---|---|
| `easy` | 0.3m | 0.10 m/s | Condiciones de Hover Track básico |
| `medium` | 0.6m | 0.25 m/s | Target descentrado, inicio perturbado |
| `hard` | 1.0m | 0.40 m/s | Condiciones post-espiral extremas |

**Métricas registradas**: survival_rate, total_reward, visibility_pct, mean_centering_dist, mean_action_jerk, r_stability, r_centering, r_vel_damp, r_smooth.

---

### 12. `tests/test_spiral_track_v3_1.py` — Test Integración con Target Móvil

**Propósito**: Evaluar los modelos v3.1 en el escenario de uso real: objetivo en movimiento (lemniscata de Bernoulli), con FSM de 2 estados que activa la espiral RL cuando el target se pierde.

**Fórmula de la lemniscata**:
```
x(t) = a·cos(t) / (1 + sin²(t))
y(t) = a·sin(t)·cos(t) / (1 + sin²(t))
velocidad 0.3 m/s → ω ≈ 0.6 rad/s
```

**FSM simplificada** (sin BRAKE ni HANDOFF):
```python
class SpiralTrackFSM:
    # TRACK → SEARCH: k=20 steps consecutivos sin target
    # SEARCH → TRACK: inmediato al detectar target
    def get_action(self, obs19, target_visible, sac_model, state13, ...):
        if self._state == TRACK:
            if not target_visible:
                self._invisible_count += 1
                if self._invisible_count >= self.k_invisible:
                    self._state = SEARCH
                    self._reset_spiral(pos_x, pos_y)
            else:
                self._invisible_count = 0
        elif self._state == SEARCH:
            if target_visible:
                self._state = TRACK
                self._invisible_count = 0
        if self._state == SEARCH:
            spiral_obs = self._build_spiral_obs(state13)   # 18-D
            return spiral_model.predict(spiral_obs, deterministic=True)
        return sac_model.predict(obs19, deterministic=True)
```

**Uso**:
```bash
python tests/test_spiral_track_v3_1.py --no-display
python tests/test_spiral_track_v3_1.py --models 400k 500k --episodes 10
```

**Escenarios** (4 niveles):

| Escenario | Speed | Offset | Vel init | Descripción |
|---|---|---|---|---|
| `slow_easy` | 0.10 m/s | 0.3m | 0.10 m/s | ≈ v4 Phase A |
| `medium` | 0.22 m/s | 0.6m | 0.20 m/s | ≈ v4 Phase B |
| `fast_hard` | 0.35 m/s | 1.0m | 0.35 m/s | ≈ v4 Phase C |
| `recovery` | 0.20 m/s | 2.5m | 0.30 m/s | Recuperación extrema |

**Resultados clave**:

| Modelo | Slow Easy | Medium | Fast Hard | Recovery | Jerk | Post-handoff |
|---|---|---|---|---|---|---|
| 150k | ~40% | **0%** | ~0% | ~0% | — | 0.82 |
| 250k | ~60% | ~40% | ~20% | ~20% | — | 0.85 |
| **400k** | ~80% | 60% | ~40% | ~40% | **0.123** | 0.839 |
| 500k | ~80% | 80% | ~40% | ~40% | 0.201 | 0.795 |

**Puntuación compuesta (target estático + target móvil)**:
- 400k: 0.633 (GANADOR)
- 500k: 0.595

---

## Entrenamiento Avanzado: Hover Tracking v4 — Objetivo Móvil

### 13. `scripts/train_hover_track_v4.py` — Fine-Tune con Target en Lemniscata

**Propósito**: Primer entrenamiento del proyecto con objetivo dinámico para el SAC. Fine-tune del mejor checkpoint v3.1 (400k) con target que sigue una lemniscata de Bernoulli.

**Decisión de diseño — Eliminar BRAKE y HANDOFF**:

El pipeline clásico (v3.1) tenía 5 estados:
```
STABILIZE → SEARCH → BRAKE → HANDOFF → TRACK
```

En v4 se simplifica a 2:
```
SEARCH ←→ TRACK
```

**Justificación**: El curriculum Phase C de v4 entrena al SAC en las mismas condiciones que BRAKE/HANDOFF gestionaban (vel_init hasta ±0.60 m/s, offset hasta 1.2m). El SAC v4 aprenderá implícitamente a manejar la transición post-espiral sin controladores intermedios, creando un agente más robusto e independiente.

**Uso**:
```bash
# Con checkpoint recomendado (v3.1/400k)
python scripts/train_hover_track_v4.py \
    --base-checkpoint ./models/hover_track_v3_1/checkpoints/model_400000_steps.zip

# Auto-detección del mejor disponible
python scripts/train_hover_track_v4.py --timesteps 750000 --no-display
```

**Clase `MovingTargetV4Wrapper`**:

Wrapper que posiciona el dron encima del target en cada reset (en lugar de mover el target):

```python
class MovingTargetV4Wrapper(Panda3DQuadrotorEnv):
    def reset(self, seed=None, options=None):
        self.target_speed = np.random.uniform(*self.target_speed_range)
        obs, info = super().reset(seed=seed, options=options)
        state = self.base_env.state.copy()
        # Posicionar dron encima del target con offset aleatorio
        state[0] = self.target_pos[0] + dx    # x
        state[2] = self.target_pos[1] + dy    # y
        state[4] = self.target_pos[2] + self.hover_height  # z
        self.base_env.state = state
        # Sincronización completa del pipeline Panda3D
        self._update_visualization()
        self.panda3d_app.graphicsEngine.renderFrame()
        self._capture_camera_images(force_capture=True)
        obs = self._build_observation(state.astype(np.float32))
        return obs, info
```

⚠️ **`min_start_distance = 0.0`**: Anula el default de 3.0m del modo moving — el target empieza en cualquier fase de su lemniscata.

**Curriculum de 3 fases (`CurriculumV4Callback`)**:

| Fase | Progreso | Speed target | Offset | Vel init | Simula |
|---|---|---|---|---|---|
| A | 0–30% | 0.05→0.15 m/s | 0.2→0.4m | 0.10→0.15 m/s | Movimiento suave, perturbación leve |
| B | 30–65% | 0.15→0.25 m/s | 0.4→0.7m | 0.15→0.30 m/s | Velocidad media, inicio descentrado |
| C | 65–100% | 0.25→0.40 m/s | 0.7→1.2m | 0.30→0.60 m/s | Velocidad alta, condiciones post-espiral |

**Auto-detección del checkpoint base**:
```python
def find_best_model():
    candidates = [
        './models/hover_track_v3_1/best_model.zip',
        './models/hover_track_v3/checkpoints/model_900000_steps.zip',
        './models/hover_track_v3/best_model.zip',
    ]
    for c in candidates:
        if Path(c).exists():
            return c
```

**Configuración del fine-tune v4**:
```python
learning_rate = 1e-4      # Conservador (igual que v3.1)
buffer_size = 500_000
# ⚠️ Replay buffer vaciado: transiciones v3.1 (target estático)
#    son incompatibles con dinámicas de target móvil
timesteps = 750_000       # 250k más que v3.1 — target móvil es más difícil
```

**Análisis de reward**: R_vel_damp sin cambios — pérdida de 0.15/step a 0.3 m/s = 3.75% de la señal de centering → completamente negligible.

**Métricas de monitorización adicionales**:
- `target_speed_mean`: velocidad media del target en el episodio (debe crecer con el curriculum)
- `mean_action_jerk`: debe mantenerse ≤0.123 (valor del 400k base)

**Outputs**:
```
models/hover_track_v4/
├── best_model.zip
├── training_log.csv
└── checkpoints/
    └── model_XXXXXX_steps.zip

experiments/hover_track_v4/
├── checkpoint_comparison.json
├── checkpoint_episodes.csv
├── checkpoint_global.png
└── checkpoint_tiers.png
```

---

### 14. `tests/evaluate_hover_track_v4.py` — Evaluador con Tiers de Velocidad

**Propósito**: Evaluación de los checkpoints v4 con tiers basados en **velocidad del target** (no en offset estático como v3.1).

**Uso**:
```bash
python tests/evaluate_hover_track_v4.py --no-display
python tests/evaluate_hover_track_v4.py --model-dir ./models/hover_track_v4 --episodes 10
```

**Tiers de evaluación** (target móvil en lemniscata):

| Tier | Speed | Offset | Vel init | Equivalencia |
|---|---|---|---|---|
| `slow` | 0.10 m/s | 0.3m | 0.10 m/s | v4 Phase A |
| `medium` | 0.25 m/s | 0.6m | 0.25 m/s | v4 Phase B |
| `fast` | 0.40 m/s | 1.0m | 0.40 m/s | v4 Phase C |

**Diferencia clave respecto a evaluate_hover_track_v3_1**: En v4, cada episodio usa el `MovingTargetV4Wrapper` — el dron se posiciona encima del target (con offset del tier) y el target comienza a moverse desde el instante 0. La evaluación es coherente con el entorno de entrenamiento.

**Métricas adicionales para target móvil**:
- `target_speed_actual`: velocidad real del target durante el episodio
- `drift_accumulated`: distancia media dron-target a lo largo del episodio (cuantifica seguimiento lateral)

---

## Resumen de Scripts por Versión

| Script | Tipo | Algoritmo | Base | Timesteps | Estado |
|---|---|---|---|---|---|
| `train_hover_track.py` | Entrenamiento | SAC | Scratch | 200k | ✅ Completado |
| `train_hover_track_v2.py` | Entrenamiento | SAC | v1 | 500k | ✅ Completado |
| `train_hover_track_v3.py` | Entrenamiento | SAC | Scratch | 1.5M | ✅ Completado |
| `train_hover_track_v3_1.py` | Fine-tune | SAC | v3/900k | 500k | ✅ Completado |
| `train_hover_track_v4.py` | Fine-tune | SAC | v3.1/400k | 750k | ⏳ Pendiente |
| `evaluate_hover_track_v3_1.py` | Evaluación | — | v3.1 checkpoints | — | ✅ Completado |
| `evaluate_hover_track_v4.py` | Evaluación | — | v4 checkpoints | — | ⏳ Pendiente |
| `test_spiral_track_v3_1.py` | Integración | SAC+PPO | v3.1 | — | ✅ Completado |

## Próximos Pasos

1. ✅ ~~Extender hover-track~~ **v3.1 completado**
2. ✅ ~~Evaluar con espiral~~ **test_spiral_track_v3_1 completado**
3. ✅ ~~Seleccionar checkpoint para v4~~ **400k recomendado**
4. **Entrenar v4**: `python scripts/train_hover_track_v4.py --base-checkpoint ./models/hover_track_v3_1/checkpoints/model_400000_steps.zip --no-display`
5. **Evaluar v4**: `python tests/evaluate_hover_track_v4.py --no-display`
6. **Validar pipeline simplificado**: Confirmar que SEARCH→TRACK directo funciona sin BRAKE/HANDOFF
7. **Tests de generalización**: Velocidades OOD (0.50+ m/s), espiral con offsets >2m

## Referencias

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [SAC Paper](https://arxiv.org/abs/1801.01290)
- [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) - Hyperparameters optimizados

## Contacto y Soporte

Para problemas o preguntas sobre el entrenamiento con SB3, consultar:
- Documentación oficial de SB3
- Issues en el repositorio del proyecto
- `PROJECT_HISTORY.md` para cambios recientes
