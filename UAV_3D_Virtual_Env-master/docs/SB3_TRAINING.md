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

## Próximos Pasos

1. **Extender hover-track**: +100-200k steps para convergencia completa (centering < 0.25).
2. **Evaluar con espiral**: Validar handoff TRACK→SEARCH→HANDOFF con vídeos.
3. **Target móvil**: Probar generalización con target lento (0.05 m/s) sin reentrenar.
4. **Fase 2**: Entrenar con target móvil (lemniscata) manteniendo obs 19-D.
5. **Spawn off-axis**: Fase 3 con búsqueda espiral al inicio del episodio.

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
