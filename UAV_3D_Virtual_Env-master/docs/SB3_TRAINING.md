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

## Próximos Pasos

1. **Entrenamiento con colisiones**:
   ```bash
   # Usar Panda3DQuadrotorEnv con obstáculos
   # Modificar train_sb3.py para usar el wrapper con colisiones
   ```

2. **Curriculum learning**:
   - Empezar con tarea fácil (hover)
   - Aumentar dificultad gradualmente
   - Añadir obstáculos progresivamente

3. **Reward shaping**:
   - Recompensa visual basada en fracción de imagen (ya implementada)
   - Parámetros configurables: `ideal_fraction`, `fraction_tolerance`, `max_visual_reward`
   - Modo filming: base env reward descartado, solo recompensa visual

4. **Transfer learning**:
   - Entrenar en simulación
   - Fine-tuning para diferentes condiciones
   - Adaptación a dron real

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
