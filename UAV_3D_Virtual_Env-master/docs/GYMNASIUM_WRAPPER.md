# Gymnasium Wrapper para Quadrotor Environment

Este documento describe la implementación del wrapper de Gymnasium para el entorno de simulación de quadrotor.

## Descripción General

La clase `quad` en `environment/quadrotor_env.py` ahora hereda de `gym.Env`, haciéndola compatible con el estándar Gymnasium y con bibliotecas populares de Reinforcement Learning como Stable-Baselines3, Ray RLlib, etc.

## Espacios de Acción y Observación

### Action Space
- **Tipo**: `Box(4,)` con valores en `[-1, 1]`
- **Descripción**: Fuerzas normalizadas de los 4 motores del quadrotor
- **Componentes**:
  - `action[0]`: Control de thrust (empuje vertical)
  - `action[1]`: Control de roll (balanceo lateral)
  - `action[2]`: Control de pitch (cabeceo)
  - `action[3]`: Control de yaw (guiñada)

### Observation Space
- **Tipo**: `Box(13,)` con límites basados en las bounding boxes del entorno
- **Descripción**: Estado completo del quadrotor
- **Componentes**:
  ```
  [x, vx, y, vy, z, vz, q0, q1, q2, q3, w_x, w_y, w_z]
  ```
  - `x, y, z`: Posición en el marco inercial (m)
  - `vx, vy, vz`: Velocidad en el marco inercial (m/s)
  - `q0, q1, q2, q3`: Orientación en cuaterniones (normalizado)
  - `w_x, w_y, w_z`: Velocidades angulares en el marco del cuerpo (rad/s)

## API de Gymnasium

### Inicialización
```python
from environment.quadrotor_env import quad

env = quad(
    t_step=0.01,        # Paso de integración (s)
    n=1000,             # Número máximo de pasos
    euler=0,            # 0: usar cuaterniones, 1: usar ángulos de Euler
    direct_control=1,   # 0: control indirecto, 1: control directo
    T=1,                # Pasos de warm-up
    render_mode=None    # Modo de renderizado ('human' o None)
)
```

### Métodos Principales

#### `reset(seed=None, options=None)`
Reinicia el entorno al estado inicial.

**Parámetros**:
- `seed` (int, opcional): Semilla para reproducibilidad
- `options` (dict, opcional): Puede contener `'det_state'` para establecer un estado inicial determinista

**Retorna**:
- `observation` (np.ndarray): Observación inicial de forma (13,)
- `info` (dict): Información adicional
  - `'solved'`: Si se alcanzó el objetivo
  - `'angular_position'`: Ángulos de Euler actuales

**Ejemplo**:
```python
# Reset aleatorio
observation, info = env.reset(seed=42)

# Reset con estado determinista
det_state = np.zeros(13)
det_state[6] = 1.0  # quaternion q0 = 1
observation, info = env.reset(options={'det_state': det_state})
```

#### `step(action)`
Ejecuta un paso de tiempo en el entorno.

**Parámetros**:
- `action` (np.ndarray): Acción a aplicar, forma (4,), valores en [-1, 1]

**Retorna**:
- `observation` (np.ndarray): Nueva observación, forma (13,)
- `reward` (float): Recompensa obtenida
- `terminated` (bool): Si el episodio terminó (violación de límites o éxito)
- `truncated` (bool): Si el episodio se truncó (límite de tiempo)
- `info` (dict): Información adicional
  - `'solved'`: Si se alcanzó el objetivo
  - `'angular_position'`: Ángulos de Euler actuales
  - `'clipped_action'`: Acción después de clipping
  - `'timestep'`: Paso de tiempo actual

**Ejemplo**:
```python
action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)
```

#### `render()`
Renderiza el entorno (actualmente placeholder).

**Retorna**: `None`

#### `close()`
Limpia recursos del entorno.

## Función de Recompensa

La función de recompensa varía según el modo de operación:

### Modo base (hover / goal-reaching)
- **Recompensa de shaping**: Basada en la distancia a la posición objetivo (origen por defecto)
- **Penalización de control**: Penaliza acciones extremas
- **Recompensa de éxito**: +500 cuando se alcanza el objetivo
- **Penalización de fallo**: -200 cuando se violan los límites

### Modo filming (`filming_mode=True`)
En filming mode, la recompensa del entorno base se **descarta completamente** (se sustituye por `0.0`), conservando únicamente la penalización de `-200` si el dron sale de los bounding boxes. Toda la señal de recompensa proviene de la **recompensa visual basada en fracción**:

- **Zona ideal** (`|fraction - ideal_fraction| ≤ fraction_tolerance`): recompensa positiva gaussiana con máximo `max_visual_reward` cuando la fracción coincide exactamente con el ideal.
- **Zona exterior** (error > tolerance): penalización exponencial creciente, limitada a `-max_visual_reward`.
- **Objetivo no visible**: penalización fija de `-5.0` para incentivar la búsqueda.

#### Parámetros configurables de la recompensa visual
| Parámetro | Default | Descripción |
|---|---|---|
| `ideal_fraction` | 0.25 | Fracción ideal de píxeles del objetivo sobre el total |
| `fraction_tolerance` | 0.05 | Tolerancia alrededor del ideal para recompensa positiva |
| `max_visual_reward` | 1000.0 | Recompensa máxima en el punto ideal |

## Condiciones de Terminación

### Terminated (Episodio terminado)
- Violación de bounding boxes de posición, velocidad o ángulos
- Alcance del objetivo (estado estable cerca del origen) — solo en modo base

### Truncated (Episodio truncado)
- Se alcanza el número máximo de pasos (`n`)
- Search timeout: si el objetivo no ha sido visto tras `search_timeout_steps` pasos

## Bounding Boxes

Los límites del entorno están definidos por:
- **Posición**: ±5 m
- **Velocidad**: ±10 m/s
- **Ángulos**: ±π/2 rad
- **Velocidad angular**: ±10 rad/s

## Ejemplos de Uso

### Ejemplo Básico
```python
import numpy as np
from environment.quadrotor_env import quad

# Crear entorno
env = quad(t_step=0.01, n=1000, direct_control=1)

# Ejecutar episodio
observation, info = env.reset(seed=42)
done = False

while not done:
    action = env.action_space.sample()  # Acción aleatoria
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
```

### Ejemplo con Control Simple
```python
# PD controller simple para hover
Kp, Kd = 0.3, 0.2

observation, info = env.reset()

for step in range(500):
    position = observation[0:6:2]
    velocity = observation[1:6:2]
    
    # Control proporcional-derivativo
    error_pos = -position
    error_vel = -velocity
    
    action = np.clip(Kp * error_pos + Kd * error_vel, -1, 1)
    action = np.append(action, 0.0)  # añadir control de yaw
    
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        break
```

### Ejemplo con Stable-Baselines3
```python
from stable_baselines3 import PPO
from environment.quadrotor_env import quad

# Crear entorno
env = quad(t_step=0.01, n=1000, direct_control=1)

# Entrenar agente PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Evaluar
observation, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

## Scripts de Prueba

### `test_gym_wrapper.py`
Script de prueba que verifica la compatibilidad con Gymnasium:
```bash
python test_gym_wrapper.py
```

### `example_gym_usage.py`
Ejemplos de uso con diferentes estrategias de control:
```bash
python example_gym_usage.py
```

## Notas de Implementación

### Diferencias con la Versión Original
- **API estándar**: Ahora sigue la API de Gymnasium v0.26+
- **Separación terminated/truncated**: Distingue entre terminación natural y truncamiento por tiempo
- **Tipo de datos**: Observaciones y acciones son `float32` para compatibilidad con frameworks de RL
- **Info dict**: Información adicional disponible en cada paso

### Compatibilidad
- ✅ Gymnasium >= 0.26
- ✅ Stable-Baselines3 >= 2.0
- ✅ Ray RLlib
- ✅ CleanRL

## Próximos Pasos

1. **Wrappers adicionales**: 
   - Normalización de observaciones
   - Frame stacking
2. **Benchmarks**: Evaluar resultados del entrenamiento con el nuevo reward visual
3. **Ajuste de hiperparámetros**: Experimentar con `ideal_fraction`, `fraction_tolerance` y `max_visual_reward`

## Referencias

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [Original Repository](https://github.com/rafaelcostafrf/UAV_3d_virtual_env)
