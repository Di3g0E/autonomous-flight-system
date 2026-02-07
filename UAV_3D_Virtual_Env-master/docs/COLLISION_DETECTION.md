# Sistema de Detección de Colisiones - Documentación

## Descripción General

Se ha implementado un sistema modular de detección de colisiones que integra Panda3D con el entorno Gymnasium del quadrotor, manteniendo la separación de responsabilidades y permitiendo entrenamiento sin dependencias de Panda3D.

## Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                    Aplicación del Usuario                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
       ┌───────────────┴───────────────┐
       │                               │
       ▼                               ▼
┌──────────────┐              ┌───────────────────┐
│ quadrotor_env│              │ Panda3DQuadrotor  │
│  (Física)    │◄─────────────┤   Env (Wrapper)   │
└──────────────┘              └─────────┬─────────┘
                                        │
                              ┌─────────┴─────────┐
                              │                   │
                              ▼                   ▼
                    ┌──────────────┐    ┌──────────────┐
                    │  Collision   │    │   Obstacle   │
                    │   Detector   │    │   Manager    │
                    └──────────────┘    └──────────────┘
```

## Componentes

### 1. `quadrotor_env.py` (Base)
- **Responsabilidad**: Física pura del quadrotor
- **Dependencias**: scipy, numpy, gymnasium
- **Sin Panda3D**: Puede usarse independientemente para entrenamiento headless

### 2. `collision_detector.py`
- **Responsabilidad**: Sistema de detección de colisiones
- **Características**:
  - Detección de colisiones usando Panda3D
  - Información detallada de colisiones (punto, normal, objeto)
  - Visualización de debug opcional
- **Clases**:
  - `CollisionDetector`: Detecta colisiones del quadrotor
  - `ObstacleManager`: Gestiona obstáculos en el entorno

### 3. `panda3d_quadrotor_env.py`
- **Responsabilidad**: Wrapper que integra física + visualización + colisiones
- **Características**:
  - Hereda de `gym.Env` (API Gymnasium)
  - Integración opcional con Panda3D
  - Puede funcionar sin Panda3D (modo headless)
  - Añade información de colisiones al `info` dict

## Uso

### Modo 1: Entrenamiento Headless (Sin Panda3D)

```python
from environment.quadrotor_env import quad

# Entrenamiento rápido sin visualización
env = quad(t_step=0.01, n=1000, direct_control=1)

observation, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### Modo 2: Wrapper sin Colisiones

```python
from environment.panda3d_quadrotor_env import Panda3DQuadrotorEnv

# Usa el wrapper pero sin Panda3D
env = Panda3DQuadrotorEnv(
    panda3d_app=None,
    quad_model=None,
    render_node=None,
    enable_collisions=False
)

# API idéntica a Gymnasium
observation, info = env.reset()
```

### Modo 3: Integración Completa con Panda3D

```python
from direct.showbase.ShowBase import ShowBase
from environment.panda3d_quadrotor_env import Panda3DQuadrotorEnv

class MyApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        
        # Setup modelos 3D (quad_model, render, etc.)
        # ...
        
        # Crear entorno con colisiones
        self.env = Panda3DQuadrotorEnv(
            panda3d_app=self,
            quad_model=self.quad_model,
            render_node=self.render,
            enable_collisions=True,
            collision_radius=0.3,
            collision_penalty=-100.0
        )
        
        # Añadir obstáculos
        self.env.add_box_obstacle(
            position=(3, 0, 2.5),
            size=(0.5, 4, 5),
            name="wall"
        )
        
        self.env.add_sphere_obstacle(
            position=(1.5, 1.5, 2),
            radius=0.4,
            name="pillar"
        )
        
        # Habilitar visualización de debug
        self.env.enable_collision_debug()
```

## API de Colisiones

### Añadir Obstáculos

#### Box Obstacle
```python
env.add_box_obstacle(
    position=(x, y, z),      # Posición del centro
    size=(width, depth, height),  # Dimensiones
    name="obstacle_name"
)
```

#### Sphere Obstacle
```python
env.add_sphere_obstacle(
    position=(x, y, z),      # Posición del centro
    radius=0.5,              # Radio
    name="obstacle_name"
)
```

#### Model Collision
```python
# Añadir colisión a un modelo 3D existente
env.add_model_collision(
    model_node=scene_model,
    name="scene_collision"
)
```

### Información de Colisiones

El diccionario `info` retornado por `step()` contiene:

```python
{
    'collision_occurred': bool,  # True si hubo colisión
    'collision': {
        'has_collision': bool,
        'collision_point': np.array([x, y, z]),  # Punto de colisión
        'collision_normal': np.array([nx, ny, nz]),  # Normal de superficie
        'collision_object': str,  # Nombre del objeto
        'distance_to_collision': float  # Distancia al punto
    },
    # ... otros campos estándar
}
```

### Configuración

```python
# Cambiar radio de colisión
env.collision_detector.set_collision_radius(0.5)

# Cambiar penalización por colisión
env.set_collision_penalty(-150.0)

# Habilitar/deshabilitar visualización de debug
env.enable_collision_debug()
env.disable_collision_debug()

# Limpiar todos los obstáculos
env.clear_obstacles()
```

## Comportamiento de Colisiones

### Cuando ocurre una colisión:

1. **`terminated = True`**: El episodio termina inmediatamente
2. **Penalización de reward**: Se añade `collision_penalty` al reward
3. **Info actualizado**: `info['collision_occurred'] = True`
4. **Detalles disponibles**: Información completa en `info['collision']`

### Ejemplo de manejo:

```python
observation, reward, terminated, truncated, info = env.step(action)

if info['collision_occurred']:
    collision_info = info['collision']
    print(f"Colisión con: {collision_info['collision_object']}")
    print(f"Punto: {collision_info['collision_point']}")
    print(f"Penalización: {env.collision_penalty}")
```

## Máscaras de Colisión

El sistema usa máscaras de bits de Panda3D:

- **Quadrotor**: `FromCollideMask = BitMask32.bit(1)` (detecta grupo 1)
- **Obstáculos**: `IntoCollideMask = BitMask32.bit(1)` (pertenecen al grupo 1)

Esto permite control fino sobre qué objetos colisionan entre sí.

## Ventajas de esta Arquitectura

### ✅ Separación de Responsabilidades
- Física independiente de visualización
- Colisiones opcionales, no obligatorias

### ✅ Flexibilidad
- Entrenamiento headless (rápido)
- Visualización cuando se necesita
- Colisiones configurables

### ✅ Compatibilidad
- API Gymnasium estándar
- Compatible con Stable-Baselines3, Ray RLlib, etc.
- Funciona con o sin Panda3D instalado

### ✅ Escalabilidad
- Fácil añadir nuevos tipos de obstáculos
- Configuración dinámica de colisiones
- Debug visual opcional

## Ejemplos Incluidos

1. **`example_collision_detection.py`**: Ejemplos sin Panda3D
2. **`example_panda3d_integration.py`**: Integración completa con Panda3D
3. **`test_gym_wrapper.py`**: Tests del wrapper Gymnasium

## Próximos Pasos

### Mejoras Sugeridas:

1. **Sensores de Proximidad**
   - Añadir raycast para detección de distancia
   - Simular LiDAR/sensores ultrasónicos

2. **Reward Shaping**
   - Recompensa por mantener distancia de obstáculos
   - Penalización gradual según proximidad

3. **Obstáculos Dinámicos**
   - Soporte para obstáculos móviles
   - Predicción de trayectorias

4. **Optimización**
   - Spatial hashing para muchos obstáculos
   - LOD para colisiones complejas

## Troubleshooting

### Problema: "Panda3D not available"
**Solución**: Normal si entrenas sin Panda3D. Usa `quad` directamente o instala Panda3D.

### Problema: No se detectan colisiones
**Verificar**:
1. `enable_collisions=True` en el constructor
2. Obstáculos añadidos correctamente
3. Máscaras de colisión configuradas
4. Radio de colisión apropiado

### Problema: Colisiones falsas
**Solución**: Ajustar `collision_radius` a un valor más pequeño

## Referencias

- [Panda3D Collision Detection](https://docs.panda3d.org/1.10/python/programming/collision-detection/index)
- [Gymnasium API](https://gymnasium.farama.org/)
- Documentación del proyecto en `GYMNASIUM_WRAPPER.md`
