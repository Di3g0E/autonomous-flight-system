# UAVs Visual Simulation, A Python Approach

For full documentation of the project, please refer to: https://osf.io/a9s54/

The presented software is a solution to a common problem in machine learning and computer vision applied to UAVS. 

In the early phases of an algorithm, it is necessary to be able to simulate it and observe if its running as intended, making necessary corrections and bug catching before downloading and running it in a real platform, enabling a faster development speed and lowering costs of Implementation.

Simulating cameras with good performance is a particularly difficult task, even more so on python, developing one from scratch would take a lot of time, effort, and a good performance wasn't garanteed, so a python game engine was chosen, called Panda3D. 

With this engine and its integration on the presented software, it is possible to simulate commom UAVs missions, as well as its dynamics, on-board or off-board cameras, object visualisation and colision detection. 


At this moment, the presented software is capable of:

1. Fully functional quadrotor
2. On-board and off-board cameras
3. Direct OpenCv integration. 
4. Fully Customizable Dynamics
5. Fully Customizable Scenario Model
6. Fully Customizable UAV Model

In future versions we plan to include:

1. Object Colision Detection


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
El proyecto está organizado siguiendo los estándares modernos de Python:

- **`src/`**: Carpeta principal del código fuente.
  - **`envs/`**: Entornos Gymnasium (Quadrotor, Panda3D, Detección de Colisiones).
  - **`agents/`**: Implementaciones de agentes de RL y utilidades de entrenamiento.
  - **`simulation/`**: Utilidades de Panda3D (setup del mundo, cámara, física).
  - **`vision/`**: Módulos de visión artificial.
- **`scripts/`**: Scripts ejecutables para simulación (`run_simulation.py`), entrenamiento (`train_sb3.py`) y evaluación.
- **`tests/`**: Suite de pruebas para verificar la integridad del sistema.
- **`assets/`**: Recursos estáticos (modelos 3D en `models/` y texturas en `textures/`).
- **`weights/`**: Pesos de los modelos entrenados.

---

### Ejecutar Pruebas y Scripts Adicionales
Para verificar la integridad o realizar entrenamientos:

```bash
# Ejecutar los tests
pytest tests/

# Entrenar un agente con Stable-Baselines3
python scripts/train_sb3.py

# Ver un agente ya entrenado
python scripts/evaluate_sb3.py
```

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
