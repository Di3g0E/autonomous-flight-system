# Pipeline de Hover Tracking con Cámara Vertical

## Visión General

El pipeline de hover tracking es el sistema principal de seguimiento visual del proyecto. El dron mantiene una cámara apuntando hacia abajo (pitch=-90°) y aprende a posicionarse sobre una esfera magenta, manteniéndola centrada a la distancia calibrada de **1.394m**.

```
┌─────────────────────────────────────────────────────────┐
│                   PIPELINE COMPLETO                      │
│                                                         │
│   [SEARCH] ─── target visible ───→ [TRACK]             │
│      │                                 │                │
│  PPO Espiral                      SAC v4               │
│  (espiral RL)               (target centrado)           │
│      │                                 │                │
│      └─── k=20 steps sin target ───────┘                │
└─────────────────────────────────────────────────────────┘
```

## Observación del Entorno (19-D)

El agente SAC recibe una observación plana de 19 dimensiones:

| Índices | Componentes | Descripción |
|---|---|---|
| 0–12 | Estado (13-D) | `[x, vx, y, vy, z, vz, q0, q1, q2, q3, wx, wy, wz]` |
| 13 | `cx` | Centroide x normalizado (-1=izquierda, +1=derecha) |
| 14 | `cy` | Centroide y normalizado (-1=arriba, +1=abajo) |
| 15 | `frac` | Fracción de píxeles magenta en imagen (0=invisible, ~0.25=ideal) |
| 16 | `visible` | Binario: 1.0 si target detectado, 0.0 si no |
| 17 | `Δcx` | Cambio en cx respecto al step anterior |
| 18 | `Δcy` | Cambio en cy respecto al step anterior |

**Detección HSV**: `cv2.inRange(hsv, (140, 100, 100), (170, 255, 255))` — rango magenta. Umbral: ≥3 píxeles = target visible.

**Ventajas de centroide vs CNN**:
- Replay buffer: ~21 MB vs ~1.2 GB con CNN (80× más pequeño)
- No necesita re-aprender la detección (HSV ya detecta perfectamente el magenta)
- MlpPolicy entrenada directamente sin feature extractor

## Acción (4-D)

```
action = [motor1, motor2, motor3, motor4]  ∈ [-1, 1]⁴
```

Normalizado: 0.0 = throttle mínimo, 1.0 = throttle máximo. El entorno mapea internamente a comandos de motor.

## Sistema de Reward

### Versión v3.1 (actual, multiplicativa)

```python
R_stability  = exp(-3·||ω||²) × exp(-5·||tilt||²)        # 0 → 1.0
R_centering  = 4.0 × exp(-6·d²)                           # 0 → 4.0  (d = dist_cent normalizada)
R_center_vel = max(0, -Δd × 3.0)                          # 0 → 1.0  (bonus por acercarse al centro)
R_scale      = asym_gaussian(frac, ideal=0.25)             # 0 → 1.0
R_vel_damp   = 0.5 × exp(-4·||vel||²)                     # 0 → 0.5
R_smooth     = -0.3 × ||action_t - action_{t-1}||²        # -0.3 → 0
R_invisible  = -1.0 (si target no visible)                 # 0 o -1.0

total = R_stability × (R_centering + R_center_vel + R_scale + 0.5)
       + R_vel_damp + R_smooth + R_invisible
```

**Por qué multiplicativa**: En la reward aditiva v3, el agente podía cobrar R_centering=4.0 con R_stability=0.2 — aprendía a corregir agresivamente sin ser estable. La multiplicación fuerza que el tracking solo se cobre en proporción a la estabilidad del vuelo. El +0.5 garantiza señal mínima de "mantente estable" cuando el target no está centrado.

**Ejemplo numérico**:

| Escenario | Reward v3 (aditiva) | Reward v3.1 (mult.) |
|---|---|---|
| Estable + centrado (stab=0.95, cent=4.0) | 6.95 | 6.68 |
| **Inestable + centrado** (stab=0.20, cent=4.0) | **6.20** | **1.80** (-71%) |
| Estable sin centrar (stab=0.95, cent=0.5) | 1.75 | 1.74 |

### Componente R_smooth

```python
R_smooth = -0.3 × ||action_t - action_{t-1}||²
```
- Δ=0.1 → penalización -0.003 (casi gratis)
- Δ=0.5 → penalización -0.075 (moderada)
- Δ=1.0 → penalización -0.300 (significativa)

El cuadrado concentra la penalización en cambios extremos, permitiendo ajustes normales sin coste.

### Componente R_vel_damp

```python
R_vel_damp = 0.5 × exp(-4 × ||vel||²)
```
- v=0.0 m/s → +0.50 (hover perfecto)
- v=0.3 m/s → +0.35 (seguimiento típico, pérdida de 0.15)
- v=0.5 m/s → +0.21

Para target móvil (v4): a 0.3 m/s la pérdida es 3.75% de la señal de centering — completamente negligible.

## Calibración del Sistema

### Hover Height (1.394m)

Distancia óptima dron-esfera para que ésta ocupe ~25% de la imagen 32×32:

| Método | Altura (m) |
|---|---|
| Teórico (pinhole, buffer 1920×1080) | 1.498 |
| Teórico (film nativo 3:2) | 1.380 |
| **Empírico 32×32 px** | **1.394** |

Se adopta el valor empírico a 32×32 porque es la misma resolución del pipeline de reward.

### Detección HSV

```python
# Rango HSV para magenta (H: 140-170°, S/V: >100)
lower = np.array([140, 100, 100])
upper = np.array([170, 255, 255])
mask = cv2.inRange(hsv_image, lower, upper)
pixels_detected = cv2.countNonZero(mask)
visible = pixels_detected >= 3
```

## Evolución del Modelo

### Línea temporal

```
v1 (200k)  →  v2 (500k)  →  v3 (1.5M)  →  v3.1 (500k FT)  →  v4 (750k FT)
  SAC          SAC+curr       +Phase0        multiplicativa      target móvil
  target fix   off target     +apretado      +R_smooth           lemniscata
  condic ideal  curriculum    centering      +R_vel_damp
```

### Comparativa de modelos

| Modelo | Base | Timesteps | Easy | Med | Hard | Jerk |
|---|---|---|---|---|---|---|
| v1 | Scratch | 200k | ~90% | — | — | — |
| v2 | v1 | 500k | 70% | 40% | 6% | — |
| v3/900k | Scratch | 900k | 100% | 100% | 40% | — |
| **v3.1/400k** | v3/900k | +400k | **100%** | **100%** | **80%** | **0.123** |
| v3.1/500k | v3/900k | +500k | 100% | 80% | 70% | 0.201 |
| v4 (completado) | v3.1/400k | +750k | 0% | 0% | 0% | 0.103 |

## Pipeline de Búsqueda

### Modelo de Espiral RL (`spiral_follow`)

Cuando el SAC pierde el target durante k=20 steps consecutivos, se activa el modelo PPO de espiral:

**Observación espiral (18-D)**:
```
[estado13, dx/r_norm, dy/r_norm, dz/h_norm, vx_ref_norm, vy_ref_norm]
```

Los últimos 5 valores son el error de posición y la dirección de velocidad respecto al punto de referencia de la espiral de Arquímedes.

**Parámetros de la espiral**:
| Parámetro | Valor |
|---|---|
| ω_base | 1.8 rad/s (adaptativo: reduce a radios grandes) |
| r_growth | 0.12 m/s |
| hover_height | 1.39m |
| arm_spacing | ~0.42m (58% overlap con vision_radius=0.5m) |
| k_invisible | 20 steps (0.2s) |

**Cobertura**: 100% (40/40 posiciones en barrido angular 8 ángulos × 5 distancias, 0.5–2.5m)

### FSM v4 (simplificada)

```
Estado: TRACK (SAC v4)
  Condición de salida: k_invisible >= 20 steps consecutivos sin target
  → transición a SEARCH

Estado: SEARCH (PPO espiral)
  Condición de salida: target detectado (1 frame)
  → transición inmediata a TRACK
```

**Por qué sin BRAKE/HANDOFF en v4**: El curriculum Phase C entrena al SAC en condiciones de alta velocidad lateral (0.30–0.60 m/s) y offset grande (0.7–1.2m), cubriendo exactamente las condiciones post-espiral. BRAKE/HANDOFF compensaban artificialmente esta brecha de distribución en v3.1; en v4 el agente la aprende directamente.

## Selección del Checkpoint para v4

### Proceso de evaluación

Se usaron dos datasets independientes con igual peso:

**1. Evaluación estática** (E8-Eval): target fijo, 30 episodios/checkpoint, 20s
**2. Evaluación dinámica** (T10): target móvil (lemniscata a 0.22 m/s), 5 episodios, 30s

### Puntuación compuesta

| Checkpoint | Surv. estático | Surv. medium (móvil) | Jerk | Post-handoff | **Score** |
|---|---|---|---|---|---|
| 150k | — | 0% | — | — | descartado |
| 400k | 93.3% | 60% | 0.123 | 0.839 | **0.633** |
| 500k | 83.3% | 80% | 0.201 | 0.795 | 0.595 |

### Por qué 400k sobre 500k

1. **Jerk 0.123 vs 0.201** (+63% en 500k): Con target móvil, acciones bruscas oscilan al dron y pierden el lock visual. La política más suave del 400k es más adecuada para seguimiento continuo.
2. **Supervivencia estática 93.3% vs 83.3%**: Mayor robustez general.
3. **Hard tier 80% vs 70%**: El 400k maneja mejor las condiciones que Phase C de v4 necesita como punto de partida.
4. **Post-handoff 0.839 vs 0.795**: Re-enganche más limpio tras espiral.

### Por qué no 150k

0% de supervivencia en el escenario medium con target móvil. El dron pasa >90% del tiempo en espiral (SEARCH) — nunca desarrolló la capacidad de tracking suficiente.

## Estructura de Archivos

```
scripts/
├── train_hover_track.py          # v1: SAC básico
├── train_hover_track_v2.py       # v2: curriculum con offset
├── train_hover_track_v3.py       # v3: Phase 0 + centering apretado
├── train_hover_track_v3_1.py     # v3.1: fine-tune multiplicativo
├── train_hover_track_v4.py       # v4: fine-tune con target móvil
└── train_spiral_follow.py        # Modelo de búsqueda en espiral

tests/
├── evaluate_hover_track_v3_1.py  # Evaluación multi-checkpoint v3.1
├── evaluate_hover_track_v4.py    # Evaluación multi-checkpoint v4
├── test_spiral_track_v3_1.py     # Test integración espiral+SAC con target móvil
└── test_hover_track_v3_video.py  # Grabación de vídeo 4 vistas

models/
├── hover_track/                  # v1: 200k steps
├── hover_track_v2/               # v2: 500k steps
├── hover_track_v3/               # v3: 1.5M steps (checkpoints cada 50k)
├── hover_track_v3_1/             # v3.1: 500k fine-tune (checkpoints cada 50k)
│   └── checkpoints/
│       └── model_400000_steps.zip   ← MEJOR CHECKPOINT (base para v4)
└── hover_track_v4/               # v4: 750k fine-tune (pendiente)

experiments/
├── hover_track_v3_1/
│   ├── checkpoint_comparison.json
│   ├── checkpoint_episodes.csv
│   ├── checkpoint_global.png
│   └── checkpoint_tiers.png
└── spiral_track_v3_1/
    ├── summary.json
    ├── results.csv
    ├── comparison_main.png
    ├── survival_rates.png
    └── videos/{modelo}_{escenario}/
```

## Resultados de v4 y Diagnóstico

El entrenamiento v4 completó 750k steps (~22h) con los siguientes resultados en la evaluación formal:

| Checkpoint | Surv% | Vis% | R_stab | Slow | Med | Fast |
|---|---|---|---|---|---|---|
| 200k | 6.7 | 16.5 | 0.928 | 0% | 20% | 0% |
| 300k | 0.0 | 33.6 | 0.807 | 0% | 0% | 0% |
| 400k | 0.0 | 33.8 | 0.647 | 0% | 0% | 0% |
| 750k | 0.0 | 40.7 | 0.586 | 0% | 0% | 0% |

**Causa**: Catastrophic forgetting en Phase C. Las condiciones extremas (vel_init hasta 0.60 m/s, offset hasta 1.2m) generaron episodios de 15-40 steps que saturaron el replay buffer, corrompiendo los value estimates del critic y degradando la estabilidad de vuelo (R_stability de 0.928 a 0.586).

**Observación positiva**: Los episodios largos de Phase C (≥100 steps) muestran vis=85.7% — el modelo puede seguir el target a 0.37 m/s cuando se estabiliza, pero no de forma consistente.

**Próxima iteración (v4.1)**:
- Phase C menos agresiva: vel_max=0.40 m/s (no 0.60), offset_max=0.8m (no 1.2m)
- 10% de episodios Phase A/B intercalados en Phase C para prevenir olvido
- Duración máxima Phase C: 1500 steps (no 3000)

## Comandos de Uso

```bash
# Activar entorno GPU
.vgpu\Scripts\activate

# COMPLETADO: v4 ya está entrenado
# Evaluar checkpoints v4 (resultados: 0% supervivencia global)
python tests/evaluate_hover_track_v4.py --no-display

# Mejor opción actual para tracking real: usar v3.1/400k
# (target estático: 93.3% surv, Easy/Med 100%, Hard 80%)
python tests/test_hover_track_v3_video.py --reward-version v3.1 \
    --model-path ./models/hover_track_v3_1/checkpoints/model_400000_steps.zip

# Test de integración espiral+SAC (funcional con v3.1)
python tests/test_spiral_track_v3_1.py --models 400k --no-display
```
