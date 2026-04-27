# Resumen de Avances: Sistema de Seguimiento Visual de Objetos Móviles (v3.1 → v6)

Este documento resume la evolución técnica del sistema de vuelo autónomo desde la línea base estable (v3.1) hasta la implementación de la arquitectura de recompensa v6 enfocada en supervivencia y estabilidad.

## 1. Evolución del Modelo y Metodología

### Línea Base: v3.1 (Estática)
*   **Estado**: Modelo SAC altamente estable para tareas de hover estático.
*   **Rendimiento**: >93% de supervivencia en entornos sin movimiento.
*   **Limitación**: Incapaz de seguir objetivos con dinámica propia (lemniscata).

### Transición a Dinámica: v4.1 / v4.2
*   **Hito**: Introducción del objetivo móvil (trayectoria de lemniscata de Bernoulli).
*   **Aprendizaje**: Se identificó el fenómeno del *Catastrophic Forgetting* al pasar de fases estáticas a dinámicas.
*   **Corrección (v4.2)**: Se eliminó la penalización por colisión (*crash penalty*) que incentivaba episodios cortos.

### Currículo Geométrico: v5
*   **Innovación**: Implementación de un currículo de inicialización basado en el porcentaje de área del Campo de Visión (FOV).
*   **Diagnóstico Crítico**: A pesar de un seguimiento visual excelente (checkpoint 250k), el modelo v5 sufría de una tasa de supervivencia del 0% debido a un error de sincronización en el motor de físicas y una priorización excesiva del centrado sobre la estabilidad.

### Arquitectura de Supervivencia: v6 (Actualizada)
*   **Hito**: Rediseño aditivo de la función de recompensa y corrección de dos bugs críticos de física (Teletransporte y Altitud).
*   **Enfoque**: Conseguir un vuelo suave y estable a la altura de misión real (`1.39m`) sin los errores de spawn que falseaban las métricas de supervivencia.

---

## 2. Innovaciones Técnicas en v6

### Reajuste del Equilibrio de Recompensa
Se han introducido tres componentes aditivos en el Wrapper de entrenamiento para penalizar la agresividad excesiva:
1.  **Bonus de Estabilidad (+2.0)**: Recompensa directa por mantener el dron nivelado.
2.  **Penalización Extra de Jerk (-1.2)**: Forzado de una política de control suave.
3.  **Penalización de Altitud (-1.0)**: Mantenimiento estricto de la cota de vuelo.

### Correcciones de Motor de Físicas
1.  **Bug de Teletransporte**: Sincronización de `state` y `previous_state` en el motor `solve_ivp`.
2.  **Spawn Altitude Bug (Corregido en v6_fixed)**: 
    - **Fallo**: El objetivo se generaba relativo al dron en el reset, resultando en un spawn a solo 14cm del suelo físico.
    - **Efecto**: Inestabilidad inmediata y supervivencia del 0% en evaluaciones.
    - **Solución**: Fijación del objetivo en el suelo físico (`z=0`) para garantizar un despegue seguro a `1.39m`.

---

## 3. Estado de Entrenamiento y Resultados

### Comparativa de Configuraciones

| Característica | v5 (Anterior) | v6 (Original) | v6_fixed (Actual) |
| :--- | :--- | :--- | :--- |
| **Punto de partida** | v4.1 @ 150k | v5 @ 250k | **v4.1 @ 150k (Base estable)** |
| **Estabilidad** | Gate mult. | Bonus aditivo (+2.0) | **Bonus aditivo (+2.0)** |
| **Altitud Inicio** | Variable | 0.14m (Bug) | **1.39m (Corregida)** |
| **Supervivencia** | 0.0% | 0.0% (Bug spawn) | **En entrenamiento** |

---

## 4. Próximos Pasos Propuestos
1.  **Validación v6_fixed**: Confirmar que con la altura corregida, el agente no solo sigue visualmente al objetivo sino que mantiene el vuelo indefinidamente.
2.  **Ajuste de Ganancias**: Si la penalización de Jerk de la v6 resulta demasiado restrictiva para la agilidad necesaria en Phase C, reducir ligeramente de -1.2 a -0.8.
