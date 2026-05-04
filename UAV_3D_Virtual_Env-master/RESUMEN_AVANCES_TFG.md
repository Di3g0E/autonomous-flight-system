# Resumen de Avances TFG — Estado Actual y Petición de Orientación

**Fecha**: 28 de abril de 2026
**Estado del proyecto**: Bloqueado en un techo de rendimiento que no consigo superar tras múltiples iteraciones.

---

He invertido muchas horas de trabajo, probando arquitecturas distintas, y aunque hay avances claros en algunas partes, **no he conseguido el objetivo principal de vuelo continuo sostenido**. Necesito tu orientación para decidir cómo cerrar el TFG.

---

## 1. Lo que SÍ funciona bien

- **El dron ve la esfera correctamente**. El detector visual identifica el objetivo en el 95–100 % de los frames durante el vuelo. Esta era una preocupación importante y está resuelta.
- **El dron se inicializa bien**. Spawn determinista sobre la esfera con la geometría correcta.
- **El sistema de tracking visual funciona**. El dron centra la esfera en el campo de visión durante los segundos que vuela.
- **El pipeline completo está montado**: entorno de simulación, entrenamiento RL (SAC), evaluaciones automáticas, generación de vídeos, registro de telemetría detallada.

---

## 2. Lo que NO he conseguido

**El dron se mantiene en el aire solo ~2 segundos** antes de perder el control. El objetivo era 30 segundos de vuelo continuo. Después de ~2 segundos, el dron pierde estabilidad (se inclina demasiado, sale de la zona de vuelo, o se cae).

Esto se traduce en **0 % de supervivencia** en todas las evaluaciones, en TODAS las versiones que he probado.

---

## 3. Lo que he probado (resumen simple)

He hecho 9 versiones del sistema entre v6 y la actual. Cada una intentaba arreglar algo distinto:

| Versión | Cambio principal | Resultado |
|---|---|---|
| v7.0–7.2 | Reward survival-first, currículo de 4 fases | Bug crítico no detectado: el dron entrenó "a ciegas" |
| **v7.3** | **Detecté y arreglé el bug crítico** | El dron empezó a ver la esfera de verdad por primera vez |
| v7.4 | Bonus por sobrevivir más, freno motor | Sigue 0 % supervivencia |
| v7.5 | Restricciones de altitud asimétricas | Sigue 0 % supervivencia (diagnóstico inicial fue erróneo) |
| v8 | **Simplifiqué TODO**: reward minimal | Sigue 0 % supervivencia |
| v8_short | Reduje el horizonte (30s → 5s) | **Sigue 0 % supervivencia** |

### El bug crítico (v7.3) merece mención especial

Durante varias sesiones, mi código leía mal una variable del entorno. El dron VEÍA la esfera (los vídeos lo confirman), pero la información llegaba mal a la función de recompensa. Resultado: durante 200.000 pasos de entrenamiento (5 horas de GPU acumuladas en v7.0–v7.2), el agente entrenaba en un mundo donde "no veía" la esfera para efectos de aprendizaje. **Encontrarlo y arreglarlo fue un avance importante**, pero también significó que parte del trabajo previo no contaba.

---

## 4. Mi preocupación

He cambiado el reward en muchas direcciones (más complejo, más simple, distintos pesos, distintas penalizaciones). He cambiado el horizonte temporal. He simplificado el spawn. He añadido y quitado mecanismos.

**Todo termina en el mismo punto: ~2 segundos de vuelo y caída**.

Esto me hace pensar que:

1. El problema **no es la función de recompensa** (lo he probado complejo y simple, mismo resultado).
2. El problema **no es el horizonte** (con 5 segundos como objetivo en lugar de 30, mismo resultado).
3. El problema **probablemente está en algo más profundo**: la combinación del algoritmo SAC con la dinámica del simulador y el número de pasos de entrenamiento (200.000) parece insuficiente para vuelo sostenido.

No sé si esto es algo que se puede resolver con más tiempo de cómputo (otras versiones anteriores mucho más largas no han funcionado), con un cambio de algoritmo, o si es una limitación intrínseca del setup que no detecté antes. Por eso te escribo.

---

## 5. Lo que tengo claro y lo que no

**Tengo claro**:
- El sistema de visión funciona.
- El sistema de tracking visual durante vuelo funciona durante ~2 segundos.
- He documentado cuantitativamente cada intento, con métricas y vídeos.

**No tengo claro**:
- Si invertir más tiempo en una v9 con cambios estructurales (filtro de paso bajo en acciones, entrenamiento de 1 millón de pasos, otro algoritmo como TQC) tendría éxito.
- Si tiene más sentido cerrar el TFG con lo que hay y documentar honestamente el techo encontrado, presentándolo como un resultado caracterizado.
- Cómo enmarcar académicamente este resultado parcial.

---

## 6. Tres opciones que veo

### Opción A — Cerrar el TFG con lo actual
Defender que el sistema logra **tracking visual robusto durante ventanas cortas (~2 s)** y cooperaría con el módulo de búsqueda en espiral en despliegue real (que ya está entrenado y funciona). El resultado es honesto, está documentado, y el módulo es funcional para su rol específico.

### Opción B — Una iteración más con cambios estructurales
Probar una v9 atacando la dinámica directamente (no el reward): filtro de suavizado de acciones, entrenamiento de 500k–1M pasos. Coste estimado: 1–2 sesiones de trabajo y ~10–15 h de GPU.

U otra opción v9 sería intentar replicar un siguelíneas pero en 3D, es decir, que el dron siga la "línea" de la esfera. Esto sería cambiar el entrenamiento y la función de recompensa. No tendría en cuenta la velocidad de la esfera, solo la posición. No se si sería una buena opción, pero es una opción.

¿Qué dirección tiene más sentido dado el calendario del TFG? 

Gracias por tu tiempo.
