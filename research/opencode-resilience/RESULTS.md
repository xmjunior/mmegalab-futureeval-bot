# research/opencode-resilience — recuperación tras caída total de proveedores gratuitos

Fecha: 2026-11-25. Base: `main` @ `ad21f59`. Sin cambios en producción. Sin llamadas a proveedores.

## Alcance

Detectar fallos de recuperación cuando **todos** los proveedores gratuitos dejan de responder.
Código nuevo: `tests/test_resilience_recovery.py` (3 pruebas, mocks `AsyncMock` + reloj monotónico sintético).
No se repiten `tests/test_route_cooldowns.py` ni `tests/test_groq_free.py`; no se tocan.

## Resultado global

```
uv run python -m unittest discover -s tests -v
Ran 33 tests ... OK (expected failures=2)
```

| # | Prueba | Resultado | Lectura |
|---|--------|-----------|---------|
| R1 | `test_simultaneous_outage_recovers_once_cooldowns_expire` | **PASS** | Caída total (503 en las 3 rutas) agota 2 intentos/ruta, lanza `RuntimeError: refusing degraded` y, pasados los cooldowns, la misma instancia vuelve a responder (`recovered`) durmiendo solo el tiempo restante (40 s). |
| R2 | `test_daily_quota_on_every_route_is_not_retried_as_transient` | **FAIL (defecto D1)** | Esperado 1 intento/ruta y cooldown ≥ 3600 s; real `[2, 2, 1]`. |
| R3 | `test_hosted_routes_are_probed_again_after_cpu_fallback` | **FAIL (defecto D2)** | Esperado volver a rutas hospedadas; real `'cpu-local'` con 0 llamadas nuevas a rutas. |

R2 y R3 están marcadas `@unittest.expectedFailure`: documentan el defecto sin romper la suite y avisarán
("unexpected success") cuando se corrija.

## Defectos detectados

### D1 — Agotamiento diario no reconocido fuera de OpenRouter (alto)

`main.py:156-158` solo trata como cuota diaria `route.model.startswith("openrouter/")`:

```python
if not retryable or (route.model.startswith("openrouter/")
                     and status == 429 and re.search(r"day|daily", str(exc), re.I)):
    delay = max(delay, 3600)
```

Un `429` con mensaje "daily quota exceeded" y sin `retry-after` en Gemini/Groq cae en
`delay = 12 * attempts`, por lo que:

- se consumen los 2 intentos de la ruta contra una cuota ya agotada (`[2, 2, 1]`);
- el cooldown queda en 36 s en vez de ≥ 3600 s, así que cada nueva llamada vuelve a golpear
  la misma ruta agotada y la clasifica como fallo transitorio;
- el diagnóstico final (`refusing degraded`) no distingue cuota diaria de caída de red.

Impacto: reintentos inútiles y presión innecesaria sobre proveedores gratuitos justo cuando la
cuota ya no se recuperará en la ventana de la ejecución.

### D2 — La degradación a CPU es permanente (alto)

`main.py:110-113` cortocircuita toda invocación posterior cuando `self.cpu` ya existe:

```python
if self.cpu is not None:
    ...
    return await asyncio.to_thread(self.cpu.invoke, prompt, system_prompt)
```

Tras una caída total con `ALLOW_CPU_FALLBACK=true`, aunque los proveedores gratuitos se recuperen
en segundos, el bot sigue respondiendo con el modelo local pequeño durante el resto del proceso
(0 llamadas a rutas en R3). No hay sonda, TTL ni reintento de rutas hospedadas, y `available_at`
queda congelado sin uso.

Impacto: calidad de pronóstico degradada de forma silenciosa y no reversible; el `to_dict()`
seguirá anunciando `cpu_fallback_enabled` sin reflejar que la sesión ya está atrapada en CPU.

### Observación menor (no cubierta por prueba)

`main.py:143-159`: los estados no reintentables `401/402/403/404` reciben `delay = 3600`, es decir,
un 403 transitorio (WAF/edge) desactiva la ruta una hora. No se prueba aquí por límite de 3 pruebas.

## Reproducción

```
cd MEGALAB-RESEARCH
uv sync --frozen
METACULUS_TOKEN=test uv run python -m unittest discover -s tests -v
METACULUS_TOKEN=test uv run python -m unittest tests.test_resilience_recovery -v
```

Evidencia cruda: `research/opencode-resilience/unittest.log`.
Pruebas sin `expectedFailure` (fallo explícito): `[2, 2, 1] != [1, 1, 1]` y `'cpu-local' != 'hosted-recovered'`.

## Garantías

- Cero llamadas a APIs de proveedores, cero gastos: todas las rutas son `AsyncMock`; el reloj es sintético.
- Sin modificar `main.py`, `groq_free.py` ni pruebas existentes.
- Sin consultar competiciones abiertas ni credenciales.
