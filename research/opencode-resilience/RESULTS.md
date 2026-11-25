# research/opencode-resilience — recuperación tras caída total de proveedores gratuitos

Base: `main` @ `ad21f59`. Fase 1: detección (R1–R3). Fase 2: corrección de D1/D2 + regresiones (R4–R7).
Sin llamadas a proveedores, sin gastos, sin merge ni despliegue.

## Resultado

```
uv run python -m unittest discover -s tests -v
Ran 37 tests ... OK
```

| # | Prueba | Resultado |
|---|--------|-----------|
| R1 | `test_simultaneous_outage_recovers_once_cooldowns_expire` | PASS |
| R2 | `test_daily_quota_on_every_route_is_not_retried_as_transient` | PASS (D1 corregido) |
| R3 | `test_hosted_routes_are_probed_again_after_cpu_fallback` | PASS (D2 corregido) |
| R4 | `test_transient_429_keeps_short_retry_after_and_second_turn` | PASS (regresión D1) |
| R5 | `test_daily_quota_respects_longer_retry_after` | PASS (regresión D1) |
| R6 | `test_choose_integers_returns_to_hosted_after_cpu_fallback` | PASS (regresión D2) |
| R7 | `test_groq_daily_cap_is_a_boolean_signal_without_provider_text` | PASS (regresión D1) |

R2 y R3 ya no llevan `expectedFailure`. Evidencia cruda: `research/opencode-resilience/unittest.log`.

## Correcciones

### D1 — Cuota diaria reconocida en todos los proveedores (`main.py`)

- `daily_quota_exhausted(exc)`: solo un `429` cuenta, y solo si el proveedor lo declara
  (`exc.daily` o texto que case `daily|per-day|day limit|requests per day|rpd`).
- Al detectarse: se consumen los intentos de esa ruta (`attempts = limit`) para no gastar
  reintentos inútiles y el cooldown usa el `Retry-After` explícito como tiempo de reset; si no
  hay hint, se asume ventana completa (≥ 3600 s).
- Un `429` transitorio (p. ej. "retry in 20s") conserva su hint corto y su segundo intento: no se
  confunde con cuota diaria (R4).
- Groq no propaga texto de proveedor: `groq_free.Unavailable` incorpora una señal booleana
  `daily` calculada desde el cuerpo del error, sin exponer mensajes (R7).

### D2 — Retorno a rutas hospedadas tras el cooldown (`main.py`)

- `self.cpu_retry_at` fija el instante en que una instancia degradada puede volver a probar rutas
  hospedadas (mínimo cooldown futuro, o inmediato si no hay cooldowns).
- `invoke`: si existe CPU pero el cooldown ya expiró, prueba una vez cada ruta lista (sin dormir) y
  vuelve a hospedado si responde; si sigue caído, mantiene CPU y reprograma la sonda.
- `choose_integers`: aplica la misma puerta y descarta el texto libre de CPU como respuesta
  estructurada, así que un run degradado recupera el JSON de las rutas gratuitas (R6).
- CPU sigue desactivada por defecto (`ALLOW_CPU_FALLBACK=false`): solo se usa con flag explícito y
  Linux x86_64, y deja de ser permanente.

## Controles conservados

- Deadline de 180 s y dos intentos por ruta en modo normal; sonda degradada acotada a un intento por
  ruta lista y sin esperas.
- Mensajes de error existentes intactos (`refusing degraded`, `CPU fallback is disabled`,
  `requires Linux x86_64`).
- Sin cambios en publicación de premios, competiciones, credenciales ni workflows.
- Suite completa en verde: 37/37.

## Reproducción

```
cd MEGALAB-RESEARCH
uv sync --frozen
METACULUS_TOKEN=test uv run python -m unittest discover -s tests -v
```
