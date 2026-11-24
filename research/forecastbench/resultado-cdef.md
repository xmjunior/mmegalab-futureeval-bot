# ForecastBench · Prueba temporal de C, D, E y F

Se evaluaron 2.749 pronósticos coincidentes de Gemini 3.5 Flash, GPT-5.4-mini y Claude Opus 4.7, correspondientes a cinco rondas entre el 2 de agosto y el 27 de septiembre de 2026. La comparación anterior terminaba el 19 de julio. El inventario de nombres de archivo determinó las versiones con cobertura antes de escribir el diseño y calcular resultados. Los archivos contienen la fecha de la ronda, anterior a la resolución en todos los casos. El conjunto procesado no conserva la hora individual de emisión. Se excluyó Metaculus. Coste de inferencia US$0.

| Regla fija | Comparador | Brier | Ganancia Brier [IC95% por ronda] | Ganancia log-loss | Rondas con mejor Brier |
|---|---|---:|---|---:|---:|
| C, contracción 0,9p + 0,05 | Gemini | 0,18403 | +0,00327 [0,00184; 0,00595] | +0,03300 | 5/5 |
| D, media 50/50 Gemini y GPT-mini | Gemini | 0,18633 | +0,00097 [−0,00122; 0,00333] | +0,02569 | 3/5 |
| E, usar Opus si la diferencia es ≥0,20 | D | 0,18501 | +0,00133 [−0,00280; 0,00639] | +0,00074 | 2/5 |
| F, precio del mercado el día anterior si procede | Gemini | 0,16822 | +0,01908 [0,01270; 0,02462] | +0,07612 | 5/5 |

El Brier individual fue 0,18730 para Gemini, 0,20235 para GPT-mini y 0,18800 para Opus. La correlación de errores firmados entre pares osciló entre 0,920 y 0,953. Entre los 3.371 casos elegibles, GPT-mini tuvo 622 probabilidades ausentes o imputadas (18,45%) y los otros dos modelos cubrieron los 3.371. E necesitó a Opus en 485 de los 2.749 casos comunes (17,64%). Su probabilidad se había emitido de manera independiente y sin acceso a los argumentos ni a las estimaciones de los otros dos modelos. D y E ofrecen una prueba de sus respectivas reglas, sin evidencia suficiente para dar por rentable combinar modelos de pago.

F usó 257 precios de Polymarket y 113 de Manifold. En esas 370 preguntas el Brier fue 0,12682 para el mercado y 0,26856 para Gemini. Las 115 preguntas de Kalshi y las 2.264 de datos conservaron Gemini según la regla fijada. Se trata del precio histórico del propio contrato en ForecastBench. Para trasladar ese resultado a FutureEval se necesitaría una fuente permitida con equivalencia comprobada de evento, fecha y resolución. En las preguntas de datos, C ganó 0,00057 de Brier pero empeoró el log-loss en 0,00626; D empeoró tanto Brier como log-loss. Esto limita cualquier extrapolación a preguntas genéricas.

Los IC se obtuvieron mediante los 3.125 remuestreos posibles de cinco rondas. Los intervalos Brier ajustados con Bonferroni también excluyeron cero para C y F. Una prueba bilateral exacta de signos por ronda, más conservadora con cinco bloques, arroja p ajustada por Holm de 0,25 para C y F, 1 para D y E. Los bloques comparten series y parte de las preguntas. Al excluir objetivos ya presentes hasta julio y conservar solo la primera observación por objetivo en el holdout, quedaron 2.579. Las ganancias Brier fueron +0,00225 para C, +0,00078 para D, +0,00141 para E y +0,01003 para F. Las bandas de D y E siguen incluyendo cero. Estas cifras corresponden a variantes zero-shot externas y preguntas binarias. El experimento no evaluó el bot, preguntas numéricas ni una configuración desplegable a US$0.

## Reproducción

Con Python 3 y NumPy, colocar juntos `compare.py`, `compare_cdef.py` y `design-cdef.md`. Utilizar el mismo tar procesado y los diez JSON de resoluciones que enumera `paired-cdef.json`, incluido el período anterior para comprobar objetivos repetidos.

```sh
python compare_cdef.py processed_forecast_sets.tar.gz directorio_de_resoluciones paired-cdef.json
```

El diseño congelado tiene SHA-256 `d77fc35aa2acfd38d735c1d188d78e2f2ef25ee1351f9fdd21b59c59192287f4`. El JSON registra los hashes del script, el tar, los quince archivos individuales de pronósticos y los diez archivos de resoluciones. También presenta todas las coberturas, fuentes, rondas, recuentos, resultados y sensibilidad. La comparación usa Brier medio pareado por fila distinto del índice oficial ajustado de ForecastBench.

Datos y pronósticos © Forecasting Research Institute, ForecastBench, CC BY-SA 4.0. Comparaciones derivadas bajo la misma licencia. [ForecastBench Datasets](https://www.forecastbench.org/datasets/) · [Tar de pronósticos](https://www.forecastbench.org/assets/data/processed-forecast-sets/processed_forecast_sets.tar.gz) · [Resoluciones congeladas](https://github.com/forecastingresearch/forecastbench-datasets/tree/86b92ba06ed34d606d633bff3e99af54dd447d3a/datasets/resolution_sets) · [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
