# ForecastBench · Comparación externa al 24-11-2026

Se usaron predicciones ya generadas por ForecastBench en cinco rondas, del 24 de mayo al 19 de julio de 2026. Se seleccionaron las variantes zero-shot de Gemini 3.5 Flash y Gemma 4 31B-it sin datos de congelación incorporados, y el baseline oficial denominado Naive Forecaster. Solo se puntuaron resoluciones binarias fechadas después de la fecha oficial de pronóstico y antes del corte de la auditoría. Los datos procesados no publican el instante individual de cada solicitud. La comparación es retrospectiva sobre pronósticos originalmente prospectivos. No se llamó a ningún modelo.

| Muestra | Filas | Brier de Gemini 3.5 | Brier de Gemma | Brier del comparador | Log-loss de Gemini | Log-loss de Gemma | Log-loss del comparador |
|---|---:|---:|---:|---:|---:|---:|---:|
| Preguntas generadas con conjuntos de datos; baseline oficial | 2.758 | 0,17205 | 0,19404 | 0,29134 | 0,49063 | 0,55208 | 0,89178 |
| Preguntas de otros mercados; precio del día anterior | 438 | 0,25950 | 0,25661 | 0,10037 | 0,91203 | 0,81879 | 0,32399 |

En la primera muestra, Gemini mejora el Brier en 0,02200 frente a Gemma, IC95% descriptivo por bloques de fecha [0,00979; 0,03397], y la pérdida logarítmica en 0,06145 [0,03237; 0,08870]. Los cinco bloques mejoran en ambas métricas. Las 2.758 filas corresponden a 671 identificadores de fuente y pregunta. Hay cinco rondas y preguntas repetidas entre ellas, lo que limita la precisión del intervalo. En la segunda muestra, Gemini empeora frente a Gemma en pérdida logarítmica en las cinco rondas; la estimación previa del mercado superó a ambos. No se utilizó el sitio ni la API de Metaculus, y las filas de origen Metaculus se excluyeron.

Gemini devolvió probabilidad propia en 4.145 de las 4.146 observaciones elegibles, frente a 3.196 para Gemma. En la muestra principal se excluyeron los pronósticos imputados de ambos modelos y del baseline. Esta comparación usa las predicciones de ForecastBench, no el prompt ni el pipeline de Mario. El nivel gratuito de Google permite probar el candidato únicamente si se comprueba que el proyecto está en esa modalidad. La documentación de Google distingue entre tarifas gratuitas y de pago. No se habilitó facturación ni se modificó ningún bot.

## Reproducir

Descargar el tar procesado desde la página de datasets y los cinco JSON de resoluciones del commit congelado, con nombres como 2026-05-24_resolution_set.json. Las fechas, el SHA del tar y los SHA-256 de todos los archivos de entrada usados están en paired.json. Con Python y NumPy, ejecutar:

```sh
python compare.py processed_forecast_sets.tar.gz directorio_con_resoluciones resultado.json
```

Las salidas contienen el detalle por fecha, coberturas y exclusiones, y los intervalos percentiles al remuestrear fechas enteras (5⁵ = 3.125 remuestras). El Brier se calcula sobre la probabilidad original y el log-loss acota la probabilidad en [10⁻⁶, 1−10⁻⁶]. Es un contraste pareado con igual peso por fila distinto del índice oficial ajustado por dificultad.

Pronósticos y resoluciones originales: Forecasting Research Institute, ForecastBench Datasets, © 2024–2026, CC BY-SA 4.0. Adaptación de métricas y comparación, bajo la misma licencia. [Conjuntos de datos](https://www.forecastbench.org/datasets/) · [Resoluciones congeladas](https://github.com/forecastingresearch/forecastbench-datasets/tree/86b92ba06ed34d606d633bff3e99af54dd447d3a/datasets/resolution_sets) · [Archivo de pronósticos procesados](https://www.forecastbench.org/assets/data/processed-forecast-sets/processed_forecast_sets.tar.gz) · [Licencia](https://creativecommons.org/licenses/by-sa/4.0/)
