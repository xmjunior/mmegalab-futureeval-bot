# ForecastBench · Clase de referencia por serie

El candidato usa resultados de la misma serie conocidos al entrenar y los combina con la referencia por fuente y horizonte. La ejecución no requirió inferencia. Las tres columnas de puntuación corresponden a las mismas 2.914 observaciones cerradas, con cobertura del 100 % de ese panel.

| Pronóstico | Brier | Pérdida logarítmica | ECE10 |
|---|---:|---:|---:|
| Clase de referencia por serie | 0,23799 | 0,66795 | 0,10862 |
| Referencia propia vigente de ForecastBench en sombra | 0,24174 | 0,67630 | 0,11031 |
| GPT-5 mini archivado por ForecastBench | 0,14612 | 0,42475 | 0,02978 |
| Probabilidad constante de 0,5 | 0,25000 | 0,69315 | 0,03672 |

**Criterio de promoción: FAIL.** El candidato mejoró el Brier en 0,00375 por evento. Al dar el mismo peso a cada ronda, el margen cayó a 0,00033, con intervalo del 95 % de −0,00280 a 0,00294 y p exacta unilateral de 0,40625. Cuatro rondas mejoraron, una empeoró y cinco empataron. Las primeras cuatro rondas aportaron 2.851 de los 2.914 eventos. Nuestras 1.558 observaciones de series conocidas fueron las únicas donde podía cambiar el pronóstico. El competidor externo preseleccionado fue más preciso que los dos métodos CPU.

El [registro previo](https://github.com/xmjunior/mmegalab-futureeval-bot/commit/35fffcdb115e3828a17f131809cb4ffd52327feb) data de antes de la extracción. Se excluyeron todos los eventos ya puntuados por el panel anterior (8.027 identidades) y las 9.983 identidades ofertadas en las cinco emisiones anteriores de entrenamiento. El inventario de exclusión para estas diez rondas figura en inputs.json. Tras excluir fechas fuera de ventana y eventos ya conocidos, 335 observaciones carecían de una resolución binaria elegible. Las métricas de los 584 grupos conservadores de series y familias complementan la comparación por fecha. Los resultados corresponden a fuentes de datos de ForecastBench y no acreditan una mejora en FutureEval.

La [ejecución de verificación](https://github.com/xmjunior/mmegalab-futureeval-bot/actions/runs/45627540870) reprodujo byte a byte la evaluación primaria y comprobó los SHA-256 de los diez archivos del modelo GPT-5 mini contra el manifiesto previo. Cada una de las 2.914 predicciones externas se unió por fecha de emisión, fuente, identificador y fecha objetivo, con dirección simple y sin imputación. El [artefacto de GitHub](https://github.com/xmjunior/mmegalab-futureeval-bot/actions/runs/45627540870/artifacts/15029496254) tiene SHA-256 f6eb54c36fd022fa88d55bc116dd2ed1fd6a0265b8bcde2d1bef92d84fd3f0a6.

Desde este directorio, con Python y NumPy, se puede reproducir la comparación completa sin red.

```bash
python -m unittest -v test_series_prior
python evaluate.py --external <(gzip -dc gpt5mini.json.gz) --out /tmp/series-comparison.json
cmp /tmp/series-comparison.json comparison.json
```

Los datos originales de ForecastBench y las transformaciones se distribuyen bajo CC BY-SA 4.0. [Datos del organizador](https://www.forecastbench.org/datasets/) · [Licencia](https://creativecommons.org/licenses/by-sa/4.0/).
