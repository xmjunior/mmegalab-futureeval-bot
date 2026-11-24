# ForecastBench · Rondas internas de MEGALAB

Se emitirá una referencia estadística propia en las nueve fechas quincenales del calendario desde el 6 de diciembre de 2026 al 28 de marzo de 2027. En la primera ronda futura está pendiente la publicación de las preguntas oficiales. La emisión, de solo CPU, se guarda en Git. La ejecución requiere acceso de lectura a la API pública de GitHub. El sistema conserva la primera versión observada, su revisión, la hora UTC de emisión y hashes del conjunto original, del modelo y de las predicciones. Una ronda perdida queda sin pronóstico. La inscripción, recepción oficial y puntuación del equipo dependen del organizador. No hay llamadas a modelos ni a un servicio oficial de entrega.

### Resultado exploratorio de la referencia

El ajuste usa exclusivamente 1.311 resultados de cuatro rondas de entrenamiento disponibles en su corte histórico. Cada serie o familia aporta un voto medio en cada ronda. Se regulariza por fuente y horizonte con cuatro pseudorrondas. Los horizontes nuevos usan la media regularizada de la fuente. Las fuentes sin entrenamiento, incluidas las de mercado, reciben 0,5. El panel del experimento binario y sus resultados agregados ya se habían examinado. Esta comparación post hoc reutiliza las mismas 5.723 observaciones y carece de un segundo holdout independiente.

| Bloque | Eventos | Rondas | Brier | Referencia 0,5 | Log Loss | Brier equilibrado por familia y ronda |
|---|---:|---:|---:|---:|---:|---:|
| Validación | 993 | 3 | 0,241305 | 0,250000 | 0,675588 | 0,243485 |
| Prueba histórica | 5.723 | 6 | 0,234188 | 0,250000 | 0,660701 | 0,231779 |

Las 5.723 observaciones finales proceden exclusivamente de cinco fuentes de series de datos. El entrenamiento incluía solo cuatro (DBnomics, FRED, Wikipedia y Yahoo Finance). ACLED y todas las plataformas de mercado reciben por tanto 0,5. La cobertura describe pronósticos presentes, incluso cuando usan 0,5. El rendimiento en mercados sigue sin validar.

En la misma prueba GPT-5 mini archivado obtuvo Brier 0,154708. El modelo estadístico queda como control en sombra. En preguntas de mercado su mejora esperada frente al comparador por construcción es cero. En los datos históricos de prueba el ECE de diez intervalos fue 0,10825. Para cada nueva ronda se registra la cobertura de ambos grupos y cuántas filas emplearon cada alternativa. Son resultados temporales correlacionados. Los recuentos de varias rondas no deben sumarse para inferir precisión estadística. Véanse los desgloses, los parámetros y los hashes en [results.json](results.json) y las condiciones en [protocol.md](protocol.md).

### Reproducir y ejecutar

Desde la raíz del repositorio, con Python 3.12 y sin paquetes adicionales:

```sh
python -m unittest discover -s research/omega/forecastbench_shadow -p 'test_*.py' -v
python research/omega/forecastbench_shadow/model.py
python research/omega/forecastbench_shadow/shadow.py run --state-dir records/forecastbench --receipts-dir records/forecastbench/receipts
```

La última orden obtiene la hora UTC real, emite solo el día oficial vigente y puntúa las emisiones existentes con recibo. El registro `YYYY-MM-DD.json` es inmutable. La puntuación requiere un recibo emitido por el flujo de GitHub antes del cierre, con `issue_sha256` y `published_at_utc` coincidentes. En `scores/YYYY-MM-DD.json` se mantienen las primeras resoluciones binarias elegibles, sus revisiones públicas, cobertura y métricas. Los resultados conocidos el propio día de emisión quedan excluidos. Cada pregunta de mercado cuenta una vez y cada pregunta de datos una vez por fecha dentro de su ronda. Se informan también medias equilibradas por serie y fuente. El último control local está autorizado hasta el 31 de marzo de 2028. Los horizontes posteriores quedan pendientes. El archivo de pronósticos para un exportador compatible está en el campo `forecast_set` de cada sello. La inscripción oficial requiere una cuenta y una ruta de carga facilitadas por ForecastBench. [Instrucciones del organizador](https://github.com/forecastingresearch/forecastbench/wiki/How-to-submit-to-ForecastBench).

Los ensayos del emisor obtienen los 500 identificadores y horizontes de una ronda histórica archivada y desplazan únicamente las fechas en memoria para probar cierres de día, autenticidad del recibo y repetición del cron. Los registros de prueba se crean en directorios temporales y se eliminan. El entrenamiento proviene de las revisiones y la licencia CC BY-SA 4.0 documentadas en [el archivo de origen](../binary/inputs.json).
