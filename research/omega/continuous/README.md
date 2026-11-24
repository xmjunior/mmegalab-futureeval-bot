# OMEGA · Anclas macroeconómicas y distribuciones continuas

CQR con una protección de colas del 3% redujo el CRPS aproximado frente a la regla del campeón en 0,002528 puntos de desempleo. El intervalo del 95% por bloques incluyó cero [−0,001053, +0,007286]. Su intervalo nominal del 80% cubrió el 68,52%, por debajo del 74% mínimo del protocolo. **FAIL para promoción.**

## Resultado fuera de muestra

Se pronosticó la **primera publicación** del desempleo estadounidense a 1, 3 y 6 meses. El origen fue el cierre de cada trimestre. Entrenamiento 200 pares, validación 104, prueba 108, purgados 8 por cruzar cortes de publicación. El test 2016–2024 contiene 36 orígenes y 73 meses objetivo únicos. El estrés de 2020–2021 figura por separado en los resultados. El código repite cada distribución y el CSV detalla los 108 pares retenidos.

| Brazo | CRPS aprox. ↓ | Log-pérdida de densidad ↓ | Brier ↓ | Cobertura 80% | Anchura 80%, pp |
|---|---:|---:|---:|---:|---:|
| Caminata aleatoria gaussiana | 0,667549 | 18,706076 | 0,167476 | 85,19% | 3,333 |
| Empírico sin ajuste | 0,550212 | 6,033127 | 0,084343 | 77,78% | 0,754 |
| Regla del campeón trasladada a esta serie | 0,541289 | 5,556168 | 0,085433 | 62,96% | 0,702 |
| Triangulación 24/60/120 meses | 0,551593 | 5,450860 | 0,075478 | 81,48% | 1,268 |
| AR(1) regularizado | 0,581361 | 25,801356 | 0,199117 | 90,74% | 2,344 |
| CQR y colas Cauchy del 3% | 0,538761 | 0,818961 | 0,087083 | 68,52% | 0,792 |

El campeón operativo de Market Pulse selecciona anchos y fuentes por instrumento. Para esta serie mensual de desempleo, ajena a los concursos, se reprodujo su regla continua de nivel con ancho 1, centro en el último dato, movimientos históricos sin deriva mediana y escala por volatilidad. El muestreo, la frecuencia y la cola extrapolada difieren de los mercados del bot. No se evaluó una predicción enviada por el campeón. [Código de referencia](https://github.com/xmjunior/mmegalab-futureeval-bot/blob/199e336d333f74b47b71a1df607cc430107dab39/market_prices.py)

El CRPS aproximado equivale a dos veces la media de 99 pérdidas de cuantiles en puntos medios equiespaciados. El Brier puntúa superar el último nivel conocido por más de 0,2 puntos porcentuales. Las métricas son específicas del dominio. La densidad y los cuantiles se normalizan en el soporte físico [0,100]. La tasa oficial se redondea a una décima, por lo que el log de densidad y el PIT dependen de la interpolación. Para intervalos comparables en un concurso debe puntuarse su masa y respetarse su tratamiento de extremos. El KS del PIT es descriptivo. La dependencia temporal y el solapamiento de horizontes impiden trasladar al test una garantía IID de cobertura conformal. [Puntuaciones propias](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf) · [Regresión cuantílica conformal](https://proceedings.neurips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html)

## Revisión temporal y contraste con el informe Gemini

El archivo oficial incluye 1.471 filas. Se conservaron sus bytes y se reconstruyó la historia publicada en cada fecha a partir de intervalos de vigencia. Las 250 observaciones de un segundo formato de ALFRED para el vintage del 1 de diciembre de 2010 coincidieron con la reconstrucción. La repetición de la descarga del archivo devolvió el mismo SHA-256. [Archivo ALFRED](https://alfred.stlouisfed.org/series/downloaddata?seid=UNRATE) · [Gráfico oficial de control](https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=UNRATE&vintage_date=2010-12-01&cosd=1990-01-01&coed=2010-11-30)

En los 108 casos de test, al menos una observación histórica había sido revisada. Utilizar los valores disponibles a 31 de diciembre de 2025, manteniendo fijos los meses disponibles y las etiquetas originales, mejoró artificialmente el CRPS en 0,012534, el Brier en 0,000996 y el log de densidad en 1,184554 en esta muestra. Por ejemplo, en marzo de 2016 enero se conocía como 4,9% y la revisión posterior lo dejó en 4,8%. Este diagnóstico aislado demuestra fuga de revisiones en este ejemplo.

El contraste estadístico del experimento 2 supera las 150 observaciones requeridas. La regla del campeón mejora CRPS y Brier puntualmente frente a la caminata aleatoria, pero su intervalo de ganancia de CRPS incluye cero. No hay brazo LLM puro ni híbrido contemporáneo para verificar el umbral del 3%. Sólo se implementó AR(1) regularizado; no se probaron selección AR(p) por AICc ni Holt. Tampoco se puede trasladar este resultado al IPC, nóminas o bonos.

El experimento 5 supera en tamaño los mínimos de 120 casos y 60 de test. CQR vence a los dos análogos estadísticos en CRPS y log de densidad, pero falla el umbral de cobertura. El experimento con cinco cuantiles **elicitados por un LLM** sigue sin medirse. La triangulación reduce el Brier binario y logra 81,48% de cobertura, con un intervalo más ancho y peor CRPS que la regla del campeón. No se promovió ningún modelo.

## Experimento prospectivo independiente

La fuente ALFRED del 23 de noviembre de 2026 carece de octubre de 2025. La política de 120 meses consecutivos produjo una abstención registrada e inmutable. Un protocolo secundario independiente permitió un solo mes interior ausente, sin imputación. Usó únicamente movimientos cuyos dos extremos estaban observados. Se sellaron **6 distribuciones** de dos modelos para noviembre de 2026, enero y abril de 2027, antes de su publicación. Este brazo es exploratorio y se evalúa por separado del backtest.

| Primera publicación programada | Inicio de consultas UTC |
|---|---|
| Noviembre de 2026 | 4 diciembre 2026, 13:30 |
| Enero de 2027 | 5 febrero 2027, 13:30 |
| Abril de 2027 | 7 mayo 2027, 12:30 |

La liquidación espera el primer valor efectivamente archivado por ALFRED, aunque se retrase. Verifica las huellas de sello, protocolo y datos, y conserva resultados ya liquidados. Coste de inferencia y de datos US$0. [Calendario oficial del BLS](https://www.bls.gov/schedule/news_release/empsit.htm)

## Reproducción

Python 3.11+ con NumPy 2.x. Ejecutar desde este directorio. La primera orden verifica las huellas de los archivos originales. El cálculo histórico y sus pruebas se ejecutan sin red. La regeneración usa los datos congelados del repositorio. La descarga opcional contrasta sus bytes con ALFRED. La liquidación sólo usa la red después de cada fecha programada y omite objetivos ya puntuados.

```sh
sha256sum -c SHA256SUMS
python -m unittest discover -p 'test_continuous.py' -v
python evaluate.py
python download.py
python settle.py --state /ruta/estado/omega-continuous.json
```

El compromiso público fija las reglas de decisión antes de evaluar los brazos. Los resultados por horizonte y régimen, cobertura 50/80, PIT, intervalos por bloques y efectos de revisión se encuentran en `results.json`. `test_pairs.csv` contiene origen, observación disponible, horizonte, mes objetivo, fecha de primera publicación, etiqueta y pérdidas CRPS pareadas. La tabla de resultados excluye el diagnóstico contaminado. [Prerregistro inmutable](https://github.com/xmjunior/mmegalab-futureeval-bot/commit/6a053aa7bf6c1335a172f87242648e28d26f94d8)
