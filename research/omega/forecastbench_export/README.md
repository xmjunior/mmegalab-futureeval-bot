# ForecastBench · Exportación y prueba histórica sin envío

`export.py` transforma predicciones ya producidas a los cinco campos del JSON de ForecastBench y comprueba la cobertura antes de guardarlo en un archivo local. Usa Python estándar. La muestra incluida procede de la ronda cerrada para envíos del 26 de octubre de 2025. El productor del archivo era **ForecastBench**, con el modelo externo GPT-5-Mini-2025-08-07 de OpenAI. El exportador conserva esa autoría. Las muestras excluyen registros sin resolver, valores imputados y fechas posteriores al 24 de noviembre de 2026. Contienen identificadores y números ya publicados. No incluyen preguntas, texto de resolución, resultados ni señales actuales. No se contactó al organizador ni se envió un archivo.

| Prueba ejecutada | Preguntas de mercado completas | Preguntas de datos completas | Fechas de datos presentes | Decisión local |
|---|---:|---:|---:|---|
| Índice oficial completo de la ronda frente a las salidas archivadas, resueltas y sin imputar | 117/250 | 0/250 | 1.220/1.996 | Rechazo correcto. Cada categoría necesitaría 238 preguntas completas. |
| Muestra de integración de 20 preguntas de mercado y 20 de datos, con un horizonte ya resuelto por pregunta de datos | 20/20 | 20/20 | 20/20 | Aprobada. El identificador comienza por `HISTORICAL-FIXTURE-ONLY`. |
| Misma muestra tras omitir una pregunta por categoría | 19/20 | 19/20 | 19/20 | Aprobada exactamente en el umbral del 95 %. |
| Tras omitir una segunda pregunta de datos | 19/20 | 18/20 | 18/20 | Rechazo correcto, aunque el grupo de mercado sí cumple. |

La validación también rechaza un índice recortado que conserve la identidad oficial de una ronda estándar de 250 preguntas por grupo. La muestra artificial de 40 preguntas comprueba exclusivamente el formato y la cobertura individual. Las pruebas cotejan sus 40 probabilidades con el archivo original y reproducen los recuentos de `results.json`.

El archivo procesado del organizador contiene únicamente parte de los horizontes de la ronda. Este corte se usa para demostrar la detección de registros faltantes. No permite juzgar la cobertura de la entrega original de ForecastBench ni representa la participación de MEGALAB. El criterio local exige todos los horizontes solicitados para declarar completa una pregunta de datos. Esta interpretación conservadora de la cobertura por pregunta se declara aparte del número de registros entregados. La aceptación oficial corresponde al organizador. [Instrucciones de envío](https://github.com/forecastingresearch/forecastbench/wiki/How-to-submit-to-ForecastBench).

## Ejecución

Desde esta carpeta, las siguientes órdenes trabajan sin red. La segunda sale con código 1 porque el subconjunto resuelto es insuficiente para el índice completo. El código 2 identifica entradas inválidas, como una fecha ajena o un duplicado. El programa nunca sobrescribe un archivo de salida.

```sh
python -m unittest discover -v
python export.py round-index.json.gz resolved-archive.json.gz
python -c "import json; from export import build,read_json; d=read_json('sample.json'); data,counts=build(d['questions'],d['predictions']); print(json.dumps(counts)); print(data['organization'],data['model'],len(data['forecasts']))"
```

Para un futuro conjunto propio, el primer argumento admite el JSON original de preguntas del organizador. El segundo requiere `organization`, `model`, `model_organization`, `question_set` y un array `forecasts` con los campos `id`, `source`, `forecast`, `resolution_date` y, opcionalmente, `reasoning`. Para preguntas de mercado `resolution_date` es `null`. Para las de datos se emite una fila por fecha solicitada en el índice de esa ronda, sin suponer que todas tengan ocho. El programa valida la identidad de la pregunta y de la ronda, conserva los ceros y unos auténticos y rechaza duplicados, valores no numéricos o infinitos y fechas inválidas. Si los dos grupos alcanzan el umbral, `--output` escribe el JSON preparado para revisión. Para entregas futuras se emplea el nombre oficial `FECHA.ORGANIZACION.N.json` y se siguen las reglas del organizador. El exportador local no se conecta a GCP.

`prepare_snapshot.py` permite reproducir los dos archivos comprimidos a partir del conjunto de preguntas y el miembro oficial del archivo tar especificados en `sources.json`. Comprueba los dos SHA-256 antes de transformar los datos. `sample.json` es una proyección inequívocamente rotulada que modifica el identificador del conjunto para impedir confundirla con la ronda oficial. `results.json` conserva los recuentos. Datos derivados bajo [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) de [Forecasting Research Institute](https://www.forecastbench.org/datasets/).

## Alta del equipo

El equipo se identifica mediante Google en el registro del organizador. ForecastBench revisa el alta y envía por correo la carpeta de carga. La página oficial no acredita por sí misma el alta de MEGALAB. El siguiente día de ronda por calendario es el 6 de diciembre de 2026 y el horario oficial de carga termina a las 23:59:59 UTC. [Registro oficial](https://register.forecastbench.org/) · [Normas de la ronda](https://github.com/forecastingresearch/forecastbench/wiki/How-to-submit-to-ForecastBench).
