# OpenAI en predicciones prospectivas archivadas

**Decisión del 24 de noviembre de 2026.** GPT-5.6 Sol es el primer candidato de OpenAI para una prueba sombra si llegara a financiarse. Obtiene la señal multirronda más favorable en el archivo y es el primer modelo OpenAI en FutureEval. Con cinco fechas por cohorte, el menor valor p unilateral exacto posible es 1/32. La corrección Holm para ocho comparaciones exige 0,00625 en el primer paso. Este archivo tiene muy pocas fechas para confirmar estadísticamente la promoción bajo el criterio fijado. La evidencia permite elegir qué probar, pero no acredita una mejora del bot de Mario. El cálculo consumió US$0 y realizó cero inferencias.

## Comparaciones con las mismas preguntas

Brier por pareja de modelos, con pesos iguales por pronóstico. Un valor menor es mejor. Todas las filas tienen etiqueta binaria y probabilidad original de cada modelo y del baseline oficial. Se excluyeron las preguntas de plataformas de pronósticos, incluida Metaculus. Se admitieron los datos financieros de Yahoo Finance. Los archivos no contienen el instante de generación de cada respuesta. Se exigió que la fecha oficial de pronóstico precediera a la resolución.

| Fechas y configuración | Casos en común | Brier |
|---|---:|---|
| 24 mayo–19 julio, 5 rondas, variantes por defecto | 2.682 | GPT-5.5 0,16962 · Gemini 3.5 Flash 0,17095 · GPT-5.4 0,18241 · Gemma 4 31B 0,19244 · GPT-5.4 mini 0,19482 · baseline 0,29367 |
| 2 agosto–27 septiembre, 5 rondas | 2.882 | GPT-5.6 Sol medio con web y ejecución de código 0,15654 · GPT-5.5 por defecto 0,16407 · Gemini 3.5 por defecto 0,16729 · baseline 0,28561 |
| 13 septiembre, 1 ronda, alta + web | 443 | GPT-6 Astra 0,12465 · Gemini 3.8 Flash 0,12517 · baseline 0,31202 |
| 27 septiembre y 11 octubre, alta + web | 727 | GPT-6 Sol 0,17598 · Gemini 3.8 0,18120 · GPT-6 Luna 0,19084 · baseline 0,28119 |
| 11 octubre, alta + web | 293 | GPT-6.1 Sol 0,23617 · Gemini 3.8 0,24374 · baseline 0,25470 |
| 25 octubre, 1 ronda, alta + web y datos de referencia incluidos en el prompt | 137 | GPT-6 Astra 0,20773 · GPT-5.5 0,21019 · Gemini 3.8 0,21289 · GPT-6.1 Sol 0,21563 · GPT-6 Sol 0,22546 · GPT-5.6 Sol 0,22704 · GPT-6 Luna 0,24043 · baseline 0,28007 |

En la primera cohorte GPT-5.5 mejora en 0,00134 de Brier frente a Gemini, IC95% por fecha [−0,00124; +0,00327], p ajustada 0,875. Frente a Gemma la mejora fue 0,02282 [0,00854; 0,03345], p ajustada 0,4375. La muestra común incluye 657 identificadores de serie y pregunta, con distintos horizontes y orígenes. Gemini y GPT-5.5 emitieron 3.649/3.649 pronósticos válidos antes del cruce. Gemma emitió 2.758/3.649.

En la segunda cohorte GPT-5.6 mejora frente a Gemini en 0,01075 de Brier [0,00260; 0,02136] y 0,02439 de log-loss [−0,00275; 0,05525]. Ganó 4/5 rondas, p ajustada de Brier 0,4375, con 829 identificadores subyacentes. Los modelos utilizaron distintas herramientas. El 25 de octubre, con configuraciones alta + web, Gemini 3.8 superó a GPT-5.6 en ambas pérdidas, y su log-loss de 0,60540 igualó prácticamente al 0,60555 de GPT-6 Astra. Ese panel de una fecha no ofrece validación multirronda. Los números de paneles distintos no se suman ni se comparan como si fueran una misma muestra.

Las fechas históricas tempranas ya se habían estudiado para otros modelos. El nuevo análisis incorporó candidatos de OpenAI y preregistró los contrastes antes de puntuarlos. Para demostrar rentabilidad harían falta cobertura del bot real, consumo facturado real y datos futuros sellados.

## Contraste externo y último modelo

El leaderboard general de FutureEval sitúa a GPT-5.6 Sol High con AskNews en el puesto 7 (11,79), 5.º entre modelos tras excluir los dos comparadores humanos. Gemini 3.7 Flash figura 6.º con 11,83 y Gemini 3.5 Flash, 9.º con 11,77. Gemma 4 del leaderboard corresponde a la variante 26B con AskNews (73.º, 2,12). La comparación de ForecastBench arriba utilizó 31B. Son sistemas, protocolos y poblaciones diferentes. [Tabla oficial](https://www.metaculus.com/futureeval/leaderboard/)

OpenAI publicó GPT-6.2 Astra y Sol en la API el 17 de noviembre. El archivo procesado oficial descargable tiene 141.253.184 bytes y fue actualizado el 17 de noviembre a las 08:27 UTC. La revisión examinada ofrece conjuntos de preguntas hasta el 22/11 y resoluciones hasta el 25/10. Una búsqueda sobre todos los nombres del archivo no encontró GPT-6.2. No se le atribuye una puntuación. [Anuncio](https://openai.com/index/introducing-gpt-6-2-astra-sol/) · [Archivos oficiales](https://www.forecastbench.org/datasets/) · [Revisión de resoluciones](https://github.com/forecastingresearch/forecastbench-datasets/tree/86b92ba06ed34d606d633bff3e99af54dd447d3a/datasets/resolution_sets)

## Presupuesto ilustrativo en dólares

Precios estándar de contexto corto. Escenario fijo de 2.000 tokens de entrada y 1.000 tokens **totales facturables de salida**, incluyendo razonamiento, por predicción. El archivo procesado carece de facturas, uso de tokens por llamada, llamadas a herramientas y latencias. El cuadro es una simulación de precio.

| Modelo | Entrada / salida por millón | 100 predicciones, solo tokens |
|---|---:|---:|
| GPT-5.4 mini | 0,75 / 4,50 | 0,60 |
| GPT-5.4 | 2,50 / 15,00 | 2,00 |
| GPT-5.5 | 5,00 / 30,00 | 4,00 |
| GPT-5.6 Sol | 4,00 / 20,00 | 2,80 |
| GPT-6 Astra | 10,00 / 50,00 | 7,00 |
| GPT-6 Sol / 6.1 Sol / 6.2 Sol | 2,00 / 10,00 | 1,40 |
| GPT-6 Luna | 0,10 / 0,50 | 0,07 |
| GPT-6.2 Astra | 8,00 / 40,00 | 5,60 |

Una búsqueda por cada uno de esos 100 pronósticos añade US$1, más los tokens de resultados. Si cada pregunta usara un contenedor nuevo de 1 GB durante una sesión de 20 minutos, se sumarían US$3 para 100 sesiones. El gasto real de los protocolos archivados no se conoce y los tokens internos pueden superar el supuesto. GPT-5.6 Sol no admite el nivel gratuito de la API y su precio promocional está anunciado hasta el 30/11/2026. No se validó la cuenta API de Mario ni se usaron créditos. Las cuotas de ChatGPT Work no prueban cobertura de un workflow externo. [Tarifas oficiales](https://developers.openai.com/api/docs/pricing) · [GPT-5.6 Sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol) · [GPT-5.5](https://developers.openai.com/api/docs/models/gpt-5.5) · [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna)

## Reproducción

La implementación requiere Python 3.11+ y NumPy. Descargar el archivo oficial procesado y los JSON de resolución en sus fechas originales desde la revisión indicada. El informe contiene los SHA-256 del archivo, de las previsiones procesadas y de cada conjunto de resoluciones. La política de fecha, las ocho hipótesis primarias Brier y el tratamiento de ausencias están en `preregister.json`. En la unión, la dirección vacía `[]` y la ausencia de dirección `null` se normalizan al mismo valor.

```sh
python compare_openai.py processed_forecast_sets.tar.gz directorio_resoluciones resultado.json
```

`results.json` conserva coberturas, exclusiones, intervalos, resultados por fecha y fuente y el análisis con primer pronóstico por objetivo. Una persona puede repetir el cálculo sin credenciales de los modelos. Los datos de ForecastBench, © Forecasting Research Institute, y las estadísticas derivadas se distribuyen bajo [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
