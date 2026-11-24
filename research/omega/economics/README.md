# OMEGA — coste de medir una mejora

**24 de noviembre de 2026. Gasto adicional autorizado y ejecutado para estos escenarios: R$0.** Son cálculos de capacidad. El código solo usa la biblioteca estándar de Python y funciona sin claves ni red. No activa un proveedor, una cuenta de pago o un torneo.

## Qué compra cada presupuesto hipotético

Un **par** son dos respuestas para el mismo caso sellado, una de GPT-5.6 Sol y otra de GPT-5.4 mini. Usan el mismo dossier y se contrastan además con una referencia estadística de coste marginal cero. Cada caso cuenta una sola vez, aunque se consulte también un tercer modelo. Tomamos como perfil central 3.000 tokens de entrada y 1.000 de salida por modelo. Los tokens de salida incluyen razonamiento facturable. Sin búsquedas adicionales, el par cuesta US$0,03875. El cierre de venta PTAX fue R$5,8573 por dólar el 24 de noviembre. El precio final en reales depende del método de pago, sus comisiones e impuestos aplicables.

| Techo | Equivalente al cambio PTAX | Pares con perfil central, solo inferencia | Pares centrales con 3 búsquedas pagadas compartidas | Pares largos aprovechables bajo estrés* | Máximo aprovechable por una ronda de 500 familias bajo estrés* |
|---:|---:|---:|---:|---:|---:|
| R$200 | US$34,15 | 881 | 544 | 225 | 225 |
| R$1.000 | US$170,73 | 4.405 | 2.720 | 1.133 | 450 |
| R$2.000 | US$341,45 | 8.811 | 5.441 | 2.266 | 450 |

\* Prueba de sensibilidad, no tasa observada. Cada modelo usa 8.000 tokens de entrada y 2.500 de salida. Un solo dossier común requiere tres búsquedas básicas a US$0,008 cada una. El dólar efectivo cuesta 10% más que la PTAX para modelar cambio y cargos, sin atribuir ese porcentaje a ningún impuesto. Si de diez casos intentados solo nueve tienen ambas respuestas válidas y desenlace observable al corte, se redondea el resultado hacia abajo. Al imponer una ronda, primero se limita a 500 intentos y después se aplica el 90%. Más actualizaciones o dossiers no compartidos multiplicarían el consumo. La tasa de resolución real puede ser cero para una ventana corta.

[PTAX oficial del Banco Central do Brasil](https://ptax.bcb.gov.br/ptax_internet/consultarUltimaCotacaoDolar.do) · [Tarifas GPT-5.6 Sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol) · [Tarifas GPT-5.4 mini](https://developers.openai.com/api/docs/models/gpt-5.4-mini) · [Tarifas y consumo de búsquedas](https://docs.tavily.com/documentation/api-credits)

## Sensibilidad al modelo y al número de tokens

| Modelo o combinación | US$/millón entrada | US$/millón salida | 800 entrada + 250 salida | 3.000 + 1.000 | 8.000 + 2.500 |
|---|---:|---:|---:|---:|---:|
| GPT-5.6 Sol | 4,00 | 20,00 | US$0,0082 | US$0,032 | US$0,082 |
| GPT-5.4 mini | 0,75 | 4,50 | US$0,001725 | US$0,00675 | US$0,01725 |
| GPT-6.2 Astra, alternativa de frontera | 8,00 | 40,00 | US$0,0164 | US$0,064 | US$0,164 |
| Groq GPT-OSS 120B, referencia de tarifa Developer | 0,15 | 0,60 | US$0,00027 | US$0,00105 | US$0,0027 |
| Sol + mini por caso | — | — | US$0,009925 | US$0,03875 | US$0,09925 |

Tarifas estándar directas, texto sin caché, sin herramientas integradas, búsquedas o almacenamiento. La tarifa promocional de Sol está publicada hasta el **30 de noviembre**. Una prueba prolongada necesita recálculo en cada tramo de tarifa. Las peticiones con cache writes, modelos regionales, múltiples razonamientos o rutas con sobreprecio pueden costar más. Astra cuesta el doble de Sol al mismo consumo en estos perfiles; con un dossier central serían US$6,40 de inferencia para 100 casos. Su ficha de producto no demuestra una mejora de forecasting. Sol y mini también pueden compartir errores de familia, por lo que se conserva el contraste externo frente a Groq y la referencia CPU. La fila Groq sirve para comparar listas públicas; el ensayo actual sigue en el plan gratuito. El precio de un solo Sol frente a una referencia CPU es la fila Sol, y el de dos paneles completos es la suma de sus usos reales.

[Tarifas de texto y condiciones de Sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol) · [Precio de lista de Astra](https://developers.openai.com/api/docs/pricing) · [GPT-5.4 mini](https://developers.openai.com/api/docs/models/gpt-5.4-mini) · [Catálogo de Groq](https://console.groq.com/docs/models)

## Cuántos resultados se podrían verificar

- El laboratorio ya dispone de 12 casos históricos en su cohorte ciega y cuatro casos prospectivos de Treasury con dos fechas de resolución. Repetir llamadas sobre ellos no crea casos nuevos. En el estado original del panel, diciembre todavía está por observar.
- ForecastBench anuncia **500 familias cada dos semanas**. Las preguntas sobre series tienen múltiples horizontes, entre una semana y un año. En la tabla se toma un único horizonte previamente fijado por familia para analizar independencia. El envío oficial tiene su propio formato y requiere al menos 95% de cobertura en **cada tipo** de pregunta. Las resueltas utilizables dependen del calendario y de que ambos brazos emitan a tiempo. La elegibilidad, la aceptación y los resultados individuales se verifican aparte. [Metodología](https://forecastbench.org/about/) · [Reglas de entrega](https://github.com/forecastingresearch/forecastbench/wiki/How-to-submit-to-ForecastBench)
- MiniBench anuncia alrededor de 60 preguntas por quincena. Con esa producción serían necesarias al menos unas dos rondas para alcanzar 100 preguntas, antes de excluir anulaciones, casos correlacionados o sin cobertura. La temporada FutureEval describe 300–400 preguntas durante unos cuatro meses. Se puede estudiar la arquitectura con casos independientes o archivos cerrados sin intervenir manualmente en las preguntas abiertas. [FutureEval y MiniBench](https://www.metaculus.com/futureeval/participate/)
- La cuota genérica publicada para GPT-OSS 120B de Groq incluye límites de peticiones y tokens. A 200.000 tokens/día, 4.000 tokens por respuesta permiten 50/día por ese límite, antes de cuotas por minuto, otras cargas o respuestas fallidas. La cuota real de la organización manda. Tres búsquedas básicas por caso consumen tres créditos Tavily; un cupo íntegro de 1.000 créditos/mes cubre 333 dossiers así. Son capacidades condicionales, sin suponer que las cuotas de la cuenta siguen intactas. [Límites Groq](https://console.groq.com/docs/rate-limits) · [Créditos Tavily](https://docs.tavily.com/documentation/api-credits)

## Cuándo aportaría valor cada tramo

**R$200.** 100 pares centrales con una referencia CPU costarían aproximadamente R$22,70 de inferencia a la tarifa del día. Sirven para medir cobertura, coste por par resuelto, latencia y distribución de diferencias de Brier/log loss. Recalcular con los tokens facturados y contrastar también frente al mejor brazo gratuito sobre exactamente los mismos casos. Esta es una simulación; el brazo de pago está bloqueado.

**R$1.000.** Unas 500 familias centrales costarían R$113,49 de inferencia; se puede separar desarrollo de evaluación posterior y publicar resultados por familia de preguntas. El capital sobrante no compra una segunda temporada, resoluciones más tempranas ni diversidad de eventos. Si se mantienen hasta 1.000 casos en varios bloques, la misma inferencia central costaría R$226,97 a precios constantes.

**R$2.000.** Solo tendría sentido como techo para replicaciones en nuevas rondas o dominios con dossier largo cuando se haya demostrado ventaja marginal del pago sobre la ruta gratuita. Para todas las opciones conviene preregistrar una métrica primaria, una diferencia mínima material, cobertura y fecha de corte. Comparar un caso por igual, usar intervalos por evento/fecha y reportar pérdidas además de victorias. Separar puntuación científica, elegibilidad para premio y rendimiento económico de los premios.

Como orientación matemática, con desviación típica *supuesta* de 0,10 para la diferencia pareada de Brier y resultados independientes, la semiamplitud normal aproximada al 95% es 0,0196 con 100 pares, 0,00877 con 500 y 0,00620 con 1.000. Un 2% relativo sobre Brier 0,20 equivale a 0,004 absoluto. Con ese ruido harían falta unos 2.401 pares independientes para que la semiamplitud fuese 0,004. Ese cálculo de precisión no es potencia del 80%, ni demuestra que sea alcanzable. Los eventos correlacionados agrandan los intervalos. Evitar examinar repetidamente los resultados para proclamar una victoria al primer cruce de un umbral.

Ningún presupuesto permite estimar una ganancia monetaria sin probabilidades previas de cada puesto y el **premio efectivo del puesto**. El valor incremental sería la suma de los cambios de probabilidad de cada puesto por su dotación, menos los costes. No se dispone de esas probabilidades. Para la decisión económica actual, el ingreso demostrado es cero.

## Contraste económico del informe de Gemini

| Afirmación | Resultado del contraste |
|---|---|
| R$200 equivalen a US$33–35 | Consistente con la PTAX de venta del día. Antes de cargos, R$200 equivalen a US$34,15. |
| US$0,01–0,04 por pregunta compleja; 800–3.000 pronósticos | A esos precios, la división da entre 853 y 3.414 llamadas simples por R$200. El perfil largo de Sol cuesta US$0,082 para **una** llamada antes de búsquedas. Las cifras de Gemini son un escenario posible, no capacidad demostrada ni resultados independientes. |
| Los cinco experimentos y un puesto superior se consiguen con US$0 | El análisis offline sobre archivos y CPU puede tener coste adicional cero. Un resultado de clasificación requiere cobertura real y resultados resueltos. Las cuotas externas pueden agotarse o no estar habilitadas. La clave API por sí sola no convierte la inferencia en gratuita. [Runners estándar de GitHub](https://docs.github.com/en/actions/concepts/billing-and-usage) · [Groq Free/Developer](https://console.groq.com/docs/billing-faqs) |
| Enrutar modelos puede ahorrar 40–50% | Cuantificable con registros emparejados de tokens, búsquedas y latencia. Consultar a todos los modelos y poner después peso cero a uno de ellos ya ha generado todos los costes. Medir la decisión de omitir llamadas **antes** de inferencia y comparar la calidad en una muestra independiente. |
| Tras 100 resoluciones se puede decidir si gastar | Un punto de revisión operativo es razonable. La magnitud, varianza, correlación y disponibilidad de resultados deciden la precisión. Cien no garantizan detectar 2% en Brier ni 0,015 en log loss. Cualquier decisión de ahorro puede tomarse antes sin declarar superioridad estadística. |

## Opciones adicionales con coste monetario cero

| Opción | Qué admite | Evidencia o límite |
|---|---|---|
| Prophet Arena Weekly | Agentes de software mediante el SDK y cuenta gratuita. Una entrada registrada por organización puede aparecer públicamente y las variantes pueden practicar en privado. | La cuenta y la admisión de MEGALAB no están verificadas. Empezar con el conjunto archivado y la validación local no implica inscribirse. [Guía de inicio](https://www.prophetarena.co/docs/quickstart) · [SDK del organizador](https://www.prophetarena.co/docs/sdk-reference) |
| Numerai Classic, sin staking | Entrenar un modelo sobre el dataset gratuito y recibir puntuaciones sobre nuevas rondas sin aportar fondos. | Mide señales bursátiles y métricas propias. Hay rondas de 20/60 jornadas; no se presupone premio, inversión ni operativa financiera. [Reglas y datos](https://docs.numer.ai/) |
| Metaculus Cup | Práctica del bot junto a humanos. | Metaculus excluye a los bots del premio en la Cup. Al elegir una ruta gratuita solo cabe atribuir valor de comparación. [Modalidades oficiales](https://www.metaculus.com/futureeval/participate/) |
| ARENA y archivo de ForecastBench | Comparaciones de código CPU sobre archivos ya publicados, sin contratar búsqueda ni inferencia. | Usar pronósticos emitidos antes de su resolución. Un LLM actual que reconteste preguntas históricas puede conocer los desenlaces. [Datos oficiales](https://forecastbench.org/datasets/) |

## Reproducción

Desde este directorio, ejecutar `python scenarios.py`. La salida debe coincidir byte por byte con `scenarios.json`. Las constantes guardan la cotización y las tarifas observadas, de manera que el resultado no cambie cuando se actualicen las páginas de precios.
