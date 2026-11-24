# Nivel y cuotas de Google AI Studio · 2026-11-24

AI Studio mostró «Nível gratuito» y «Configurar faturamento» para el único proyecto importado visible en la sesión existente. La interfaz avisa que puede haber otros proyectos sin importar. En «Todos os modelos», la tabla de cuotas del mismo proyecto dio estos límites de texto; son límites, no cuota restante.

| Modelo | RPM | TPM | RPD |
|---|---:|---:|---:|
| Gemini 3.5 Flash Lite | 15 | 250.000 | 500 |
| Gemma 4 31B | 30 | 16.000 | 14.400 |
| Gemma 4 26B | 30 | 16.000 | 14.400 |
| Gemini 3.8 Flash | 5 | 250.000 | 20 |
| Gemini 3.6 Flash | 5 | 250.000 | 20 |

No apareció una fila para Gemini 3.5 Flash sin «Lite». La combinación observada de modelos utilizados coincide con la configuración del bot. No basta para vincular inequívocamente el secreto de GitHub con el proyecto mostrado. No se abrieron páginas de claves. La configuración de Actions inspeccionada tampoco identifica el proyecto propietario ni incluye identidad de Google Cloud con permisos de lectura de facturación. `keys.lookupKey` y `projects.getBillingInfo` oficiales requieren autorización de administración/lectura por IAM/OAuth. Los topes de gasto configurables pueden procesarse con retraso. No se verificó un tope duro por solicitud ni se cambió la configuración o ejecutó inferencia. Coste de esta verificación US$0.

Fuentes: [proyectos y facturación](https://ai.google.dev/gemini-api/docs/billing#verify-billing), [límites del proyecto](https://ai.google.dev/gemini-api/docs/rate-limits), [IAM para identificar el proyecto de una clave](https://docs.cloud.google.com/api-keys/docs/reference/rest/v2/keys/lookupKey), [consulta del estado de facturación](https://docs.cloud.google.com/billing/docs/reference/rest/v1/projects/getBillingInfo), [latencia de topes](https://ai.google.dev/gemini-api/docs/billing#project-spend-caps). Sin datos personales ni números de proyecto en este informe.

El trabajo autenticado de GitHub confirmó además que la clave OpenRouter entregada al laboratorio está en la modalidad gratuita. Su endpoint de cuenta no devuelve cuántas peticiones gratuitas quedan ese día. En esa ejecución, el modelo nombrado de OpenRouter respondió HTTP 429 y los dos Gemma directos devolvieron 503. Ninguna inferencia de pago se habilitó. [Resultado de capacidades por proveedor](https://github.com/xmjunior/mmegalab-futureeval-bot/blob/32498e6cf6fe820d1a4ad512d8762e091ed08583/records/runs/45463803958-1/provider-status.json)
