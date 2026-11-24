# GitHub Models y alternativas con las credenciales existentes

Comprobado el 24 de noviembre de 2026. El catálogo, la inferencia y BYOK de GitHub Models cerraron definitivamente el 30 de julio de 2026, también para clientes activos. `GITHUB_TOKEN` con `models:read` ya no sirve para inferencia. La página oficial vigente y el anuncio de retirada lo confirman. No se creó ningún workflow ni se invocó ningún modelo.

- https://docs.github.com/en/github-models
- https://github.blog/changelog/2026-07-30-github-models-is-now-retired/

OpenRouter documenta un tope de 50 consultas diarias para `:free` sin compra, compartido entre modelos; los intentos fallidos cuentan. Su catálogo público, consultado sin autenticación, no mostró ningún otro modelo de chat identificable con entrada y salida gratuitas, salvo el enrutador `openrouter/free`. Los otros dos identificadores sin sufijo y con precios cero correspondían a Lyria, para música. La alternativa existente es respetar la ventana del proveedor y hacer un único reintento, o esperar a que se libere la cuota de un modelo `:free` ya identificado. Google cataloga Gemma 4 con entrada y salida gratuitas y sin tarifa de pago; los modelos a 31B y 26B ya están contemplados en el código. No se verificó disponibilidad actual por inferencia.

- https://openrouter.ai/collections/free-models
- https://openrouter.ai/api/v1/models
- https://ai.google.dev/gemini-api/docs/pricing/
