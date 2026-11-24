# Publicación y puntuación de las rondas propias de MEGALAB

El registro de investigación de MEGALAB es público. Cada pronóstico en sombra se conserva en la rama `research/omega-forecastbench-shadow` del proyecto. Este procedimiento carece de credenciales de subida al concurso. La tabla oficial y la elegibilidad de MEGALAB siguen dependiendo de la aprobación del equipo por ForecastBench y de una entrega aceptada por el organizador.

La emisión está acotada a nueve domingos alternos, entre el 6 de diciembre de 2026 y el 28 de marzo de 2027. El job ejecuta los métodos congelados, sin inferencia externa. El horario principal de los domingos es 05:11 y 17:11 UTC. El control existente puede invocar el mismo job a las 07:17 y 19:17 UTC. GitHub serializa los accesos de ambos mediante el mismo grupo de concurrencia. Las comprobaciones fuera de ronda pueden incorporar resoluciones oficiales posteriores a marzo.

Cada ronda conserva:

1. `forecastbench/leases/FECHA.json`, con el SHA de la implementación y la primera ejecución que intentó emitir. El programa guarda esta reserva antes de generar las probabilidades. Un fallo anterior al sellado permite repetir el intento ese mismo día con el código reservado.
2. `forecastbench/FECHA.json`, las predicciones completas. Se guardan una sola vez en un commit independiente mediante un push normal. Un avance concurrente en la rama bloquea la escritura.
3. `forecastbench/receipts/FECHA.json`, con SHA-256 de los bytes, SHA del commit, ejecución de GitHub Actions y `created_at` del estado que GitHub asignó a ese commit. Se exige una marca del bot oficial, en el repositorio y contexto previstos, correspondiente al día de la ronda en UTC. El tiempo del servidor permite comprobar que el sello existía antes de las 00:00 UTC del día siguiente. Los campos de fecha de autor o committer de Git no deciden la validez.
4. `forecastbench/scores/FECHA.json`, resultados en los casos para los que el organizador publica datos de resolución. Se contrasta de nuevo el recibo con los estados de GitHub cada vez que cambian las puntuaciones. Una respuesta perdida de GitHub se recupera consultando el estado existente. Si al cierre falta una marca válida, la ronda queda excluida de la evaluación.

La cuenta utilizada pertenece al workflow y necesita `contents: write` y `statuses: write` en este repositorio. No es una cuenta de Google, no remite entregas a GCP y no utiliza claves de pago. Una ejecución en la rama de verificación prueba el permiso y el reloj del servidor con el contexto `megalab/shadow-clock-self-test`. Ese contexto nunca certifica un pronóstico. La continuidad depende de que GitHub Actions permanezca habilitado y de que el organizador publique los conjuntos y las resoluciones.

Las pruebas reproducen el cierre UTC, el acceso concurrente, la conservación de bytes, los reintentos y el rechazo de estados incorrectos en repositorios Git temporales.

```sh
python -m unittest discover -s research/omega/forecastbench_shadow_ops -v
```

[Formato oficial de los estados de un commit de GitHub](https://docs.github.com/en/rest/commits/statuses) · [Normas oficiales de ForecastBench](https://github.com/forecastingresearch/forecastbench/wiki/How-to-submit-to-ForecastBench)
