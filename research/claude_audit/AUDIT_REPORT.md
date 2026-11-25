# MEGALAB OMEGA INFINITY · AUDITORÍA MATEMÁTICA ADVERSARIAL Y COMPETICIÓN CIENTÍFICA

**Rama exclusiva:** `research/claude-opus-audit` (base `main` @ `7b03d44dbf010f8388ac641cc045d8235b2af1f2`)  
**Coste API / inferencia de pago:** `US$0,00` (`0` llamadas de red, `0` preguntas abiertas consultadas, `0` predicciones publicadas)  
**Archivos productivos / workflows modificados:** `0`

---

## 0. Recuperación del estado previo de Claude Opus

Antes de continuar, se auditó el estado exacto dejado en disco por la sesión previa interrumpida:

| Elemento | Estado recuperado | Acción quirúrgica tomada |
|---|---|---|
| Rama `research/claude-opus-audit` | Creada localmente desde `main` (`7b03d44`), sin commits ni push remoto | Conservada como única rama de trabajo |
| `research/claude_audit/metaculus_math.py` | Implementado (geometría continua oficial de `Metaculus/metaculus`, `formulas.py` @ `81225ff0` y `score_math.py` @ `059e0a55`, validador API y wrappers SDK `0.3.5`) | Recuperado; corregido un defecto propio en `true_bucket_probs` para bordes cerrados (`open_lower=False` / `open_upper=False`), censurando masa fuera de rango hacia las cubetas extremas `1` y `N` |
| `research/claude_audit/builders.py` | Implementado (`tq_cdf` Student-$t_3$ en eje natural $u$, `normal_fit_cdf`, `metalog_cdf` Keelin 2016) | Recuperado; endurecida la validación ante entradas `NaN`/`Inf` y percentiles fuera de $(0,1)$ |
| Sondas y competiciones OOS | Sólo existía una sonda sintética preliminar en `scratch/probe.py`; faltaban la auditoría transversal de ramas (`champion-v2`, `omega-infinity`, `market-pulse/26q4`), la suite automatizada adversarial y las 3 competiciones científicas | Completadas íntegramente sin repetir trabajo previo |

---

## 1. Los Tres Defectos Matemáticos de Mayor Impacto (Código Verificado vs. Hipótesis)

### Clasificación estricta: Defectos reales demostrados vs. Hipótesis

- **[DEFECTO REAL DEMOSTRADO — D1]** Patologías de construcción CDF continua en `forecasting-tools==0.3.5` (`NumericDistribution.get_cdf`, usado en `main.py:574,609`, `main_with_no_framework.py:850`, `market_pulse.py:98`, `market_prices.py:332`) y desalineación de puntuadores continuos internos (`market_prices.py:339`, `paired_economic.py:107`).
- **[DEFECTO REAL DEMOSTRADO — D2]** Falacia de correlación residual aleatoric-epistémica ($\text{Corr}(p_i - Y, p_j - Y) \approx 0.94$) + sesgo de censura por la derecha en calibración isotónica Venn-Abers (`n_calibration = 0` para horizontes $>30$ días en `research/omega/binary/evaluate.py:91,108,237` y `compare_cdef.py:86`) + truncamiento binario `[0.01, 0.99]` frente al rango oficial `[0.001, 0.999]` en `main.py:383,389`.
- **[DEFECTO REAL DEMOSTRADO — D3]** Desajuste de ventanas de volatilidad (`505` días empíricos escalados por `252` días en `market_prices.py:282,316-320`; ventana de `30` días copiada como `30` meses en `research/omega/continuous/evaluate.py:170`), mezcla no estandarizada de regímenes heterocedásticos e infracobertura severa (`62,96%` campeón y `68,52%` CQR frente al `80%` nominal).
- **[HIPÓTESIS NO DEMOSTRADA EN PRODUCCIÓN CON LLM VIVO]** Que sustituir la agregación o calibración posproceso mejore la clasificación en vivo de FutureEval/Market Pulse cuando los cuantiles provienen de rutas gratuitas sujetas a *rate-limits*; requiere validación prospectiva sellada sin tocar flujos activos.

---

### Detalle matemático de los 3 defectos prioritarios

#### Defecto 1: Construcción CDF continua y reglas oficiales de puntuación de Metaculus
En `Metaculus/metaculus` (`scoring/score_math.py:99-104`, `utils/the_math/formulas.py:130-148`):
$$\text{Baseline}_{\text{cont}}(b) = \frac{100}{2}\ln\!\left(\frac{\text{pmf}[b]}{\text{base}[b]}\right), \qquad \text{base}[b] = \begin{cases}0.05, & b \in \{0, N+1\},\\ \frac{1 - 0.05\,k_{\text{open}}}{N}, & b \in \{1,\dots,N\},\end{cases}$$
y para un campo fijo de oponentes $\Delta\text{Peer} = 50\,\Delta\ln\text{pmf}[b]$. Sin embargo, `NumericDistribution.get_cdf()` introduce cuatro fallos deterministas:

1. **Impuesto de cola del 10,08% y pérdida de invariancia de rango (D1a):** cuando `open_lower_bound=True` y `open_upper_bound=True` con cuantiles declarados en `[P10..P90]`, `_add_explicit_upper_lower_bound_percentiles` fija `range_min -> 5%` y `range_max -> 95%` e interpola linealmente la CDF en `[range_min, P10]` y `[P90, range_max]`. Fuerza `cdf[0] = 0.0504` y `1 - cdf[-1] = 0.0504` (**50,4x** el mínimo oficial `0.001`) y aplana los hombros de la distribución. Para una creencia exacta $\mathcal{N}(500, 20^2)$ en `[0, 1000]` abierto, pierde **$-27,85$ puntos Baseline esperados** (y **$-145,3$ puntos** ante un resultado a $2,25\sigma$, $x=545$). Además, al cruzar `P90` de `99.99` a `100.00` en `[0, 100]`, `1 - cdf[-1]` salta discontinuamente de `0.0504` a `0.0998`.
2. **Colapso por `buffer = 1` en escalas logarítmicas cerradas y desplazamiento de masa en cero (D1b):** `buffer = 1 if range_size > 100 else 0.01 * range_size` se calcula en unidades lineales incluso cuando `zero_point is not None`. En `[0.001, 1000]` (`zero_point = 0`, borde inferior cerrado), `buffer = 1.0` sobrescribe **todos** los percentiles $\le 1.001$ (las 100 primeras cubetas logarítmicas) a `1.001`, desplazando la moda `20x` por encima de una mediana en `0.05` y perdiendo **$>80$ puntos Baseline esperados**. En preguntas de conteo `[0, 50]` con `P10=P20=P40=0`, desplaza la masa puntual en `0` de la cubeta `1` (donde `formulas.py` puntúa `0.0`) a `0.50` (cubeta `3`), perdiendo **`57,1` puntos** cuando el resultado es `0`.
3. **Inversión silenciosa de monotonía y fallo de serialización por cuantiles repetidos (D1c):** `_check_and_update_repeating_values` resta `(1 - p) * 1e-6` *después* de validar orden creciente. En escalas estrechas (`<1e-5`, p. ej. índices repetidos de `main.py:606`), invierte `declared_percentiles` (`P20 < P10`), crea saltos espurios de CDF y lanza `ValueError` al revalidar (`model_dump -> model_validate`).
4. **Discrepancias con la especificación oficial en evaluadores internos (D1d/D1e):**
   - `market_prices.py:339` (`_mass`) usa `math.floor((y - low)*bins/(high - low))` (intervalos `[a, b)`), mientras `formulas.py:146` usa `max(int(u*N + 1 - 1e-10), 1)` (intervalos `(a, b]`), errando de cubeta en cualquier frontera interior exacta.
   - `paired_economic.py:107` (`champion-v2`) asigna masa exterior `0.01 / 122 = 8.2e-5 < 0.001` en VIX (`n=120`), inválida según el API de Metaculus.
   - `main.py:689-690` trunca límites de `DateQuestion` con `.date().isoformat()`, induciendo fechas `00:00:00Z` inferiores a un `lower_bound` intradía cerrado.

#### Defecto 2: Conflación de varianza de Bernoulli con correlación epistémica y censura temporal en calibración binaria
En `research/omega/binary/evaluate.py:91-94` (y `compare_cdef.py:86`) se estima la correlación entre modelos mediante `np.corrcoef((probs - y[:, None]).T)` ($\approx 0.944$) y se introduce en $\alpha = \sqrt{K / (1 + (K-1)\hat{\rho})}$.

**Teorema exacto (demostrado y verificado en `binary_aggregation.py`):** Para pronosticadores calibrados $p_k = \pi + \varepsilon_k$ con $\mathbb{E}[\varepsilon_k \mid \pi] = 0$, $\text{Var}(\varepsilon_k) = \sigma_k^2$ e $Y \sim \text{Bernoulli}(\pi)$:
$$\text{Cov}(p_i - Y, p_j - Y) = \text{Cov}(\varepsilon_i, \varepsilon_j) + \mathbb{E}[\pi(1-\pi)].$$
Como la varianza irreducible de realización $\mathbb{E}[\pi(1-\pi)] \approx 0.17\text{--}0.25 \gg \sigma_k^2 \approx 0.005$, resulta $\text{Corr}(p_i - Y, p_j - Y) \approx 0.94$ **incluso cuando $\text{Corr}(\varepsilon_i, \varepsilon_j) = 0$**. Además, demostramos la **identidad exacta de cancelación de varianza**:
$$\frac{1}{2}\mathbb{E}\big[(p_k - Y)^2 + p_k(1 - p_k)\big] = \frac{(A + \sigma_k^2) + (A - \sigma_k^2)}{2} \equiv A = \mathbb{E}[\pi(1-\pi)],$$
acotada superiormente por el límite exacto PSD de Sherman-Morrison $A_{\max} = (\mathbf{1}^\top \Sigma_{\text{raw}}^{-1}\mathbf{1})^{-1}$. En los datos reales de ForecastBench (`train+valid`, $N=2.304$), $\hat{A} = 0.17180$ frente a $A_{\max} = 0.17266$, revelando que la correlación epistémica media real es **`0.0555`** (no `0.9438`).

Adicionalmente:
- **Censura por horizonte en `valid` (D2b):** el purgado estricto dejó **`0` casos de horizonte $>30$ días** en `train` (`0/1.311`) y `valid` (`0/993`), frente a **`2.872 / 5.723` (`50,2%`)** en `test`. Ajustar Venn-Abers isotónico (`venn_mean`) sobre `valid` sesgó las probabilidades en `test`, degradando Brier (`0.1572 -> 0.1626`), duplicando ECE (`0.0415 -> 0.0772`) y disparando `log_loss_p95` (`0.968 -> 1.665`).
- **Clipping binario subóptimo en `main.py:383,389` (D2c):** `[0.01, 0.99]` frente al límite oficial `[0.001, 0.999]`, imponiendo una pérdida de `1,31` puntos Baseline en eventos casi seguros (`99,9%`) y un suelo `100x` mayor en Brier (`1e-4` vs `1e-6`).

#### Defecto 3: Desajuste de ventanas de volatilidad y mezcla heterocedástica en series macro/mercado
- En `market_prices.py:282,316-320`, los cambios históricos se muestrean sobre **505 sesiones** pero el multiplicador $\sqrt{\sigma_{30} / \sigma_{252}}$ divide únicamente por las últimas **252 sesiones**. Tras una transición de régimen calmo ($\sigma=0.4$) a volátil ($\sigma=2.0$), `multiplier = 1.0` mientras la mitad de la muestra proviene del régimen calmo, subestimando el IQR en **$>40\%$** y el ancho al 80% en **$>20\%$**.
- En `research/omega/continuous/evaluate.py:170`, `diff[-30:]` tradujo literalmente 30 sesiones bursátiles como **30 meses (2,5 años)** en el desempleo mensual `UNRATE`, y `calibration()` entrenó CQR sólo en `1990–2006` con expansión truncada $\ge 0$, provocando infracobertura grave en `test` (`62,96%` `champion_rule`, `68,52%` `cqr_tails` frente al mínimo `74%` del protocolo).

---

## 2. Batería Automatizada de Pruebas Adversariales

Archivo: `research/claude_audit/test_adversarial_audit.py`  
Ejecución verificada: `.venv/bin/python -m unittest discover -s research/claude_audit -p 'test_*.py' -v` -> **`15/15` pruebas en `6,47 s`, `0` fallos.**

Cubre determinísticamente:
- `D1a–D1e`: impuesto de cola abierta del 5%, pérdida de invariancia de rango, discontinuidad en `P90 = upper_bound`, colapso logarítmico por `buffer=1`, desplazamiento de masa en `0` en borde cerrado, inversión de orden + fallo Pydantic en cuantiles empatados, desalineación de cubetas en `market_prices._mass` y violación `<0.001` en `paired_economic.masses`, y truncamiento de fechas intradía en `main.py`.
- `D2a–D2c`: inflación de correlación residual bajo independencia condicional ($\rho_{\text{raw}} > 0.95$ vs $\hat{\rho}_{\text{debiased}} < 0.08$), degradación de Venn-Abers bajo validación censurada por horizonte y penalización del clipping `[0.01, 0.99]`.
- `D3a–D3b`: subdispersión por desajuste `505` vs `252` corregida mediante Filtered Historical Simulation (Hull & White 1998) e **invariancia causal exacta de prefijo (`np.array_equal`)** ante corrupción arbitraria de etiquetas/observaciones futuras tanto en binario como en continuo.
- Estabilidad numérica en escalas `[1e-9, 1e15]`, colas pesadas, tamaños discretos y rechazo limpio (`ValueError`) de entradas corruptas (`NaN`, `Inf`, cuantiles decrecientes o `<= zero_point`).

---

## 3. Competición Científica Fuera de Muestra (Cero Gasto, Controles Holm)

Archivo reproducible: `research/claude_audit/run_competitions.py` -> `research/claude_audit/results_competition.json` (`23,84 s` totales en CPU). Todos los hiperparámetros se fijaron antes de evaluar `test`.

### Competición 1: Construcción CDF Continua (`tq_cdf` vs Producción y Alternativas)

**1A. Benchmark matemático exacto (8 regímenes $\times$ 50 distribuciones = 400 casos OOS):**

| Método | Baseline esperado oficial ↑ | RPS discreto ↓ | Fallos / 400 | Latencia media | Ganancia vs `sdk_cdf` (IC 95%) | $p$ exacta bloques (Holm) | Veredicto |
|---|---:|---:|---:|---:|---:|---:|---|
| `sdk_cdf` (producción actual) | `50,828` | `0,009952` | `0` | `12,72 ms` | referencia | — | Superado en 400/400 casos |
| `metalog_cdf` (Keelin 2016) | `56,880` | `0,033408` | `51` (`12,8%`) | `1,14 ms` | `+6,05` `[+1,76, +10,21]` | `0,3633` (`0,3633`) | **DESCARTADO** (infactibilidad) |
| `normal_fit_cdf` (referencia simple) | `71,282` | `0,000765` | `0` | `0,98 ms` | `+20,45` `[+18,54, +22,55]` | `0,0039` (`0,0117`) | Sólido en gaussianas; cede en colas/asimetría |
| **`tq_cdf` (Student-$t_3$ eje $u$ + censura)** | **`72,630`** | **`0,000832`** | **`0`** | **`0,92 ms`** | **`+21,80` `[+19,96, +23,79]`** | **`0,0039` (`0,0117`)** | **PASS (gana 8/8 regímenes, 100% casos, 13,9x más rápido)** |
| *Cota teórica `oracle_true_cdf`* | *`73,738`* | *`0,000938`* | *`0`* | — | *`+22,91`* | — | *Techo con distribución verdadera conocida* |

**1B. Cohorte real congelada Market Pulse 26Q4 ALFRED (`199e336`, $N=6$ subpreguntas resueltas de crédito HY y bono 10Y, 100% offline):**
- `sdk_cdf`: media **`+21,45`** puntos Baseline (sufre `-60,44` en `46212` y `-92,44` en `46236` por el aplanamiento de hombros).
- **`tq_cdf`**: media **`+49,35`** puntos Baseline (**`+27,90` puntos/pregunta** sobre `sdk_cdf`). `normal_fit_cdf` alcanza `+56,17` en estas 6 series de tipos/spreads ($N=6$ descriptivo).

---

### Competición 2: Agregación y Calibración Binaria en ForecastBench (`80bdeaa`, `test` $N = 5.723$)

| Método | Brier ↓ | Log Loss ↓ | Log Loss P95 ↓ | ECE-10 ↓ | Tiempo (`5.723` casos) | Ganancia Brier vs `mini` (IC95% 2-rondas) | Veredicto científico |
|---|---:|---:|---:|---:|---:|---:|---|
| **`mini` (mejor modelo individual previo)** | **`0,15471`** | **`0,45130`** | `1,0498` | **`0,02443`** | — | referencia | **Líder imbatido en Brier** |
| `correlation_pool` (OMEGA) | `0,15472` | `0,45079` | `1,0164` | `0,03313` | `5.490 ms` (lote) | `-0,00001` `[-0,00117, +0,00133]` | No supera a `mini` (`p_Holm = 1,00`) |
| `satopaa_debiased` (publicado + D2a fix) | `0,15575` | `0,45192` | `1,2082` | `0,04143` | `1,87 ms` | `-0,00104` `[-0,00318, +0,00091]` | **DESCARTADO** (cola P95 y dilución) |
| `heb_lop_online` (propuesto causal) | `0,15596` | `0,45601` | `1,0312` | `0,02788` | `171,81 ms` | `-0,00125` `[-0,00372, +0,00101]` | **DESCARTADO** frente a `mini` (`p_Holm = 1,00`) |
| `heb_lop_static` (propuesto estático) | `0,15624` | `0,45729` | `1,0020` | `0,03072` | `35,21 ms` | `-0,00154` `[-0,00475, +0,00135]` | **DESCARTADO** frente a `mini` |
| `median` / `inverse_brier` / `hedge` / `mean` | `0,15639–0,15724` | `0,45690–0,46308` | `0,9613–0,9676` | `0,02916–0,04154` | `<2 ms` | `-0,00168` a `-0,00253` | Superados por `mini` |
| `venn_mean` (OMEGA isotónico) | `0,16259` | `0,47119` | `1,6650` | `0,07716` | — | `-0,00788` `[-0,01099, -0,00522]` | **FALLO SEVERO** por censura de horizonte |

**Hallazgo clave:** `heb_lop_online` corrige el colapso de `venn_mean` (`-66 bps` Brier, `-63 bps` LogLoss P95, ECE `0,0279` vs `0,0772`) y supera a `mean`/`median`/`hedge`/`inverse_brier`, pero **ningún ensemble de 3 modelos supera a `mini` en solitario** porque `deepseek` degradó su Brier en `test` hasta `0,18354` (`+288 bps` peor que `mini`) y `train+valid` carecía de horizontes largos (`>30d = 0`). Siguiendo la regla estricta de la Sección 4, **se descartan `heb_lop_*` y `satopaa_debiased` para promoción en este panel** y se recomienda una compuerta de *forecast encompassing* (Chong & Hendry 1986) antes de mezclar modelos con brecha de Brier $>150$ bps.

---

### Competición 3: Distribuciones Continuas Macro Point-in-Time (`unrate_realtime.csv` ALFRED, `test` $N = 108$, `2016–2024`)

| Brazo | CRPS ↓ | Log-pérdida densidad (NLL) ↓ | Brier (+0,2pp) ↓ | Cobertura 80% (`[74%, 86%]`) | Anchura 80% (pp) | PIT KS ↓ | Criterio E5 estadístico |
|---|---:|---:|---:|---:|---:|---:|---|
| **`robust_t_aci` (Huber-Momentum $t_{3.5}$ + ACI causal)** | **`0,51464`** | **`0,78995`** | **`0,07254`** | **`77,78%`** | `0,739` | **`0,1431`** | **ÚNICO EN CUMPLIR (`True`)** |
| `cqr_tails` (mejor OMEGA previo) | `0,53876` | `0,81896` | `0,08708` | `68,52%` (Falla) | `0,792` | `0,2082` | `False` |
| `champion_rule` (regla campeón trasladada) | `0,54129` | `5,55617` | `0,08543` | `62,96%` (Falla) | `0,702` | `0,2159` | `False` |
| `empirical` | `0,55021` | `6,03313` | `0,08434` | `77,78%` | `0,754` | `0,2148` | `False` |
| `three_scales` | `0,55159` | `5,45086` | `0,07548` | `81,48%` | `1,268` | `0,2037` | `False` |
| `holt_damped_t` (Gardner-McKenzie 1985) | `0,56733` | `0,97887` | `0,11094` | `76,85%` | `0,718` | `0,1367` | `False` (**DESCARTADO** por COVID) |
| `ridge_ar` | `0,58136` | `25,80136` | `0,19912` | `90,74%` (Falla) | `2,344` | `0,3148` | `False` |
| `naive_rw` | `0,66755` | `18,70608` | `0,16748` | `85,19%` | `3,333` | `0,3102` | `False` |

**Contrastes pareados OOS (`test`, 36 trimestres, `778,6 ms` para las 404 evaluaciones):**
- `robust_t_aci` vs `champion_rule`: ganancia media CRPS **`+0,02665` (`-4,92%`)**, **IC 95% moving-block bootstrap (4 trimestres, 2.000 réplicas, semilla oficial `20261124`) = `[+0,00220, +0,05871]` (excluye cero)**. Incluso en el shock COVID-19 (`2020–2021`, $N=24$) reduce CRPS (`1,541` vs `1,650`), NLL (`3,529` vs `19,080`) y Brier (`0,1163` vs `0,1763`) elevando cobertura de `25,0%` a `41,7%`.
- `robust_t_aci` vs `cqr_tails`: ganancia media CRPS **`+0,02412` (`-4,48%`)**, `25` trimestres ganados vs `11` perdidos, **IC 95% moving-block bootstrap = `[+0,00225, +0,05302]` (excluye cero)**, $p$ unilateral exacta sobre 9 macro-bloques de 4 trimestres = `0,0352` (`p_Holm = 0,1758` por el límite discreto de potencia con sólo 9 bloques independientes).
- **Veredicto riguroso:** **PASS estadístico en el protocolo E5 y en todos los intervalos block-bootstrap al 95%**; **NO promoción automática a producción** hasta completar validación prospectiva sellada y cruce con cuantiles LLM (por potencia finita $2^9 = 512$ bajo corrección Holm y ausencia de cuantiles LLM archivados en `UNRATE`).

---

## 4. Limitaciones y Recomendaciones Quirúrgicas de Integración para Work

### Limitaciones explícitas
1. Ningún experimento ejecutó inferencias LLM nuevas (restricción absoluta de coste cero); en `Competition 1`, `tq_cdf` se evaluó sobre cuantiles verdaderos/congelados exactos, aislando matemáticamente el paso `cuantiles -> continuous_cdf`.
2. Con 9 bloques trimestrales no solapados en `UNRATE`, cualquier test exacto de signos con 5 comparaciones simultáneas tiene potencia limitada bajo Holm; los intervalos bootstrap por bloques de 4 trimestres sí excluyen cero.
3. En ForecastBench binario, la ventana histórica disponible (`train+valid`) está truncada a horizontes $\le 30$ días.

### Recomendaciones accionables por orden de impacto para ChatGPT Work / Codex
1. **[PRIORIDAD 1 — Integración inmediata tras tests de humo]** Sustituir la construcción interna `NumericDistribution.get_cdf()` en `main.py` (`_numeric_prompt_to_forecast`) y `market_pulse.py` (`numeric_cdf`) por `tq_cdf(percentiles, geo, nu=3.0)` manteniendo intacta `_standardize_cdf`. Elimina de raíz el impuesto del 10% en límites abiertos, el bug `buffer=1` en preguntas logarítmicas cerradas, el desplazamiento de átomos en `0` y los fallos por cuantiles repetidos (`+21,8` puntos Baseline esperados, `13,9x` menor latencia).
2. **[PRIORIDAD 2 — Corrección de 3 líneas en `main.py` y `market_prices.py`]**
   - En `main.py:383,389`: cambiar `max(0.01, min(0.99, ...))` a `max(0.001, min(0.999, ...))` para alinear el rango binario con `MetaculusClient` y `score_math.py`.
   - En `main.py:689-690`: usar `.isoformat()` completo cuando `lower_bound`/`upper_bound` de `DateQuestion` tengan hora UTC distinta de `00:00:00`.
   - En `market_prices.py:339`: reemplazar `math.floor(...)` por la regla exacta de cubetas de Metaculus (`metaculus_math.Geometry.bucket`), y alinear `long_vol` con la ventana muestral completa o usar `fhs_standardized_quantiles`.
3. **[PRIORIDAD 3 — Regla de seguridad para ensembles binarios e integración sombra macro]**
   - Prohibir calibración isotónica Venn-Abers cuando la muestra de calibración esté censurada por horizonte; exigir test de *forecast encompassing* antes de promediar modelos secundarios con el modelo líder.
   - Incorporar `robust_t_aci` como brazo sombra adicional en el ensayo prospectivo continuo de `UNRATE` sin alterar liquidaciones ya selladas.

---

## 5. Reproducción Offline y Evidencias Criptográficas

```bash
sha256sum -c research/claude_audit/SHA256SUMS
.venv/bin/python -m unittest discover -s research/claude_audit -p 'test_*.py' -v
.venv/bin/python research/claude_audit/run_competitions.py
```

### Commits y fuentes oficiales verificadas
- Base `main`: `7b03d44dbf010f8388ac641cc045d8235b2af1f2`
- `origin/research/champion-v2`: `a152a4eb29a23cf5c1a00a38a04e35e025141bf4`
- `origin/research/omega-infinity`: `80bdeaaf9c099005cd7ab9aa838332dd497d390e` (`rows.jsonl` SHA-256 `6d7b464141d5c0c5a141df2fdc8b3ec25dcb0ab1f2299950324795c5b6776860`, `unrate_realtime.csv` SHA-256 `ec6167168d03bc6f56b57a0165545a9346e22c84052b89fa3484907ddb45fe08`)
- `origin/market-pulse/26q4`: `199e336d333f74b47b71a1df607cc430107dab39`
- `Metaculus/metaculus:scoring/score_math.py` Git blob SHA: `059e0a55a9edf95a79c07985e4ada9bed2631440`
- `Metaculus/metaculus:utils/the_math/formulas.py` Git blob SHA: `81225ff04451a1201cc514ab01453a1cafcf43d1`
