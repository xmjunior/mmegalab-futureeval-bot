# OMEGA binary archive experiment

**Decision: zero of four challengers met the registered screen.** Across 5,723 matched final events, log-odds pooling with an estimated error correlation improved mean Brier by 0.002517 over arithmetic averaging (1.60%). The preselected GPT-5 mini already had a slightly lower Brier, with one archived output per case compared with three for the pool. This result supports no change to the live bot or paid inference.

| Final method | Brier | Log loss | ECE, 10 bins |
|---|---:|---:|---:|
| GPT-5 mini, best individual on training | 0.154708 | 0.451300 | 0.024432 |
| Correlation-aware log-odds pool | 0.154721 | 0.450793 | 0.033125 |
| Specialist Hedge | 0.157108 | 0.462759 | 0.040442 |
| Arithmetic mean | 0.157238 | 0.463085 | 0.041538 |
| Venn-Abers on mean | 0.162589 | 0.471190 | 0.077157 |
| Disagreement shrinkage | 0.163391 | 0.477677 | 0.051815 |
| Constant 0.5 | 0.250000 | 0.693147 | 0.086580 |

Residual correlations on training were 0.935–0.958, resulting in a log-odds multiplier of 1.092. For the pool vs the mean, the 95% envelope from date-block and family-block bootstraps was [0.000202, 0.004268] in paired Brier gains. Six test rounds gave a one-sided exact round sign p of 0.109375 and a Holm p of 0.4375. The inferential tests and the comparison with the preselected individual both matter. Actual model API fees in the archive were unavailable. New API charges were zero.

## Samples and limits

- Training had 1,311 matched events from four scored rounds, calibration 993 from three, and test 5,723 from six. The common test coverage was 5,723 / 5,786 eligible first issues (98.91%). Fifteen rounds were registered. Two recent resolution sets were absent from the official historical snapshots and correctly provided zero fitting rows. Sample eligibility and per-model coverage are broken down by issue round in `inputs.json`.
- Events are deduplicated at the underlying identifier and target-date level. Preceding immature events are still embargoed. Each resolved label must already occur in the immutable official Git snapshot for its split. The processed model archive supplies official round deadlines without individual API timestamps. Historical leaderboard eligibility is an input to the replay.
- 2,872 final test cases had horizons above 30 days, while all training and calibration cases were short; the long-horizon Hedge buckets therefore fell back to global weights. This study does not establish skill learned specifically on long horizons. Results for each horizon and data source are in the JSON.
- In the secondary deterministic family-disjoint split there were 802/190/1,372 training/calibration/test events across 226/94/224 disjoint families. ACLED country and French-temperature geography are grouped using official metadata; otherwise the series or question ID is kept. Test Brier in this secondary population was 0.15757 for correlation pooling, 0.16021 for the mean and 0.16740 for Venn-Abers. The best training model changed to DeepSeek in that smaller sample; GPT-5 mini scored 0.15683 on its final slice. Stable series IDs, country groups, whole-source and common-date sensitivities cannot capture every relation among underlying events.
- Scores are custom paired averages on binary data-source questions. They do not measure official rankings, new forecasts or outcomes on other competitions. Conditional comparisons omit missing or imputed forecasts. ECE depends on bin choice. The study has six test rounds despite thousands of events; family and temporal resampling are sensitivity analyses, not a guarantee of independent samples.

## Reproduction

The full 15-round protocol was published before calculation of held-out scores as Git commit `6a053aa7bf6c1335a172f87242648e28d26f94d8`. Use Python 3.11 or later and NumPy. The self-contained derived rows are compressed JSON lines, licensed under CC BY-SA 4.0. `inputs.json` identifies columns, immutable resolution commits, per-file digests and the decompressed payload digest. The additional final row field is the conservative family key.

```sh
python -m unittest discover -s research/omega/binary -p test_binary.py -v
python research/omega/binary/evaluate.py --out /tmp/omega-replay.json
```

For independent regeneration download the processed archive at the URL in `inputs.json`. The derivation verifies its exact digest, retrieves historical resolution versions and question text for grouping from immutable official commits, and recreates the compressed archive, in a separate directory if desired.

```sh
python research/omega/binary/derive.py processed_forecast_sets.tar.gz label-cache --output /tmp/omega-rebuilt
python research/omega/binary/evaluate.py --inputs /tmp/omega-rebuilt --out /tmp/omega-rebuilt/results.json
```

Original datasets and forecasts © Forecasting Research Institute, CC BY-SA 4.0. These curated rows, comparisons and source-derived displays are shared under the same license. Sources: https://www.forecastbench.org/datasets/ and https://github.com/forecastingresearch/forecastbench-datasets and https://creativecommons.org/licenses/by-sa/4.0/.
