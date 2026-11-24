# Prospective Treasury forecast register

Run from the pinned research checkout using Python 3.11+ and standard library.

```sh
python -m unittest discover -s research -p test_shadow.py -v
python research/shadow.py --ledger /path/to/observations/ledger.json
```

GitHub Actions supplies the optional GEMINI_API_KEY and OPENROUTER_API_KEY as secrets. The ledger belongs on a separate, durable observations branch, with a single serialized writer and the research revision recorded by the scheduling workflow. On recovery, restore that branch before running. Starting an empty ledger would lose the first externally timestamped forecasts. No Metaculus token is used. The reported cases are original Treasury questions. Competition scores and eligibility come from the separate competition audit.

Each ISO week gets at most one new question, at the actual first run of that week. Horizons rotate over 3, 7 and 14 calendar days. The threshold alternates 5 basis points above/below the last prior-date Treasury 10-year par rate. The first Treasury observation in the seven-day target window settles the binary event, with a two-day publication delay. Strict inequality handles ties. The first observed resolution is locked, including its source digest. Dates of already-missed weeks are never reconstructed. The input evidence freezes on first issue. All arms must issue within that same UTC calendar date. Same-day retries can see newer world knowledge from a provider, so their exact timestamps remain available for comparison. If a scheduled check misses a week, coverage falls.

| Arm | Forecast procedure |
|---|---|
| champion | Uses both production research and binary prompt literals taken from the SHA-256-verified main.py at 0459814720cd51c3effa7cfa412cec8aefcdb105. Calls Google Gemma 4 31B then 26B if necessary. Same temperature, final probability syntax, integer-list clarification (when budget permits), and 1–99% clamp. For a clean model-family contrast this research comparator omits the production pool's opaque OpenRouter router and never invokes CPU. It uses direct HTTP, 1,536-token output cap, one bounded retry policy, and synthetic site-independent question fields. It is a specified reproduction of the Google path, not the complete live bot or official site results. |
| A | Prespecified challenger NVIDIA Nemotron 3 Ultra 550B via the named zero-price OpenRouter endpoint, using the two production prompts and identical parsing. Its strength versus Gemma remains an empirical question. |
| B | Same challenger as A, with the research paragraph replaced by official Treasury curve and as-of historical change evidence plus the NY Fed SOFR observation when available. Source links, retrieval times and hashes live in the ledger. AskNews is disabled pending evidence of an entitled account with a hard zero-cost quota. |
| C | Empirical reference probability from non-overlapping changes and Beta(2,2) shrinkage. |
| D | Equal-weight linear pool when both Google Gemma and NVIDIA Nemotron return valid predictions. Independence refers to model families; statistical error correlation can remain. |
| E | Measures the absolute disagreement of the two families; records a 20-percentage-point trigger. Makes at most one adjudication request per case to a named :free variant of Claude Opus 5, GPT-5.6 Sol or Kimi K3, in that order, only if the live catalogue attests zero on every price component. No listed free variant qualified on 24 November. A paid model with a credit balance cannot enable this arm. A valid forecast requires a zero-cost receipt. |
| F | Fixed economic binary router: use C with at least 20 analogues and horizon up to 7 days; otherwise use B. If its chosen arm is unavailable it abstains. Numerical and other formats are evaluated separately. |
| coin | Fixed 0.5 uninformative comparator. |

The ledger logs successful forecasts, missing arms, model and prompt/output hashes, wall time per request, token usage when reported, and source hashes. The first probability per arm and case is immutable. Free-form hosted output only covers public, self-authored economic cases. The shares of missing cases, paired Brier improvement and log loss in nats stay distinct. Calibration-in-the-large is mean forecast probability minus event rate. This single-series binary test does not establish general FutureEval, MiniBench or Market Pulse uplift. Precision about statistical significance requires enough distinct resolved time blocks. No automatic promotion is implemented.

With secrets available the lab permits at most three calls per named model on issue day, including research, repair, and retries. Successful calls and rate-limit cooldowns persist across reruns. HTTP 429/503/timeout apply a one-hour minimum cooldown, or longer at provider request. No retries take place after the original UTC issue date. Named Google Gemma routes require their current official pricing table to say both inputs and outputs are free with paid-tier inference unavailable. The OpenRouter route requires the specific :free ID, all catalog pricing fields equal to zero, and zero per-request, prompt and completion price ceilings. Credentialed requests refuse redirects. No billing or account modifications are made.

Sources: [US Treasury daily rates](https://home.treasury.gov/resource-center/data-chart-center/interest-rates), [New York Fed SOFR](https://www.newyorkfed.org/markets/reference-rates/sofr), [Google AI Studio pricing](https://aistudio.google.com/docs/pricing), [OpenRouter live catalogue](https://openrouter.ai/api/v1/models), [OpenRouter price caps](https://openrouter.ai/docs/guides/routing/provider-selection#max-price).
