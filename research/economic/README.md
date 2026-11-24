# Paired economic forecast study

This independent study evaluates ending credit spreads and maximum intraday VIX over 5 and 10 weekdays. The experiment uses public economic series, no LLM inference, no credentials, no tournament questions, no community forecasts, and no publication functionality. It needs Python 3.11+ and NumPy. Raw Cboe and ICE observations are retrieved from their public sources as allowed on the day of use and are not included in this repository.

## Reproduce

Run from this directory with the Python environment containing NumPy.

```sh
python fetch_inputs.py
python paired_economic.py
python credit_extension.py --fetch
python summarize_evidence.py
```

The manifest records every endpoint, its vintage and SHA-256 or the omission reason. The scripts validate input bytes, forecast before scoring, fit using development labels and produce summaries and individual outcomes locally. Vendor retention may make previously available historical vintages unavailable later. For the bit-identical source replay, the original research machine must retain the exact raw files listed in the manifests. Local output per forecast can include data subject to source licensing. Keep it private. Independent economic dates, targets and rules are fixed in `design.json` and `extension.json`.

## Reference and inference

The spread reference reproduces the original production numerical forecast component at width 1.25. VIX is an experimental shadow comparator at width 1.00. Running them through a common custom log score allows comparison on the same target date and underlying economic quantity. The production LLM gate, official Metaculus CDF and tournament peer scoring are outside this study. The optional parity probe compares 20 quantiles across four real as-of forecast targets. On 2026-11-24 it found zero numerical difference.

To check parity, supply the original `market_prices.py` from branch `market-pulse/26q4`, commit `ba21e72fa116b21c12bfcf41e81fc969a4262f74`.

```sh
python check_comparator.py /path/to/market_prices.py
```

A test observation pairs forecasts on the same date and horizon. Gains are averaged first across horizons, then across forecast-date blocks. Nonoverlapping quarterly blocks underpin the main study. Nonoverlapping four-week blocks underpin the shorter supplementary spread cohort. P-values are one-sided and paired, with Holm correction over C, F and G; sign tests assume block signs are exchangeable under the null. Bootstrap intervals resample whole blocks and remain sensitive to serial dependence across blocks.

VIX history is the exchange's retrospective OHLC archive, censored at the information date. Vendor corrections cannot be ruled out. Credit inputs use original ALFRED publication vintages. The finite-bin log score is not comparable in level with Metaculus results. These historical experiments do not qualify a model for a cash prize or authorize replacement of production.

## Data

- [Cboe VIX history](https://www.cboe.com/tradable-products/vix/vix-historical-data)
- [ALFRED ICE BofA US High Yield Index Option-Adjusted Spread and retention policy](https://alfred.stlouisfed.org/series?seid=BAMLH0A0HYM2)

## Observed results on 2026-11-24

- VIX, 96 paired held-out forecasts over 48 monthly origins and 16 full quarters. Calibration and routing retained their comparator. Statistical G changed mean log mass by −0.0897 nats per origin, with a 95% block interval [−0.2245, +0.0149] and Holm-adjusted p = 1.0.
- Supplemental credit, 54 paired held-out forecasts over 27 biweekly origins and 13 complete four-week blocks. C and F retained A. G changed mean log mass by −0.0246 nats per origin. The block estimand was −0.0267 [−0.1022, +0.0375], Holm-adjusted p = 1.0. G won 16 of the 27 origins; the larger losses outweighed those wins. The comparator's 80% interval covered 44/54 outcomes.
- Original credit schedule, only five paired monthly origins and a single complete quarter were recoverable. Confidence bounds are undefined for a single block. No inference from this cohort.
- No challenger cleared the archived screen. Production should remain unchanged.
