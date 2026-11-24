# MEGALAB ARENA

This offline arena compares frozen probabilities on the research lab's independent US Treasury panel. It reuses the original settlement book and costs zero additional inference. Each model saw the empirical Treasury reference in its original prompt. Its separate submission is a model arm; errors on shared cases can be correlated.

## Historical comparison

The input is the 184 already published Treasury vintage records across 46 monthly origins. Each forecast used the ALFRED vintage preceding its origin; each outcome was read from the vintage 40 days after that origin (see `research/treasury_paired.py`). The new affine calibration fit used 92 records whose **outcome vintage** preceded 1 January 2025. December 2024 is embargoed. The fixed fit was evaluated without updates on all 88 matched events in the subsequent 22 months, through October 2026. Original algorithms had already been studied on this archive. This is a retrospective comparison, with the prospective panel providing separate evidence.

| Method | Brier | Natural log loss |
|---|---:|---:|
| Constant 0.5 | 0.250000 | 0.693147 |
| Calibrated empirical | 0.259159 | 0.712128 |
| Empirical reference | 0.264654 | 0.724324 |
| Equal empirical / volatility pool | 0.265488 | 0.725984 |
| Volatility-scaled empirical | 0.266922 | 0.728952 |

The fitted correction uses least squares with fixed identity shrinkage, then clips to [0.01, 0.99]. Against the empirical method, the Brier gain was +0.005495 with a 95% circular four-month block bootstrap interval of [−0.003954, +0.014949]. Adjacent monthly or within-month forecasts are dependent. The constant forecast had the lowest loss here. No model is promoted. The JSON includes each method's calibration-in-the-large, fixed-bin expected calibration error, bin sizes, and monthly paired changes.

## Prospective comparison

The four Treasury questions target 2 and 9 December 2026. They share series, dates and evidence. The three completed archival model arms are a Codex Work session (API snapshot unknown), Gemini web 4 Neon and GPT-5.6 Sol (one four-question response). ARENA also seals five methods: constant 0.5, the registered empirical reference, its historically calibrated version fitted to all outcomes available at issue, equal-weight mean of the three model arms and a 50/50 mean of that pool and the empirical reference. No inference was requested for this comparison. Missing provider responses are excluded and receive no scores. The constituent model probabilities were available when the ensemble was defined. Outcomes were still in the future.

The original records are bound to the research branch panel and the ledger at commit `ad6ccf32b1ec86e5f6a5d95489bd0491b0fe948c`. The eight-arm payload was sealed before the issue deadline with SHA-256 `32c886ed36e4e1f3c9c9860fcb7fc5db5df805bda9dca4f17b17fb74c43d1fde`. Its full probabilities, fit, timestamp and input digests are in `prospective.json`. The Git commit provides an external timestamp. Publication on GitHub must precede the target dates for prospective interpretation. Do not reissue a completed panel.

The scorer consumes only the existing settlement job's book for this panel. It verifies the sealed payload and constituent forecasts, applies the registered resolution window and publication lag, then reports Brier and log loss on identical cases for all eight arms. Pending and void outcomes never enter either denominator. Four correlated events cannot support a general winner.

## Reproduction

From a checkout of the research revision, Python 3.11 or later and the standard library suffice.

```sh
python -m unittest discover -s research/arena -p 'test_*.py' -v
python research/arena/arena.py backtest
python research/arena/arena.py score state/promo-score.json
```

For an independent replay of the archived inputs before scoring, use the original settlement program. It queries the US Treasury only once outcomes can be published and reads the already-issued forecasts. Preserve the original state on retries.

```sh
git show ad6ccf32b1ec86e5f6a5d95489bd0491b0fe948c:promo-ledger.json > /tmp/arena-source-ledger.json
python research/promo/settle.py --state state/promo-score.json --journal /tmp/arena-source-ledger.json
```
