# ECB evidence and resolution tests

The archive contains 48 official policy release editions in English, Spanish and Portuguese from 16 scheduled meetings. The full calendar year 2024 supplies the reference period. All eight meetings in 2025 form the retrospective holdout. Each holdout meeting supplies three editions. Its 276 possible pairs share those 24 texts and eight events.

| Method on 2025 | Translation pairs found of 24 | Different meetings incorrectly merged of 252 |
|---|---:|---:|
| Exact URL | 0 | 0 |
| Normalized compression distance ≤ 0.35 | 0 | 23 |
| Five-token Jaccard ≥ 0.80 | 0 | 0 |
| Verified ECB release ID and matching numerical rate tuple | 24 | 0 |

Five incorrect compression merges joined releases with different deposit rates. A one-digit adversarial mutation produced a distance of 0.0374. An update veto at 0.35 would suppress that change. When two editions share a trusted release ID but disagree on rates, the implemented rule retains a conflict. Matching publisher IDs are specific to the ECB website. Success on this corpus gives no multilingual web-wide guarantee.

For the deposit rate in force a day after each 2025 meeting, applying the announced rate immediately gave 4/8 correct. Reading the effective date gave 8/8. The official rate history supplies the answer for each exact date. All four cuts took effect after the test day. These are document and contract checks. The retrospective corpus contains no sealed probabilities.

## Prospective paired dossier experiment

On 24 November 2026 both methods issued probabilities for the first three future scheduled ECB decisions. Each event asks whether the ECB deposit facility rate seven calendar days after the scheduled meeting is strictly lower than the rate one calendar day before it. The definition covers unscheduled rate action inside the same window. Equality resolves false.

The simple dossier uses all 16 meetings in 2024–25 with eight cuts. The challenger adds the seven completed 2026 meetings. That deliberately sought contrary evidence contains two increases and zero cuts. Both use the same Laplace-smoothed empirical frequency. Their frozen probabilities are 0.50 and 0.36 respectively for each event. This is a controlled comparison of two fixed evidence sets with one deterministic calculation. It does not measure the causal effect of an LLM critic. The estimates are correlated across meetings and the three cases alone cannot establish superiority. Marginal model and retrieval spend was zero.

| Meeting | Exact before/after observations | First scoring check, UTC |
|---|---|---|
| 17 December 2026 | 16 and 24 December | 25 December at 12:00 |
| 4 February 2027 | 3 and 11 February | 12 February at 12:00 |
| 18 March 2027 | 17 and 25 March | 26 March at 12:00 |

The frozen inputs contain 46 exact-day official readings. The seal reproduces from those inputs. The resolver accepts the first retrieved pair of actual observations in the exact ECB series, stores the time and hash of the response, and scores Brier and natural-log loss once per arm. Data can arrive later than the first check. Unavailable or provisional readings keep events pending. A successful delayed run records its own first retrieval, which can be later than original publication. Later revisions are excluded after settlement. Incomplete independent coverage and three related events preclude promotion.

## Reproduce

Run from this directory with standard Python 3.11 or later.

```sh
python experiment.py
python -m unittest discover -s . -p 'test_*.py' -v
python prospective.py verify
python prospective.py settle --state /tmp/megalab-ecb-receipts.json
```

The default historical replay uses only the frozen numerical measurements. To audit upstream source drift, run `python experiment.py --fetch /tmp/megalab-ecb-html` with a new cache directory. The published URL and normalized-text hash of each release appear in `measurements.json`. The sealed input file includes the calendar, endpoint and exact days for the forward study. No online lookups occur in the prospective scorer before a dated observation is due.

[Official ECB rate history](https://www.ecb.europa.eu/stats/policy_and_exchange_rates/key_ecb_interest_rates/html/index.en.html) · [Official ECB daily series](https://data.ecb.europa.eu/data/datasets/FM/FM.D.U2.EUR.4F.KR.DFR.LEV) · [Official ECB meeting calendar](https://www.ecb.europa.eu/press/calendars/mgcgc/html/index.en.html).
