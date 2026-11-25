# Dated NHC intensity guidance

This research connector retrieves the National Hurricane Center forecast discussion explicitly cited in a wind question. The returned packet identifies the issuing center, storm, advisory number, issue time, forecast valid time, wind in knots, publisher URL and content hash. It abstains on ambiguous or corrected editions and on dates outside the issue-to-target window. The issue timestamp plus one hour approximates historical availability. A digital receipt from the original publication time was unavailable.

The season test used the first discussion of each numbered Atlantic system found in the official 2025 best-track index. Four discussions failed the conservative correction check. Nine provided comparable winds. Four bulletins triggered the conservative text check for a correction. This filter can also match a correction discussed in the prose, and the four excluded records have no independent revision timestamp in the result. The official 24-hour guidance had a mean absolute error of 5.00 kt versus 8.89 kt for persistence of the initial wind. It won five times, tied twice and lost twice. The one-sided sign test gave 0.2265625, above the frozen threshold of 0.05. The connector remains in research. This experiment measured point forecasts of wind. An improvement in MEGALAB tournament probability scores remains unmeasured.

The code uses the raw bulletin and cutoff to construct each packet. The scoring path separately obtains the later official best track for the exact valid hour. The intake path accepts one explicit HTTPS bulletin link on the NHC publisher's host. Missing, ambiguous and expired source data result in abstention. Historical ingestion captured the archive editions available at collection time. The public byte hashes and dated links let a later auditor detect drift.

Reproduce the local regression and the frozen comparison from this directory.

```sh
python -m unittest -v test_evidence
python -c 'import json; from evaluate import stats; s=json.load(open("season-result.json")); print(stats(s["rows"], s["technical_retrieval_complete"]))'
```

[Frozen protocol](https://github.com/xmjunior/mmegalab-futureeval-bot/commit/9b8c8ba953c9d89fddcffd850a2176f816b9819b) · [Initial scientific receipt](https://github.com/xmjunior/mmegalab-futureeval-bot/actions/runs/45626565582) · [Official season index](https://ftp.nhc.noaa.gov/atcf/archive/2025/)

The authenticated resolved-questions feed returned 52 distinct IDs on 25 November. Neither its list nor the matching post-detail records yielded a string in the SDK's resolution-criteria field for these questions. The eligible proportion is unknown. None of the five already closed submissions from the prior participation audit matched these 52 IDs. No tournament probability score can be inferred from this sample. See `applicability-result.json` and [the original receipt](https://github.com/xmjunior/mmegalab-futureeval-bot/actions/runs/45626979764).
