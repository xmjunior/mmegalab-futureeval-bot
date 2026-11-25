"""Score the single preregistered, CPU-only series reference class."""
import argparse
from collections import defaultdict
from datetime import date
import gzip
from itertools import product
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "forecastbench_shadow"))
import model as champion

ARMS = ("series", "champion", "neutral")


def fit(train):
    by_series = defaultdict(lambda: defaultdict(list))
    for phase, issued, source, series, target, direction, *tail in train:
        if phase != "train":
            continue
        if issued >= "2026-01-04" or target >= "2026-01-04":
            raise ValueError("Training source was immature at the cutoff")
        by_series[source, series][issued].append(tail[-2])
    return {key: (sum(mean(ys) for ys in rounds.values()), len(rounds)) for key, rounds in by_series.items()}


def predict(model, source, series_id, base):
    successes, n = model.get((source, series_id), (0., 0))
    return (successes + 4 * base) / (n + 4), bool(n)


def macro(values, groups):
    cells = defaultdict(list)
    for value, group in zip(values, groups, strict=True):
        cells[group].append(value)
    return {str(k): mean(v) for k, v in sorted(cells.items())}


def bootstrap(blocks):
    # Independent draws of complete blocks; report each grouping separately.
    values = np.array(list(blocks.values()))
    rng = np.random.default_rng(20261125)
    averages = []
    for _ in range(20):
        samples = rng.integers(len(values), size=(500, len(values)))
        averages.extend(values[samples].mean(axis=1))
    return np.quantile(averages,[.025,.975]).tolist()


def scores(named, y, arms):
    out = {}
    for name in arms:
        pairs = list(zip(named[name], y, strict=True))
        if not pairs:
            out[name] = None
            continue
        bins = defaultdict(list)
        for p, outcome in pairs:
            bins[min(9, int(p*10))].append(p-outcome)
        # Keep original probabilities for Brier/ECE. Only logarithms use epsilon.
        out[name] = {"n":len(pairs),
                     "brier":mean((p-outcome)**2 for p,outcome in pairs),
                     "log_loss":mean(-math.log(max(1e-6, p if outcome else 1-p)) for p,outcome in pairs),
                     "ece10":sum(abs(sum(v)) for v in bins.values())/len(pairs)}
    return out


def read_panel():
    manifest = json.loads((HERE / "inputs.json").read_text())
    raw = gzip.decompress((HERE / "panel.jsonl.gz").read_bytes())
    if hashlib.sha256(raw).hexdigest() != manifest["complete_rows_sha256"]:
        raise ValueError("Scoring panel checksum differs")
    return [json.loads(line) for line in raw.splitlines()]


def evaluate(records, external=None):
    prior_panel = champion.rows()
    model = fit(prior_panel)
    baseline = champion.fit(prior_panel)
    named, labels, dates, groups, known, sources, external_rows = defaultdict(list), [], [], [], [], [], []
    for issued, source, series_id, target, family, outcome in records:
        p, _ = champion.prediction(baseline, source, (date.fromisoformat(target) - date.fromisoformat(issued)).days)
        q, seen = predict(model, source, series_id, p)
        for name,val in zip(ARMS,(q,p,.5),strict=True):
            named[name].append(val)
        dates.append(issued)
        labels.append(outcome)
        known.append(seen)
        sources.append(source)
        groups.append(family)
        if external is not None:
            v=external.get((issued,source,series_id,target))
            if v is not None:
                external_rows.append((q,p,float(v),outcome))
    if not labels:
        raise ValueError("No eligible, resolved cases")
    result = scores(named, labels, ARMS)
    diff = [(p-y)**2-(q-y)**2 for p,q,y in zip(named["champion"],named["series"],labels,strict=True)]
    rg, fg = macro(diff, dates), macro(diff, groups)
    rci, fci = bootstrap(rg), bootstrap(fg)
    observed = sum(rg.values())
    exact = sum(sum(s*v for s,v in zip(signs,rg.values(),strict=True)) >= observed-1e-12
                for signs in product((-1,1),repeat=len(rg))) / 2**len(rg)
    subgroups = {}
    for name, indices in {
        **{s:[i for i,k in enumerate(sources) if k==s] for s in sorted(set(sources))},
        "seen_series":[i for i,v in enumerate(known) if v],
        "unseen_series":[i for i,v in enumerate(known) if not v],
    }.items():
        subgroups[name] = {"n":len(indices), "metrics":scores({arm:[named[arm][i] for i in indices] for arm in ARMS},
                                                               [labels[i] for i in indices], ARMS)}
    coverage = len(named["series"]) / len(labels)
    gate = {
        "coverage_at_least_97pct":coverage>=.97,
        "positive_macro_round_gain":mean(rg.values())>0,
        "positive_macro_family_gain":mean(fg.values())>0,
        "round_lower95_positive":rci[0]>0,"family_lower95_positive":fci[0]>0,
        "exact_round_p_lt_05":exact<.05,
        "mean_log_loss_not_worse":result["series"]["log_loss"]<=result["champion"]["log_loss"],
        "ece10_not_worse":result["series"]["ece10"]<=result["champion"]["ece10"],
    }
    doc = {"rows":len(labels),"families":len(fg),"rounds":len(rg),"scored_case_coverage":coverage,
           "metrics":result,"paired_brier_gain":mean(diff),
           "mean_round_brier_gain":mean(rg.values()),"mean_family_brier_gain":mean(fg.values()),
           "round_bootstrap95":rci,"family_bootstrap95":fci,"round_signflip_p":exact,
           "gains_by_round":rg,"counts_by_round":{d:dates.count(d) for d in rg},"subgroups":subgroups,"shadow_gate":gate,"shadow_pass":all(gate.values()),
           "archive_comparator":{"status":"not supplied","n":0},
           "model_api_calls":0,"new_cost_usd":0}
    if external is not None:
        cols=list(zip(*external_rows)) if external_rows else ([],[],[],[])
        doc["archive_comparator"]={"status":"paired official archived probabilities","n":len(external_rows),
                                  "coverage":len(external_rows)/len(labels),
                                  "metrics":scores(dict(zip(("series","champion","archived_gpt5mini"),cols[:3])),
                                                   cols[3],("series","champion","archived_gpt5mini"))}
    return doc


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--external",type=Path, help="Strictly matched, pinned archive extraction")
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    ext=None
    if args.external:
        doc=json.loads(args.external.read_bytes())
        ext={tuple(row[:4]):row[4] for row in doc["rows"]}
    result=evaluate(read_panel(), ext)
    args.out.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in ("rows","rounds","families","metrics","shadow_pass")}))
