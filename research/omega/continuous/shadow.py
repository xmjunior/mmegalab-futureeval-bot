"""Issue a fixed, independent unemployment panel once; score only its first releases."""
import argparse
import csv
import hashlib
import io
import json
from datetime import date, datetime, timedelta, timezone
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import evaluate as e

VINTAGE = date(2026, 11, 23)
ISSUE_BY = date(2026, 11, 26)
FROZEN = e.HERE / "shadow_20261124.json"
HISTORY = e.HERE / "shadow_input_20261123.csv"


def historical_input():
    params = dict(id="UNRATE", vintage_date=str(VINTAGE), cosd="2015-01-01", coed=str(VINTAGE))
    url = "https://alfred.stlouisfed.org/graph/alfredgraph.csv?" + urlencode(params)
    with urlopen(url, timeout=60) as r:
        raw = r.read()
    rows = csv.reader(io.StringIO(raw.decode("utf-8-sig")))
    if next(rows) != ["observation_date", "UNRATE_"+VINTAGE.strftime("%Y%m%d")]:
        raise ValueError("Unexpected official vintage")
    hist = {date.fromisoformat(k):float(v) for k,v in rows if v and v!="."}
    last = max(hist)
    if last > VINTAGE or VINTAGE-last > timedelta(days=62):
        raise ValueError("Stale or future data")
    days = [e.month_shift(last,k) for k in range(-119,1)]
    missing = [d for d in days if d not in hist]
    return raw, url, last, missing, hist, days


def issue():
    if FROZEN.exists():
        print("Already sealed; no new forecasts.")
        return
    now = datetime.now(timezone.utc)
    if not VINTAGE < now.date() < ISSUE_BY:
        raise ValueError("Issue window closed")
    raw, url, last, missing, hist, days = historical_input()
    seal = dict(issued_utc=now.isoformat(), earliest_information_cutoff=str(VINTAGE),
                source=url, source_sha256=hashlib.sha256(raw).hexdigest(), last_observation=str(last),
                input_requirement="120 consecutive monthly observations; abstain if incomplete",
                primary_comparator="champion_rule", resolution="ALFRED first public release", questions=[],
                cost_usd=0, missing_months=[str(d) for d in missing])
    if missing:
        seal["status"] = "abstained_incomplete_history"
    else:
        expansion=e.calibration(e.cohort(e.archive()))
        values=np.array([hist[d] for d in days])
        for h in e.HORIZONS:
            target=e.month_shift(last,h)
            if e.month_shift(target,1) <= now.date():
                raise ValueError("Target could have been released")
            question=dict(target_month=str(target), horizon=h, threshold=round(values[-1]+.2,2), forecasts={})
            for name,d in e.arms(values,h,expansion[h]).items():
                question["forecasts"][name]=dict(center=d.center,scale=d.scale,mix=d.mix,
                                                  knots=d.q.tolist() if d.q is not None else None,
                                                  quantiles=d.quantile(e.P).tolist())
            seal["questions"].append(question)
        seal["status"] = "sealed_awaiting_first_release"
    with HISTORY.open("xb") as f:
        f.write(raw)
    with FROZEN.open("x") as f:
        json.dump(seal,f,sort_keys=True,indent=2)
        f.write("\n")
    print(seal["status"], len(seal["questions"]), seal["missing_months"])


GAP_SEAL = e.HERE / "shadow_gap_20261124.json"


def gap_arms(history, h):
    if len(history) != 120 or np.count_nonzero(~np.isfinite(history)) > 1 or not np.all(np.isfinite(history[-2:])):
        raise ValueError("Gap cohort is ineligible")
    diffs = history[h:] - history[:-h]
    moves = diffs[np.isfinite(diffs)]
    daily = np.diff(history)
    observed = daily[np.isfinite(daily)]
    recent = daily[-30:][np.isfinite(daily[-30:])]
    sd = float(np.std(observed, ddof=1))
    multiplier = float(np.clip(np.sqrt(np.std(recent, ddof=1)/sd), .7, 2.)) if sd else 1.
    center = history[-1]
    q = np.quantile(moves - np.median(moves), e.P)
    return {"empirical_gaps": e.Distribution(center, sd, center+q),
            "champion_rule_gaps": e.Distribution(center, sd, center+multiplier*q)}


def issue_gap():
    if GAP_SEAL.exists():
        print("Gap cohort already sealed.")
        return
    now = datetime.now(timezone.utc)
    if not VINTAGE < now.date() < ISSUE_BY:
        raise ValueError("Issue window closed")
    seal = json.loads(FROZEN.read_text())
    if hashlib.sha256(HISTORY.read_bytes()).hexdigest() != seal["source_sha256"]:
        raise ValueError("Input changed")
    raw = csv.reader(io.StringIO(HISTORY.read_text(encoding="utf-8-sig")))
    if next(raw) != ["observation_date", "UNRATE_"+VINTAGE.strftime("%Y%m%d")]:
        raise ValueError("Wrong vintage")
    series = {date.fromisoformat(k):float(v) for k,v in raw if v and v!="."}
    last = max(series)
    if str(last) != seal["last_observation"] or last > VINTAGE:
        raise ValueError("Invalid anchor")
    hist = np.array([series.get(e.month_shift(last,k), np.nan) for k in range(-119,1)])
    questions=[]
    for h in e.HORIZONS:
        target=e.month_shift(last,h)
        if e.month_shift(target,1) <= now.date():
            raise ValueError("Late issue")
        forecasts = {k:dict(center=d.center, scale=d.scale, mix=d.mix,
                            knots=d.q.tolist(), quantiles=d.quantile(e.P).tolist())
                     for k,d in gap_arms(hist,h).items()}
        questions.append(dict(target_month=str(target), horizon=h, threshold=round(hist[-1]+.2,2), forecasts=forecasts))
    record=dict(issued_utc=now.isoformat(), source=seal["source"], source_sha256=seal["source_sha256"],
                prereg_sha256=hashlib.sha256((e.HERE/"shadow_gap_protocol.json").read_bytes()).hexdigest(),
                vintage=str(VINTAGE), missing_months=seal["missing_months"], questions=questions,
                status="sealed_awaiting_first_release", cost_usd=0)
    with GAP_SEAL.open("x") as f:
        json.dump(record,f,sort_keys=True,indent=2)
        f.write("\n")
    print(len(questions), "future targets sealed")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["issue","issue-gap"])
    args=parser.parse_args()
    {"issue":issue, "issue-gap":issue_gap}[args.mode]()
