"""Frozen monthly Treasury comparison. Each forecast receives only its own historical vintage."""
import argparse
import hashlib
import json
import math
import random
import statistics
import time
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from alfred_vintage import fred_vintage


def series_as_of(day, start):
    data = []
    for name in ("DGS2", "DGS10"):
        time.sleep(0.75)  # Keep the serial job comfortably below the official API rate limit.
        data.append(fred_vintage(name, day.isoformat(), start.isoformat()))
    overlap = sorted(data[0].keys() & data[1].keys())
    return [(date.fromisoformat(k), data[0][k], data[1][k] - data[0][k]) for k in overlap]


def forecast(history, origin):
    rows = history[-252:]
    if len(rows) < 240 or rows[-1][0] >= origin or origin - rows[-1][0] > timedelta(days=7):
        raise ValueError("incomplete_asof_history")
    calls = {}
    for leg, offset in [(1,Decimal("0.02")), (2,Decimal("0.01"))]:
        v = [row[leg] for row in rows]
        moves = [float(b-a) for a,b in zip(v,v[1:])]
        full = statistics.pstdev(moves)
        ratio = min(2, max(0.5, statistics.pstdev(moves[-20:]) / full)) if full else 1
        for lag in (7,12):
            sample = [float(v[i]-v[i-lag]) for i in range(len(v)-1,lag-1,-lag)]
            empirical = (2+sum(d > float(offset) for d in sample))/(4+len(sample))
            scaled = (2+sum(ratio*d > float(offset) for d in sample))/(4+len(sample))
            calls[leg,lag] = (v[-1]+offset, empirical, scaled)
    return calls


def interval(gains):
    # Paired moving-block bootstrap; preserve cross-leg dependence inside month.
    n = len(gains)
    r = random.Random(20261124)
    samples = []
    for _ in range(5000):
        rolled = []
        while len(rolled) < n:
            i = r.randrange(n)
            rolled.extend(gains[(i+j)%n] for j in range(4))
        samples.append(statistics.mean(rolled[:n]))
    samples.sort()
    return [round(samples[int(p * (len(samples)-1))],6) for p in (0.025,0.975)]


def study():
    cases, groups = [], []
    for year in range(2023,2027):
        for month in range(1,13):
            origin = date(year,month,1)
            if origin > date(2026,10,1):
                break
            prev = origin-timedelta(days=1)
            forecasts = forecast(series_as_of(prev, prev-timedelta(days=430)),origin)
            # No output-side observation is requested until all four forecasts are fixed.
            after = series_as_of(origin+timedelta(days=40),origin+timedelta(days=1))
            if len(after)<12:
                raise ValueError("outcomes_unavailable")
            paired = []
            for (leg,lag),(threshold,reference,scaled) in forecasts.items():
                day, two, spread = after[lag-1]
                if day<=origin:
                    raise ValueError("overlapping_input_and_outcome")
                y = int((two if leg==1 else spread)>threshold)
                paired.append((reference-y)**2-(scaled-y)**2)
                cases.append(dict(origin=origin.isoformat(), target=day.isoformat(),
                                  leg="2y" if leg==1 else "spread", lag=lag, threshold=str(threshold),
                                  reference=reference, scaled=scaled, outcome=y))
            groups.append(statistics.mean(paired))
    if len(groups)!=46 or len(cases)!=184:
        raise ValueError("unexpected_sample")
    def losses(key):
        bs = statistics.mean((r[key]-r["outcome"])**2 for r in cases)
        ll = -statistics.mean(math.log(r[key] if r["outcome"] else 1-r[key]) for r in cases)
        return dict(brier=round(bs,6),log_loss=round(ll,6))
    return dict(months=len(groups), pairs=len(cases), reference=losses("reference"),scaled=losses("scaled"),
                paired_brier_gain=round(statistics.mean(groups),6), ci95_month_blocks=interval(groups),
                improved_months=sum(g>0 for g in groups), tied_months=sum(g==0 for g in groups),
                cases=cases)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("output",type=Path)
    args=p.parse_args()
    results=study()
    args.output.write_text(json.dumps(results, sort_keys=True,indent=2)+"\n")
    print(json.dumps({k:v for k,v in results.items() if k!="cases"}, sort_keys=True))
    print("sha256="+hashlib.sha256(args.output.read_bytes()).hexdigest())
