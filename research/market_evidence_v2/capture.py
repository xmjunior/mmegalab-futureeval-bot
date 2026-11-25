"""Capture derived blind packets from archived ALFRED releases; labels stay separate."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
from unittest.mock import patch

import market_prices as prices
from market_pulse import options_for, reviewer_prompt
from evidence import grounded_row


HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / "protocol.json").read_bytes())
UTC = timezone.utc
SERIES = prices.SPECS[46010][1]
Q4_HASHES = {46212: '4d702a46f2615ade2b77c316feaf553c8064ae1bbe99aaebaa70a02f36fbbd8c', 46213: 'f31a0dc3e0a526ffca3aaec5e80d879a6531e3a5c3cc35a73d19626708f5fd0c', 46709: 'f51037ad3daec5a20f4ca66ca1106af7d86d3cec8f661f61191a6f66f2b7c648'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def vintage(cutoff, cache=None, expected=None):
    url = "https://alfred.stlouisfed.org/graph/alfredgraph.csv?" + prices.urlencode(
        {"id": SERIES, "cosd": (cutoff - timedelta(days=810)).isoformat(), "vintage_date": cutoff.isoformat()})
    raw = cache.read_bytes() if cache and cache.exists() else prices._get(url)
    digest = sha(raw)
    if expected and digest != expected:
        raise ValueError("Frozen ALFRED release changed")
    rows = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    field = SERIES + "_" + cutoff.strftime("%Y%m%d")
    if rows.fieldnames != ["observation_date", field]:
        raise ValueError("Unexpected ALFRED release schema")
    values = {date.fromisoformat(r["observation_date"]): float(r[field]) for r in rows if r[field] not in {"", "."}}
    if not values or max(values) > cutoff:
        raise ValueError("Release contains invalid observation dates")
    return url, digest, values


def windows():
    for h in prices.HISTORY:
        if h["post_id"] == 46010:
            yield str(h["question_id"]), h["spot"], h["resolve"], "diagnostic"
    for s in PROTOCOL["reserve_spots_utc"]:
        start = date.fromisoformat(s)
        spot = datetime.combine(start, time(3), UTC)
        end = datetime.combine(start + timedelta(days=11), time(16), prices.NY).astimezone(UTC)
        yield s, spot, end, "reserve"


def packet(spec, cache):
    key, spot, end, group = spec
    asof = spot - timedelta(seconds=1)
    cutoff = asof.astimezone(prices.NY).date() - timedelta(days=1)
    replay_key = (46010, *(dt.isoformat() for dt in (asof, spot, end)))
    archive = prices._FRED_REPLAY.get(replay_key)
    url, digest, obs = vintage(cutoff, cache / (key + ".csv") if cache else None, archive[1] if archive else None)
    with patch.object(prices, "_fred", return_value=(obs, url)):
        est = prices.model(46010, asof, spot, end)
    control = options_for(est, [1.25], asof, end)
    candidate = grounded_row(control, obs, asof, spot, end, est)
    sealed = {arm:sha(reviewer_prompt([row]).encode()) for arm,row in (("control",control),("candidate",candidate))}
    if group == "diagnostic" and sealed["control"] != Q4_HASHES[int(key)]:
        raise ValueError("Control prompt differs from the originally tested contract")
    return {"key":key, "group":group, "cutoff":asof.isoformat(),"source":url,"source_sha256":digest,
            "target":end.isoformat(),"control":control,"candidate":candidate,"prompt_sha256":sealed,
            "naive_quantiles":list(map(float, est.naive_quantiles))}


def reserve_label(spec):
    key, _, end, _ = spec
    target = end.astimezone(prices.NY).date()
    url, digest, values = vintage(target + timedelta(days=7))
    latest = max(d for d in values if d <= target)
    if (target - latest).days > 4:
        raise ValueError("No timely reserve settlement")
    return key, {"value":values[latest], "observed":latest.isoformat(),"release":url,"sha256":digest}


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["freeze-inputs","reserve-labels"])
    p.add_argument("--cache",type=Path)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    specs=list(windows())
    if any(s[2] >= datetime.now(UTC) for s in specs):
        raise ValueError("Only resolved historical windows may be evaluated")
    if args.mode=="freeze-inputs":
        # The model input binding is process-global; construct packets serially.
        result=[packet(s,args.cache) for s in specs]
    else:
        with ThreadPoolExecutor(max_workers=3) as pool:
            result=dict(pool.map(reserve_label,(s for s in specs if s[-1]=="reserve")))
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
