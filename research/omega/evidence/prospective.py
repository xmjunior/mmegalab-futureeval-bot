"""Sealed ECB research-only forecasts and first-observed, append-only scoring."""
import argparse
import csv
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import io
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from sources import RELEASES

ROOT = Path(__file__).resolve().parent
KEY = "FM.D.U2.EUR.4F.KR.DFR.LEV"
BASE = "https://data-api.ecb.europa.eu/service/data/FM/D.U2.EUR.4F.KR.DFR.LEV"
CALENDAR_2026 = "https://www.ecb.europa.eu/press/pr/date/2025/html/ecb.pr250424~2643533670.en.html"
CALENDAR = "https://www.ecb.europa.eu/press/calendars/mgcgc/html/index.en.html"
HISTORICAL_2026 = "02-05 03-19 04-30 06-11 07-23 09-10 10-29".split()
MEETINGS = ("2026-12-17", "2027-02-04", "2027-03-18")
TRAINING = ROOT / "prospective_inputs.json"
SEAL = ROOT / "prospective_seal.json"


def digest(value):
    return hashlib.sha256(value).hexdigest()


def encode(obj):
    return (json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def get_csv(start, end):
    query = urlencode({"startPeriod": start, "endPeriod": end, "format": "csvdata"})
    url = BASE + "?" + query
    req = Request(url, headers={"User-Agent": "MEGALAB ECB public data research"})
    with urlopen(req, timeout=45) as response:
        if response.url != url:
            raise ValueError("Unexpected ECB data endpoint redirect")
        return response.read()


def observations(raw, needed):
    """Accept only unique, officially observed values for each exact requested day."""
    result = {}
    for row in csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))):
        day = row.get("TIME_PERIOD")
        if day not in needed:
            continue
        if row.get("KEY") != KEY or row.get("UNIT") != "PCPA" or row.get("UNIT_MULT") != "0":
            raise ValueError("Wrong ECB series, unit or multiplier")
        if row.get("OBS_STATUS") != "A" or not row.get("OBS_VALUE"):
            continue
        rate = Decimal(row["OBS_VALUE"])
        if not rate.is_finite() or day in result:
            raise ValueError("Duplicate or invalid ECB observation")
        result[day] = str(rate)
    return result


def window(meeting):
    day = date.fromisoformat(meeting)
    return ((day - timedelta(days=1)).isoformat(), (day + timedelta(days=7)).isoformat())


def collect_inputs():
    dates = [datetime.strptime(release[:6], "%y%m%d").date().isoformat() for release in RELEASES]
    dates += ["2026-" + m for m in HISTORICAL_2026]
    required = {t for m in dates for t in window(m)}
    raw = get_csv(min(required), max(required))
    values = observations(raw, required)
    if values.keys() != required:
        raise ValueError("Historical observations incomplete")
    return {"series": KEY, "source": BASE, "unit": "percent per annum",
            "calendar_sources": [CALENDAR_2026, CALENDAR],
            "source_response_sha256": digest(raw), "measurements": [
                {"meeting": m, "before_date": window(m)[0], "after_date": window(m)[1],
                 "before_rate": values[window(m)[0]], "after_rate": values[window(m)[1]]}
                for m in dates]}


def estimates(rows):
    result = {}
    for name, cohort in (
        ("two_year_dossier", [r for r in rows if r["meeting"] < "2026-01-01"]),
        ("counterevidence_dossier", rows),
    ):
        cuts = sum(Decimal(r["after_rate"]) < Decimal(r["before_rate"]) for r in cohort)
        result[name] = {"meetings": len(cohort), "cuts": cuts,
                        "p_cut": (1 + cuts) / (2 + len(cohort))}
    return result


def create_seal():
    raw = TRAINING.read_bytes()
    training = json.loads(raw)
    rows = training["measurements"]
    probabilities = estimates(rows)
    cases = []
    for meeting in MEETINGS:
        before, after = window(meeting)
        first_fetch = date.fromisoformat(after) + timedelta(days=1)
        cases.append({"meeting": meeting, "before": before, "after": after,
                      "first_fetch_utc": first_fetch.isoformat() + "T12:00:00+00:00",
                      "p_cut": {k: v["p_cut"] for k, v in probabilities.items()}})
    return {"frozen_on_utc": "2026-11-24", "event_family": "ECB deposit facility easing",
            "criterion": "Daily official deposit rate on meeting+7 calendar days strictly below meeting-1. Equality is false. Unscheduled action inside this fixed window counts. Use actual observations for both exact dates.",
            "resolution": "First successful retrieval of both actual official ECB observations on or after first_fetch_utc. Preserve the retrieval time and values. Missing data stays pending. Later source revisions are not used to rescore.",
            "source": BASE, "series": KEY, "input_sha256": digest(raw),
            "last_training_observation": max(r["after_date"] for r in rows),
            "arm_design": "Both arms apply the same Beta(1,1) smoothed empirical cut frequency. Simple uses all 2024–25 scheduled meetings. Counterevidence adds all seven completed 2026 meetings through October, including the new tightening evidence. No event-specific tuning.",
            "estimates": probabilities, "events": cases, "marginal_cost_usd": 0,
            "promotion": "Exploratory correlated event family, no promotion on these three alone."}


def validate_state(state, forecast, seal_hash):
    if state.get("seal_sha256") != seal_hash or not isinstance(state.get("settled"), dict):
        raise ValueError("Settlement file is unrelated to the frozen forecasts")
    cases = {c["meeting"]: c for c in forecast["events"]}
    expected_status = "complete" if state["settled"].keys() == cases.keys() else "waiting"
    if state.get("status") != expected_status:
        raise ValueError("Inconsistent settlement status")
    for key, receipt in state["settled"].items():
        case = cases[key]
        raw_rates = receipt["official_rates"]
        if set(raw_rates) != {case["before"], case["after"]}:
            raise ValueError("Settlement date mismatch")
        reproduced = score(case, raw_rates, receipt["retrieved_at"], receipt["response_sha256"])
        if reproduced != receipt:
            raise ValueError("Inconsistent pre-existing settlement")


def score(case, values, when, raw_hash):
    if datetime.fromisoformat(when) < datetime.fromisoformat(case["first_fetch_utc"]):
        raise ValueError("Settlement attempted before evidence could be observed")
    if len(raw_hash) != 64 or any(c not in "0123456789abcdef" for c in raw_hash):
        raise ValueError("Invalid source hash")
    rates = [Decimal(values[case[key]]) for key in ("before", "after")]
    if not all(rate.is_finite() for rate in rates):
        raise ValueError("Invalid rate")
    cut = int(rates[1] < rates[0])
    scores = {method: {"brier": (p-cut)**2, "log_loss": -math.log(p if cut else 1-p)}
              for method, p in case["p_cut"].items()}
    return {"retrieved_at": when, "response_sha256": raw_hash, "official_rates": values,
            "event": cut, "scores": scores}


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as tmp:
        tmp.write(encode(data))
    Path(tmp.name).replace(path)


def settle(path, clock, fetch=get_csv):
    raw_seal = SEAL.read_bytes()
    forecast = json.loads(raw_seal)
    if raw_seal != encode(create_seal()):
        raise ValueError("Seal no longer matches locked inputs and protocols")
    path = Path(path)
    state = (json.loads(path.read_bytes()) if path.exists()
             else {"seal_sha256": digest(raw_seal), "status": "waiting", "settled": {}})
    validate_state(state, forecast, digest(raw_seal))
    changed = False
    for case in forecast["events"]:
        if clock < datetime.fromisoformat(case["first_fetch_utc"]) or case["meeting"] in state["settled"]:
            continue
        try:
            raw = fetch(case["before"], case["after"])
        except HTTPError as exc:
            if exc.code == 404:
                continue
            raise
        values = observations(raw, {case["before"], case["after"]})
        if len(values) != 2:
            continue
        state["settled"][case["meeting"]] = score(case, values, clock.isoformat(), digest(raw))
        if len(state["settled"]) == len(forecast["events"]):
            state["status"] = "complete"
        save(path, state)
        changed = True
    if not path.exists():
        save(path, state)
    return {"settled": len(state["settled"]), "pending": len(forecast["events"]) - len(state["settled"]),
            "new_results": changed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify")
    settle_parser = sub.add_parser("settle")
    settle_parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "settle":
        print(json.dumps(settle(args.state, datetime.now(timezone.utc))))
    else:
        if SEAL.read_bytes() != encode(create_seal()):
            raise SystemExit("Seal verification failed")
        print("Verified", digest(SEAL.read_bytes()), "for", len(MEETINGS), "events")
