"""Offline, paired MEGALAB ARENA for the lab's independent Treasury panel."""
import argparse
import hashlib
import json
import math
import os
import tempfile
import sys
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
PROMO = HERE.parent / "promo"
sys.path.insert(0, str(PROMO))
import settle as promo
from treasury_paired import interval as four_month_block_interval

HISTORY = HERE.parent / "evidence/treasury-vintage-results.json"
HISTORY_HASH = "b6cbfed526814ce7a49978cddd040dc9f0a99fae3ab654c7b9e2f01edcc97736"
JOURNAL_COMMIT = "ad6ccf32b1ec86e5f6a5d95489bd0491b0fe948c"
JOURNAL_HASH = "68f01dcbcbff9a6d75ecf8d30d2a9ce901ac3a8363692489dccecb7a98e08fad"
MODELS = ("codex-work-session", "gemini-web-4-neon", "openai")
TEST_START = date(2025, 1, 1)
SEALED = HERE / "prospective.json"
EXPECTED_SEAL = "32c886ed36e4e1f3c9c9860fcb7fc5db5df805bda9dca4f17b17fb74c43d1fde"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_history():
    promo.require(digest(HISTORY) == HISTORY_HASH, "unexpected ALFRED archive")
    return promo.read_json(HISTORY)["cases"]


def fit(rows, as_of):
    # The archived script queried labels at origin + 40 days. Even an earlier
    # event in the previous month is unavailable until that vintage is visible.
    train = [r for r in rows if date.fromisoformat(r["origin"]) + timedelta(days=40) < as_of]
    promo.require(len(train) >= 48, "insufficient historical labels")
    x = [r["reference"] - 0.5 for r in train]
    y = [r["outcome"] - 0.5 for r in train]
    # Minimize sum (y - a - b*x)^2 + 4*a^2 + (b-1)^2; shrink toward identity.
    xx = sum(t*t for t in x) + 1
    xy = sum(s*t for s, t in zip(x, y)) + 1
    n = len(x) + 4
    sx, sy = sum(x), sum(y)
    b = (n*xy - sx*sy) / (n*xx - sx*sx)
    return {"a": (sy-b*sx)/n, "b": b, "training_cases": len(x),
            "last_training_origin": max(r["origin"] for r in train)}


def calibrated(p, model):
    return max(0.01, min(0.99, 0.5 + model["a"] + model["b"] * (p-0.5)))


def metrics(py):
    # Five fixed-width bins; equal probabilities (including baseline 0.5)
    # always land in the same bin.
    groups = defaultdict(list)
    for p, y in py:
        groups[min(4, int(p*5))].append((p, y))
    bins = [{"bin": i, "n": len(g), "mean_p": mean(p for p, _ in g),
             "event_rate": mean(y for _, y in g)} for i, g in sorted(groups.items())]
    return {
        "n": len(py), "brier": mean((p-y)**2 for p, y in py),
        "log_loss_nats": -mean(math.log(p if y else 1-p) for p, y in py),
        "mean_probability_minus_event_rate": mean(p-y for p, y in py),
        "ece_5_bins": sum(g["n"] * abs(g["mean_p"] - g["event_rate"]) for g in bins) / len(py),
        "bins": bins,
    }


def backtest():
    rows = load_history()
    fit_result = fit(rows, TEST_START)
    test = [r for r in rows if date.fromisoformat(r["origin"]) >= TEST_START]
    origins = sorted({r["origin"] for r in test})
    arms = defaultdict(list)
    monthly_gains = defaultdict(list)
    for row in test:
        p, y = row["reference"], row["outcome"]
        calibrated_p = calibrated(p, fit_result)
        for arm, prob in (("coin", 0.5), ("reference", p), ("volatility", row["scaled"]),
                          ("calibrated_reference", calibrated_p),
                          ("reference_volatility_mean", (p+row["scaled"])/2)):
            arms[arm].append((prob, y))
        monthly_gains[row["origin"]].append((p-y)**2 - (calibrated_p-y)**2)
    paired = [mean(monthly_gains[d]) for d in origins]
    return {
        "paired_calibration_gain": mean(paired),
        "paired_calibration_gain_95pct_four_month_blocks": four_month_block_interval(paired),
        "source_sha256": HISTORY_HASH,
        "published_vintage_rule": "origin + 40 days must precede fit date",
        "fit": fit_result, "model_refits_in_test": 0, "test_first_origin": origins[0],
        "test_last_origin": origins[-1], "test_months": len(origins), "test_cases": len(test),
        "scores": {k: metrics(v) for k, v in arms.items()},
        "calibrated_vs_reference_monthly_brier_gain": {d: mean(monthly_gains[d]) for d in origins},
        "inference_cost_usd": 0, "previously_studied_archive": True, "promotion_allowed": False,
    }


def panel():
    promo.require(digest(PROMO / "panel.json") == promo.PANEL_SHA
                  and digest(PROMO / "common-prompt.txt") == promo.PROMPT_SHA,
                  "changed Treasury panel or prompt")
    return promo.read_json(PROMO / "panel.json")


def issue(journal, now):
    spec = panel()
    deadline = datetime.fromisoformat(spec["issue_deadline_utc"])
    promo.require(now.tzinfo is not None and now <= deadline, "issue deadline passed")
    promo.require(digest(journal) == JOURNAL_HASH, "provider ledger differs from pinned commit")
    archival = promo.sealed_arms(PROMO / "sealed", spec, journal)
    promo.require(set(archival) == {"C", *MODELS}, "unexpected original arm coverage")
    promo.require(all(datetime.fromisoformat(a["issued_at"]) <= now for a in archival.values()),
                  "a constituent was issued later than the ensemble")
    model = fit(load_history(), now.date())
    values = {k: a["probabilities"] for k, a in archival.items()}
    keys = spec["cases"].keys()
    values["coin"] = dict.fromkeys(keys, 0.5)
    values["calibrated_reference"] = {k: calibrated(values["C"][k], model) for k in keys}
    values["model_mean"] = {k: mean(values[arm][k] for arm in MODELS) for k in keys}
    values["model_reference_mean"] = {k: (values["model_mean"][k]+values["C"][k])/2 for k in keys}
    payload = {"schema": "megalab-arena-v1", "issued_at_utc": now.astimezone(UTC).isoformat(),
               "panel_sha256": promo.PANEL_SHA, "history_sha256": HISTORY_HASH,
               "audit_ledger_commit": JOURNAL_COMMIT, "audit_ledger_sha256": JOURNAL_HASH,
               "source_forecast_sha256": {k: a["forecast_sha256"] for k, a in archival.items()},
               "fit": model, "model_members": list(MODELS), "probabilities": values}
    return {"sha256": promo.fingerprint(payload), "payload": payload}


def checked_seal(record, spec):
    payload = record["payload"]
    promo.require(promo.fingerprint(payload) == record["sha256"] == EXPECTED_SEAL
                  and payload["schema"] == "megalab-arena-v1"
                  and payload["panel_sha256"] == promo.PANEL_SHA, "seal mismatch")
    at = datetime.fromisoformat(payload["issued_at_utc"])
    promo.require(at.tzinfo is not None and at <= datetime.fromisoformat(spec["issue_deadline_utc"]),
                  "invalid seal chronology")
    arms = payload["probabilities"]
    promo.require(set(arms) == {*MODELS, "C", "coin", "calibrated_reference",
                                "model_mean", "model_reference_mean"}, "incomplete seal")
    for name, values in arms.items():
        promo.require(values.keys() == spec["cases"].keys()
                      and all(type(p) in (int,float) and math.isfinite(p) and 0 < p < 1 for p in values.values()),
                      f"invalid forecast for {name}")
    return arms


def score(record, book, now=None):
    if now is None:
        now = datetime.now(UTC)
    spec = panel()
    probabilities = checked_seal(record, spec)
    promo.require(book["schema"] == promo.SCHEMA and book["panel_sha256"] == promo.PANEL_SHA,
                  "wrong settlement book")
    for arm, expected in record["payload"]["source_forecast_sha256"].items():
        promo.require(promo.fingerprint(book["arms"][arm]["probabilities"]) == expected,
                      f"settlement did not use the sealed forecast for {arm}")
    resolved = book["resolutions"]
    promo.require(resolved.keys() <= spec["cases"].keys(), "unexpected case")
    for key, item in resolved.items():
        promo.require(item["status"] in ("resolved", "void"), "unknown resolution status")
        case = spec["cases"][key]
        target = date.fromisoformat(case["target"])
        when = datetime.fromisoformat(item["recorded_utc"])
        promo.require(when.tzinfo is not None and when <= now, "unpublished settlement")
        if item["status"] == "void":
            promo.require(item["reason"] == "no_evaluable_official_observation_in_window"
                          and target + timedelta(days=8) <= when.astimezone(UTC).date(),
                          "premature or unexplained void")
        else:
            observed = date.fromisoformat(item["observed_date"])
            promo.require(target <= observed <= target + timedelta(days=6)
                          and observed + timedelta(days=2) <= when.astimezone(UTC).date()
                          and Decimal(item["threshold"]) == Decimal(case["threshold"])
                          and type(item["outcome"]) is int
                          and item["outcome"] == int(Decimal(item["value"]) > Decimal(case["threshold"])),
                          "invalid published resolution")
    result = promo.scores({k: {"probabilities": v} for k, v in probabilities.items()}, resolved)
    count = sum(item["status"] == "resolved" for item in resolved.values())
    return {"schema": "megalab-arena-score-v1", "seal_sha256": record["sha256"],
            "cases": len(spec["cases"]), "target_dates": sorted({c["target"] for c in spec["cases"].values()}),
            "resolved": count, "void": len(resolved)-count, "pending": len(spec["cases"])-len(resolved),
            "scores": result, "promotion_allowed": False}


def write_once(path, record):
    # Publish complete bytes atomically; a second writer cannot replace the seal.
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        tmp_path.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
                            encoding="utf-8")
        os.link(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("backtest")
    issuer = sub.add_parser("issue")
    issuer.add_argument("journal", type=Path)
    scorer = sub.add_parser("score")
    scorer.add_argument("state", type=Path)
    args = parser.parse_args()
    if args.cmd == "backtest":
        result = backtest()
    elif args.cmd == "issue":
        result = issue(args.journal, datetime.now(UTC))
        write_once(SEALED, result)
    else:
        result = score(promo.read_json(SEALED), promo.read_json(args.state))
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
