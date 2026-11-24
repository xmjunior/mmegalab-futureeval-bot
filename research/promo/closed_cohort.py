"""A date-masked historical calibration experiment on twelve disjoint Treasury horizons."""
import argparse
import csv
import io
import json
import math
import os
import re
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

import runner as r

STUDY = "arena-ust-closed-blind-1"
DATA = Path(__file__).with_name("treasury-asof.csv")
DATA_SHA = "802ac4001ae8dae72a7b01132db702f7136d96ddf94c07da4f4dfa0f7c23ac25"
ORIGINS = tuple(range(64, 197, 12))
EXPIRY = datetime.fromisoformat("2026-11-26T00:00:00+00:00")
VERIFIED_GRANT_FLOOR = r.LIMIT_MICRO_USD  # The verified grant covers the whole authorized ceiling.
RUN_WINDOW = timedelta(minutes=16)
ROOT = Path(__file__).resolve().parents[2]
OWNER_WINDOW = Path("research/promo/owner-window.json")
REFERENCE_RECEIPT = "030877ec92fbc638a3f33c67aa4b7f30b5dc2e0a326c8fcf7b8ca854efda2692"
OWNER_SCOPE = "owner_cohort_project_key"


def load_values():
    raw = DATA.read_bytes()
    r.require(r.sha(raw) == DATA_SHA, "frozen_series_changed")
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    dates = [row["observation_date"] for row in rows]
    r.require(dates == sorted(set(dates)) and dates[-1] == "2026-11-23", "invalid_dates")
    return [int(Decimal(row["two_year_percent"]) * 100) for row in rows]


def case(values, origin):
    # Only past differences are exposed. No dates, absolute yields, outcomes or other origins.
    increases = sum(values[i] > values[i - 5] for i in range(origin, origin - 60, -5))
    recent = [values[i] - values[i - 1] for i in range(origin - 4, origin + 1)]
    prompt = (
        "Predict whether a benchmark government yield will increase strictly over its next "
        "five observations. Units are basis points. Use only these summaries from the "
        "forecast origin. No tools. The twelve preceding disjoint five-observation changes "
        f"included {increases} strict increases. Last five daily changes, oldest first: "
        f"{recent}. Return only a JSON object with key p and a numeric probability between 0 and 1."
    )
    return prompt, (increases + 2) / 16


def score(predictions, outcomes):
    pairs = list(zip(predictions, outcomes, strict=True))
    return {
        "n": len(pairs),
        "brier": sum((p - y) ** 2 for p, y in pairs) / len(pairs),
        "log_loss": -sum(math.log(p if y else 1 - p) for p, y in pairs) / len(pairs),
        "mean_probability": sum(p for p, _ in pairs) / len(pairs),
        "observed_fraction": sum(y for _, y in pairs) / len(pairs),
    }


def owner_scope(now, root):
    # Authorization is a single-data-file child of the reviewed executable commit.
    # No account identifiers or credentials appear in that file.
    scope = json.loads((root / OWNER_WINDOW).read_bytes())
    head = os.getenv("GITHUB_SHA", "")
    r.require(os.getenv("GITHUB_REPOSITORY") == "xmjunior/mmegalab-futureeval-bot"
              and os.getenv("GITHUB_REF") == "refs/heads/verify/arena-closed-grant"
              and os.getenv("GITHUB_EVENT_NAME") == "push" and os.getenv("GITHUB_RUN_ATTEMPT") == "1"
              and re.fullmatch("[0-9a-f]{40}", head), "wrong_owner_ci_context")
    parent = scope.get("reviewed_code_sha", "")
    r.require(isinstance(parent, str) and re.fullmatch("[0-9a-f]{40}", parent)
              and r.git(root, "rev-list", "-n", "1", "--parents", "HEAD") == f"{head} {parent}"
              and r.git(root, "diff", "--name-only", parent, head) == OWNER_WINDOW.as_posix()
              and not r.git(root, "status", "--porcelain", "--untracked-files=no"),
              "unreviewed_owner_window_commit")
    r.require(scope.get("study") == STUDY and scope.get("snapshot_sha256") == r.OWNER_CAPTURE_SHA256
              and scope.get("prior_run_id") == "45493080434"
              and scope.get("verified_orgwide_including_pending") is True
              and type(scope.get("external_upper_micro_usd")) is int
              and 0 <= scope["external_upper_micro_usd"] <= VERIFIED_GRANT_FLOOR,
              "invalid_exposure_evidence")
    checked = datetime.fromisoformat(scope["attested_at"])
    end = datetime.fromisoformat(scope["valid_through"])
    r.require(datetime.fromisoformat(r.OWNER_CHECKED) <= checked <= now
              and now + RUN_WINDOW <= min(end, EXPIRY), "exposure_or_credit_window_closed")
    return {**scope, "credential_scope": OWNER_SCOPE, "authorization_commit": head}


def reviewed_scope(now):
    if (ROOT / OWNER_WINDOW).is_file():
        return owner_scope(now, ROOT)
    # Keep the verified wallet snapshot. Only new exposure since then needs bounding.
    # This record is prepared after checking the same organization and project. The bound
    # includes all non-journal charges, pending usage predating the screenshot and
    # concurrent/outstanding requests through valid_through, including other projects.
    raw = os.environ.get("ARENA_EXPOSURE_RECORD", "").encode()
    fingerprint = os.environ.get("ARENA_EXPOSURE_SHA256", "")
    r.require(raw and len(fingerprint) == 64 and r.sha(raw) == fingerprint,
              "external_exposure_unbounded")
    scope = json.loads(raw)
    r.require(scope.get("snapshot_sha256") == r.OWNER_CAPTURE_SHA256
              and scope.get("verified_orgwide_including_pending") is True
              and scope["org"].startswith("org-") and scope["project"].startswith("proj_")
              and type(scope["external_upper_micro_usd"]) is int
              and 0 <= scope["external_upper_micro_usd"] <= VERIFIED_GRANT_FLOOR,
              "invalid_exposure_evidence")
    start = datetime.fromisoformat(scope["covered_from"])
    end = datetime.fromisoformat(scope["valid_through"])
    observed = datetime.fromisoformat(r.OWNER_CHECKED)
    r.require(start <= observed <= now and now + RUN_WINDOW <= min(end, EXPIRY),
              "exposure_or_credit_window_closed")
    return scope


def prior_reservations(book):
    rows = list(book["arms"].values()) + list(book.get("experiments", {}).values())
    r.require(all(type(row.get("max_micro_usd")) is int and row["max_micro_usd"] >= 0
                  for row in rows), "invalid_prior_reservations")
    r.require(all(row.get("status") != "reserved" for row in rows), "prior_reservation_in_flight")
    return sum(row["max_micro_usd"] for row in rows)


def dry():
    values = load_values()
    outcomes = [int(values[i + 5] > values[i]) for i in ORIGINS]
    baseline = [case(values, i)[1] for i in ORIGINS]
    return {"study": STUDY, "kind": "masked_retrospective", "source_sha256": DATA_SHA,
            "independent_requests_planned": len(ORIGINS), "inference_requests": 0,
            "largest_possible_reservation_micro_usd": len(ORIGINS) * r.plan(r.MAX_INPUT),
            "baseline": score(baseline, outcomes), "neutral": score([0.5] * len(ORIGINS), outcomes)}


def run(journal, scope, now):
    book = journal.book
    r.require(STUDY not in book.get("experiments", {}), "study_already_reserved")
    prior = prior_reservations(book)
    if scope.get("credential_scope") == OWNER_SCOPE:
        historical = book["arms"].get("openai", {})
        r.require(historical.get("status") == "completed"
                  and historical.get("reviewed_capture_sha256") == r.OWNER_CAPTURE_SHA256
                  and historical.get("github_run_id") == scope["prior_run_id"]
                  and historical.get("request", {}).get("response_id_sha256") == REFERENCE_RECEIPT,
                  "prior_owner_receipt_missing")
    values = load_values()
    inputs = []
    for origin in ORIGINS:
        prompt, baseline = case(values, origin)
        # Verify current official pricing and free exact token count. No inference in this loop.
        inp, count, reserve = r.prepare_openai(prompt, scope)
        inputs.append((origin, inp, count, reserve, baseline))
    reserve = sum(item[3] for item in inputs)
    protected = min(r.LIMIT_MICRO_USD, VERIFIED_GRANT_FLOOR - scope["external_upper_micro_usd"])
    r.require(prior + reserve <= protected, "insufficient_protected_promotional_credit")
    end = min(datetime.fromisoformat(scope["valid_through"]), EXPIRY)
    r.require(now() + RUN_WINDOW <= end, "exposure_or_credit_window_closed")
    row = {"status": "reserved", "started_at": now().isoformat(), "max_micro_usd": reserve,
           "source_sha256": DATA_SHA, "planned": len(inputs), "receipts": []}
    if scope.get("credential_scope") == OWNER_SCOPE:
        row["authorization_commit"] = scope["authorization_commit"]
        row["prior_receipt_sha256"] = REFERENCE_RECEIPT
    book.setdefault("experiments", {})[STUDY] = row
    journal.commit("Reserve the complete blinded historical cohort before inference")
    predicted = []
    observed = []
    baseline_probs = []
    for origin, inp, count, cap, baseline in inputs:
        try:
            r.require(now() + timedelta(seconds=70) < end, "grant_window_closed")
            answer, receipt = r.try_openai(inp, count, scope)
            row["receipts"].append(dict(receipt, prompt_sha256=r.sha(inp["input"].encode()),
                                        origin_index=origin, max_micro_usd=cap))
            r.require(isinstance(answer, str), "missing_answer")
            probability = r.probabilities(answer, {"cases": {"p": None}})["p"]
            outcome = int(values[origin + 5] > values[origin])
            row["receipts"][-1].update(probability=probability, baseline=baseline, outcome=outcome)
            predicted.append(probability)
            observed.append(outcome)
            baseline_probs.append(baseline)
        except (OSError, ValueError, KeyError, TypeError, ArithmeticError) as exc:
            row.update(status="stopped_no_retry", stopped_origin_index=origin,
                       reason=r.shadow.safe_status(exc))
            break
        journal.commit("Record one blinded case receipt; retain the full reservation")
    if len(predicted) == len(inputs):
        row["status"] = "completed"
    if predicted:
        row["paired_model"] = score(predicted, observed)
        row["paired_baseline"] = score(baseline_probs, observed)
    row["finished_at"] = now().isoformat()
    journal.commit("Close the blinded cohort and retain its reserved maximum")
    return row["status"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dry", "run"))
    mode = parser.parse_args().mode
    if mode == "dry":
        print(json.dumps(dry(), sort_keys=True))
    else:
        try:
            scope = reviewed_scope(r.utc())
            r.require(bool(os.getenv("OPENAI_API_KEY")), "missing_project_credential")
            with TemporaryDirectory() as folder:
                result = run(r.Journal(Path(folder)), scope, r.utc)
            print(json.dumps({"study": STUDY, "status": result}))
            raise SystemExit(int(result != "completed"))
        except (OSError, ValueError, KeyError, TypeError, ArithmeticError) as exc:
            print(json.dumps({"study": STUDY, "status": "blocked", "reason": r.shadow.safe_status(exc)}))
            raise SystemExit(2)
