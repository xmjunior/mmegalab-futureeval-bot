"""One free Groq call per origin on the same date-masked closed Treasury cohort."""
import argparse
import json
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "promo"))
import closed_cohort as study

r = study.r
ARM = "groq-gpt-oss-120b"
MODEL = "openai/gpt-oss-120b"  # Open weights served by Groq only.
POOL = "free_experiments"  # Independent of OpenAI reservations, limits and billing.
PAUSE_SECONDS = 16
MAX_OUTPUT = 448


def summary(book):
    arm = book[POOL][study.STUDY][ARM]
    values = study.load_values()
    rows = []
    for origin in study.ORIGINS:
        record = arm["cases"].get(str(origin))
        if record and record["status"] == "completed":
            prompt, baseline = study.case(values, origin)
            r.require(record["prompt_sha256"] == r.sha(prompt.encode()), "prompt_changed")
            rows.append((origin, record, baseline, int(values[origin+5] > values[origin])))
    result = {"study": study.STUDY, "model": MODEL, "source_sha256": study.DATA_SHA,
              "kind": "date_masked_retrospective", "planned": len(study.ORIGINS),
              "scored": len(rows), "missing": len(study.ORIGINS)-len(rows),
              "inference_cost_usd": 0, "openai_inference_calls": 0,
              "competition_submissions": 0, "promotion_allowed": False}
    if rows:
        actual = [y for _, _, _, y in rows]
        probabilities = [row["p"] for _, row, _, _ in rows]
        result.update(groq=study.score(probabilities, actual),
                      baseline=study.score([p for _, _, p, _ in rows], actual),
                      neutral=study.score([0.5]*len(actual), actual))
        result["paired_brier_gain_vs_baseline"] = result["baseline"]["brier"]-result["groq"]["brier"]
        result["paired_log_loss_gain_vs_baseline"] = result["baseline"]["log_loss"]-result["groq"]["log_loss"]
    # Compare only when origin, prompt hash, source, baseline and truth match.
    other = book.get("experiments", {}).get(study.STUDY, {})
    matched = []
    if other:
        r.require(other["source_sha256"] == study.DATA_SHA, "other_cohort_changed")
        lookup = {}
        for receipt in other["receipts"]:
            if "probability" in receipt:
                origin = receipt["origin_index"]
                r.require(origin not in lookup, "duplicate_other_origin")
                lookup[origin] = receipt
        for origin, record, baseline, truth in rows:
            if origin in lookup:
                peer = lookup[origin]
                r.require(peer["prompt_sha256"] == record["prompt_sha256"]
                          and peer["outcome"] == truth and peer["baseline"] == baseline,
                          "mismatched_peer_origin")
                matched.append((record["p"], peer["probability"], truth))
    result["openai_paired_n"] = len(matched)
    if matched:
        gp, op, outcomes = zip(*matched)
        result["groq_on_shared"] = study.score(gp, outcomes)
        result["openai_on_shared"] = study.score(op, outcomes)
        result["equal_model_pool_on_shared"] = study.score([(g+o)/2 for g, o, _ in matched], outcomes)
    return result


def run(journal, complete, pause=time.sleep, now=r.utc):
    book = journal.book
    arms = book.setdefault(POOL, {}).setdefault(study.STUDY, {})
    values = study.load_values()
    if ARM not in arms:
        arms[ARM] = {"source_sha256": study.DATA_SHA, "model": MODEL, "created_at": now().isoformat(),
                     "status": "running", "cases": {}, "max_micro_usd": 0}
        journal.commit("Register zero-cost Groq cohort outside the OpenAI allowance")
    arm = arms[ARM]
    r.require(arm["source_sha256"] == study.DATA_SHA and arm["model"] == MODEL, "arm_changed")
    if arm["status"] in {"completed", "partial"}:
        return summary(book)

    for origin in study.ORIGINS:
        if any(row.get("stopped_cohort") for row in arm["cases"].values()):
            break
        key = str(origin)
        if key in arm["cases"]:
            # A pre-call lease without a receipt is ambiguous after interruption.
            # Do not re-infer or impute that probability.
            continue
        prompt, _ = study.case(values, origin)
        if arm["cases"]:
            pause(PAUSE_SECONDS)
        receipt = {"status": "leased", "prompt_sha256": r.sha(prompt.encode()),
                   "origin_index": origin, "issued_at": now().isoformat()}
        arm["cases"][key] = receipt
        journal.commit("Lease one Groq origin before the single provider request")
        try:
            answer, usage = complete(prompt, max_tokens=MAX_OUTPUT)
            receipt.update(usage=usage, output_sha256=r.sha(answer.encode()))
            receipt["p"] = r.probabilities(answer, {"cases":{"p": None}})["p"]
            receipt["status"] = "completed"
        except (ValueError, RuntimeError, OSError) as exc:
            status = getattr(exc, "status_code", None)
            receipt.update(status="unscored", reason=f"http_{status}" if status else type(exc).__name__)
            if status in {401, 402, 403, 404, 429}:
                # Do not retry a potentially completed call or exhaust a free quota.
                receipt["stopped_cohort"] = True
        receipt["closed_at"] = now().isoformat()
        journal.commit("Checkpoint one Groq result without changing any OpenAI charge")
        if receipt.get("stopped_cohort"):
            break
    report = summary(book)
    arm.update(status="completed" if report["missing"] == 0 else "partial",
               finished_at=now().isoformat(), report=report)
    journal.commit("Close zero-cost Groq cohort with matched historical losses")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["run", "score"])
    parser.add_argument("--journal", type=Path, help="local read-only journal for score")
    args = parser.parse_args()
    if args.mode == "score":
        if not args.journal:
            parser.error("score requires --journal")
        report = summary(json.loads(args.journal.read_bytes()))
    else:
        import groq_free
        r.require(groq_free.configured() and groq_free.MODEL == MODEL, "free_groq_unavailable")
        with TemporaryDirectory() as tmp:
            report = run(r.Journal(Path(tmp)), groq_free.GroqFreeRoute().complete)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return int(args.mode == "run" and report["missing"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
