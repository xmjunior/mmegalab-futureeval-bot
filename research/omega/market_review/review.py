"""Verify the Market Pulse interval-review contract with one named Groq Free model."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

FIXTURE = Path(__file__).with_name("cases.json")
FIXTURE_HASH = "990759d77e91146d9d87a76db019b7dfb52d45c0cb73ac85d4b45728b1f507c6"


def cases(path=FIXTURE):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FIXTURE_HASH:
        raise ValueError("Historical source differs from the sealed fixture")
    entries = json.loads(raw)["cases"]
    for entry in entries:
        if not datetime.fromisoformat(entry["cutoff"]) < datetime.fromisoformat(entry["resolution"]) < datetime.now(timezone.utc):
            raise ValueError("Window is not closed")
        if hashlib.sha256(entry["prompt"].encode()).hexdigest() != entry["prompt_sha256"]:
            raise ValueError("Prompt differs from the original numeric reviewer")
    return entries


def parse_response(raw, entry):
    # Same one-row width contract as the existing market_pulse.review.
    row, = json.loads(entry["prompt"].rsplit("\n", 1)[1])
    choices = [option["width"] for option in row["choices"]]
    answer = json.loads(raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
    if not isinstance(answer, list) or len(answer) != 1:
        raise ValueError("Invalid number of interval decisions")
    pick = answer[0]
    if pick is not None and (type(pick) not in (int, float) or pick not in choices):
        raise ValueError("Ineligible interval width")
    return pick


def trial(complete):
    receipts = []
    for entry in cases():
        try:
            answer, usage = complete(entry["prompt"], max_tokens=512)
            pick = parse_response(answer, entry)
        except Exception as exc:
            # Preserve earlier decisions, stop after the first error, never retry.
            return {"status": "incomplete", "error_type": type(exc).__name__, "failed_case": entry["id"],
                    "hosted_requests_attempted": len(receipts) + 1, "cases": receipts, "published": 0}
        receipts.append({"id": entry["id"], "prompt_sha256": entry["prompt_sha256"], "width": pick, "usage": usage})
    abstentions = sum(row["width"] is None for row in receipts)
    return {"status": "completed", "hosted_requests_attempted": len(receipts), "abstentions": abstentions,
            "numeric_decisions": len(receipts) - abstentions, "cases": receipts, "published": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("offline", "groq"))
    args = parser.parse_args()
    if args.mode == "offline":
        print(json.dumps({"verified_closed_cases": len(cases()), "hosted_requests": 0, "published": 0}))
    else:
        from groq_free import GroqFreeRoute
        transport = GroqFreeRoute()
        result = trial(transport.complete)
        result.update(model=transport.model, fixture_sha256=FIXTURE_HASH)
        print(json.dumps(result, sort_keys=True))
        raise SystemExit(0 if result["status"] == "completed" else 1)
