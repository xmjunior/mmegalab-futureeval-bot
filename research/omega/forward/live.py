"""Forward-only weekly spread forecasts. No access to competition data."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "promo"))
import settle
import runner

FIRST = date(2026, 11, 24)
LAST = FIRST + timedelta(weeks=4)
STATE_BRANCH = "research/omega-forward-ledger"
MODEL = "openai/gpt-oss-120b"
NAMES = ("blind", "anchored")
SCHEMA = "omega-spread-anchor-v1"


def sha(content):
    return hashlib.sha256(content).hexdigest()


def identify(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def new_case(now, source):
    day = now.date()
    series = {}
    sources = {}
    for yr in (day.year, day.year - 1):
        data, sources[str(yr)] = source.observations(yr)
        series.update({d: row["10y"] - row["2y"] for d, row in data.items()
                       if d < day and {"10y", "2y"} <= row.keys()})
        if len(series) >= 181:
            break
    days = sorted(series)[-181:]
    if len(days) < 101 or day - days[-1] > timedelta(days=7):
        raise ValueError("insufficient or stale historical spreads")
    spreads = [series[d] for d in days]
    changes = [spreads[i] - spreads[i-5] for i in range(len(days)-1, 4, -5)]
    wins = sum(x > 0 for x in changes)
    n = len(changes)
    anchor = (wins + 2) / (n + 4)
    target = day + timedelta(days=7)
    spec = {"target": target.isoformat(), "leg": "spread", "threshold": str(spreads[-1])}
    evidence = {"observations": [[d.isoformat(), str(series[d])] for d in days],
                "positive": wins, "intervals": n, "source_sha256": sources}
    design = {"issue_day": day.isoformat(), "created_utc": now.isoformat(), "spec": spec,
              "evidence": evidence, "reference": anchor}
    return {"design": design, "design_sha256": identify(design), "models": {}, "resolution": None}


def prompt(case, name):
    design = case["design"]
    evidence = design["evidence"]
    base = design["reference"]
    series = evidence["observations"]
    vals = [settle.Decimal(v) for _, v in series[-6:]]
    changes = [int((b-a)*100) for a,b in zip(vals,vals[1:])]
    spec = design["spec"]
    text = (f"Forecast at {design['issue_day']} UTC: will the US Treasury 10-year minus 2-year par yield "
            f"spread be strictly greater than {spec['threshold']} percentage points on the first official "
            f"complete observation dated {spec['target']} or the following six days? Equality is NO. "
            f"The last observation is {series[-1][0]}. Of {evidence['intervals']} preceding non-overlapping "
            f"five-observation intervals {evidence['positive']} had positive changes. The last five "
            f"daily changes in basis points, in order, are {changes}. ")
    if name == "anchored":
        text += f"The Beta(2,2)-smoothed empirical reference probability is {base:.6f}. "
    return text + 'Use only this evidence. Return one JSON object {"p": probability} with 0 < p < 1.'


def seal_for(case, name, record):
    fields = {k: v for k, v in record.items() if k != "sha256"}
    return identify([case["design_sha256"], name, fields])


def validate(key, case):
    design = case["design"]
    if identify(design) != case["design_sha256"] or key != design["issue_day"]:
        raise ValueError("changed frozen evidence")
    issued = datetime.fromisoformat(design["created_utc"])
    if issued.tzinfo is None or issued.astimezone(UTC).date().isoformat() != key:
        raise ValueError("invalid issue time")
    for name, arm in case["models"].items():
        if (name not in NAMES or arm["model"] != MODEL or arm["status"] not in {"leased","sealed","unavailable"}
                or arm["prompt_sha256"] != sha(prompt(case, name).encode())):
            raise ValueError("unexpected arm")
        if arm["status"] == "sealed":
            stamped = datetime.fromisoformat(arm["issued_utc"])
            if (stamped.tzinfo is None or not issued <= stamped or stamped.astimezone(UTC).date().isoformat() != key
                    or seal_for(case, name, arm) != arm["sha256"]
                    or not 0 < arm["p"] < 1):
                raise ValueError("invalid sealed arm")
    resolution = case["resolution"]
    if resolution is not None:
        if seal_for(case, "resolution", resolution) != resolution["sha256"]:
            raise ValueError("changed resolution")
        target = date.fromisoformat(design["spec"]["target"])
        recorded = datetime.fromisoformat(resolution["recorded_utc"]).astimezone(UTC).date()
        if resolution["status"] == "resolved":
            obs = date.fromisoformat(resolution["observed_date"])
            threshold = settle.Decimal(design["spec"]["threshold"])
            if (not target <= obs <= target + timedelta(days=6) or recorded < obs + timedelta(days=2)
                    or settle.Decimal(resolution["threshold"]) != threshold
                    or resolution["outcome"] != int(settle.Decimal(resolution["value"]) > threshold)):
                raise ValueError("invalid resolution criteria")
        elif resolution["status"] != "void" or recorded < target + timedelta(days=8):
            raise ValueError("invalid void")


def summary(book):
    paired = []
    counts = {k: 0 for k in NAMES}
    settled = 0
    for key, case in book["cases"].items():
        validate(key, case)
        res = case["resolution"]
        for name, arm in case["models"].items():
            counts[name] += int(arm["status"] == "sealed")
        if not res or res["status"] != "resolved":
            continue
        settled += 1
        if all(case["models"].get(k, {}).get("status") == "sealed" for k in NAMES):
            paired.append((res["outcome"], {"coin": .5, "reference": case["design"]["reference"],
                                           **{k: case["models"][k]["p"] for k in NAMES}}))
    scores = {}
    for name in ("coin","reference",*NAMES):
        vals = [(y, row[name]) for y, row in paired]
        if vals:
            scores[name] = {"brier": mean((y-p)**2 for y,p in vals),
                            "log_loss": -mean(math.log(p if y else 1-p) for y,p in vals),
                            "mean_probability_minus_outcome": mean(p-y for y,p in vals)}
    return {"issued": len(book["cases"]), "resolved": settled, "model_coverage": counts,
            "paired": len(paired), "scores": scores, "additional_cost_usd": 0, "promotion": False}


def advance(book, source, now, call, checkpoint):
    if book["schema"] != SCHEMA:
        raise ValueError("unexpected state")
    for key, case in book["cases"].items():
        validate(key, case)
        if case["resolution"] is None:
            result = settle.resolution(case["design"]["spec"], source, now())
            if result is not None:
                result["sha256"] = seal_for(case, "resolution", result)
                case["resolution"] = result
                checkpoint("Record the first eligible official Treasury resolution")
    stamp = now()
    day = stamp.date()
    if not FIRST <= day <= LAST or (day - FIRST).days % 7:
        return
    key = day.isoformat()
    if key not in book["cases"]:
        book["cases"][key] = new_case(stamp, source)
        checkpoint("Freeze the weekly evidence and reference before inference")
    case = book["cases"][key]
    order = NAMES if ((day-FIRST).days//7)%2 == 0 else NAMES[::-1]
    for name in order:
        if name in case["models"] or now().date() != day:
            continue
        query = prompt(case, name)
        arm = {"status": "leased", "prompt_sha256": sha(query.encode()), "model": MODEL}
        case["models"][name] = arm
        checkpoint("Reserve the single weekly Groq request for " + name)
        try:
            text, usage = call(query, max_tokens=448)
            reply = runner.probabilities(text, {"cases": {"p": None}})
            issued = now()
            if issued.date() != day:
                raise ValueError("late forecast")
            arm.update(status="sealed", p=reply["p"], issued_utc=issued.isoformat(), tokens=usage,
                       response_sha256=sha(text.encode()))
        except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
            arm.update(status="unavailable", reason=type(exc).__name__)
        if arm["status"] == "sealed":
            arm["sha256"] = seal_for(case, name, arm)
        checkpoint("Seal the weekly " + name + " outcome without repeating requests")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state_checkout", type=Path)
    opts = parser.parse_args()
    folder = opts.state_checkout.resolve()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(folder), *args], text=True).strip()
    if git("branch", "--show-current") != STATE_BRANCH:
        raise ValueError("wrong state branch")
    path = folder / "ledger.json"
    book = json.loads(path.read_bytes())
    def checkpoint(message):
        book["summary"] = summary(book)
        runner.shadow.write_ledger(path, book)
        git("add","--","ledger.json")
        git("-c","user.name=MEGALAB research","-c","user.email=forecast@users.noreply.github.com",
            "commit","-qm",message)
        git("push","--quiet","origin","HEAD:refs/heads/"+STATE_BRANCH)
    import groq_free
    if groq_free.MODEL != MODEL:
        raise ValueError("different model")
    advance(book, settle.OfficialTreasury(), lambda: datetime.now(UTC),
            groq_free.GroqFreeRoute().complete, checkpoint)
    print(json.dumps(summary(book), sort_keys=True))


if __name__ == "__main__":
    main()
