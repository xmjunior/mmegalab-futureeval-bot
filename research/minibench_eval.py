"""Prospective MiniBench binary calibration trial. GET only, own forecasts, resolved records only."""

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import re
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

PROTOCOL = "7c66f59eb9ad2634ed93322a6d2f4fa9d9817b253d4289fe388989430550730c"
BOT = "megalab-mario-bot"
BOT_ID = 313890
BASE = "https://www.metaculus.com/api"


def timestamp(value):
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    ):
        return datetime.fromtimestamp(value, UTC)
    if not isinstance(value, str):
        raise TypeError("Missing timestamp")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("Ambiguous timezone")
    return parsed.astimezone(UTC)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class Reader:
    def __init__(self, token):
        self.token = token
        self.opener = build_opener(NoRedirect)
        self.last = 0.0

    def get(self, path, **params):
        url = BASE + path + ("?" + urlencode(params, doseq=True) if params else "")
        for attempt in range(3):
            time.sleep(max(0.0, 4 - (time.monotonic() - self.last)))
            self.last = time.monotonic()
            try:
                req = Request(
                    url,
                    method="GET",
                    headers={
                        "Authorization": "Token " + self.token,
                        "User-Agent": "megalab-resolved-minibench-v1",
                    },
                )
                with self.opener.open(req, timeout=35) as resp:
                    return json.load(resp)
            except (HTTPError, URLError, TimeoutError) as exc:
                code = getattr(exc, "code", None)
                if attempt == 2 or code not in (None, 408, 429, 500, 502, 503, 504):
                    raise RuntimeError(f"Read failed, status {code}") from None
                retry = getattr(exc, "headers", None)
                delay = retry.get("Retry-After", "") if retry else ""
                time.sleep(
                    min(
                        120,
                        max(5 * (attempt + 1), int(delay) if delay.isdigit() else 0),
                    )
                )


def verify_owner(me):
    if not (
        me.get("username") == BOT
        and me.get("id") == BOT_ID
        and me.get("is_bot") is True
        and me.get("is_primary_bot") is True
    ):
        raise RuntimeError("Primary account not verified")


def tournament_id(meta):
    ident = meta.get("id")
    slug = meta.get("slug")
    if (
        type(ident) is not int
        or ident < 1
        or not isinstance(slug, str)
        or not re.fullmatch(r"minibench(?:-[a-z0-9-]+)?", slug)
    ):
        raise RuntimeError("Unrecognized MiniBench project")
    return ident


def belongs(post, ident):
    projects = post.get("projects", {})
    candidates = [
        projects.get("default_project") or {},
        *(projects.get("tournament") or []),
    ]
    return any(isinstance(p, dict) and p.get("id") == ident for p in candidates)


def metadata(api, ident):
    rows = {}
    skipped = 0
    offset = 0
    while True:
        page = api.get(
            "/posts/",
            tournaments=ident,
            statuses="resolved",
            with_cp="false",
            include_descriptions="false",
            limit=100,
            offset=offset,
        )
        posts = page["results"]
        for post in posts:
            if not belongs(post, ident) or post.get("status") != "resolved":
                raise RuntimeError("Project/status filter mismatch")
            question = post.get("question")
            if not question:
                skipped += 1
                continue
            if question["status"] != "resolved" or question["id"] in rows:
                raise RuntimeError("Unexpected or repeated resolved question")
            rows[question["id"]] = {
                k: question.get(k)
                for k in (
                    "id",
                    "type",
                    "open_time",
                    "actual_close_time",
                    "scheduled_close_time",
                    "actual_resolve_time",
                    "resolution",
                    "resolution_set_time",
                )
            }
        if not page.get("next"):
            return rows, skipped
        if not posts or offset > 5000:
            raise RuntimeError("Incomplete results pagination")
        offset += len(posts)


def histories(api, ids):
    result = {}
    for at in range(0, len(ids), 40):
        batch = ids[at : at + 40]
        rows = api.get("/questions/bulk-forecast-read/", question_ids=batch)["results"]
        for row in rows:
            qid = row["question_id"]
            if (
                qid not in batch
                or qid in result
                or not isinstance(row["forecasts"], list)
            ):
                raise RuntimeError("Unexpected bulk history")
            if any(
                f.get("author_id") != BOT_ID or f.get("question_id") != qid
                for f in row["forecasts"]
            ):
                raise RuntimeError("Forecast owner/question mismatch")
            result[qid] = row["forecasts"]
        if not set(batch).issubset(result):
            raise RuntimeError("Incomplete bulk forecast response")
    return result


def new_stats():
    return {
        "n": 0,
        "ln_base_sum": 0.0,
        "ln_c_sum": 0.0,
        "brier_base_sum": 0.0,
        "brier_c_sum": 0.0,
        "bins": {k: [[0, 0.0, 0] for _ in range(5)] for k in ("base", "c")},
    }


def score(stats, p, outcome):
    for k, value in [("base", p), ("c", 0.9 * p + 0.05)]:
        likelihood = value if outcome else 1 - value
        stats[f"ln_{k}_sum"] -= math.log(likelihood)
        stats[f"brier_{k}_sum"] += (value - outcome) ** 2
        cell = stats["bins"][k][min(int(value * 5), 4)]
        cell[0] += 1
        cell[1] += value
        cell[2] += outcome
    stats["n"] += 1


def classify(q, history, cutoff, now, stats):
    opens = timestamp(q["open_time"])
    if opens < cutoff:
        return "opened_before_registration"
    if q["type"] != "binary":
        return "other_format"
    if not history:
        return "no_forecast"
    if len(history) != 1:
        return "multiple_forecasts"
    raw = q["resolution"]
    if not isinstance(raw, str) or raw.lower() not in ("yes", "no"):
        return "nonstandard_resolution"
    f = history[0]
    try:
        closes = timestamp(q["actual_close_time"] or q["scheduled_close_time"])
        resolved = timestamp(q["actual_resolve_time"] or q["resolution_set_time"])
        start = timestamp(f["start_time"])
        end = timestamp(f["end_time"]) if f.get("end_time") is not None else None
    except (ValueError, TypeError, OverflowError, KeyError):
        return "unverifiable_window"
    if not (opens <= start < min(closes, resolved) and resolved <= now) or (
        end and end < min(closes, resolved)
    ):
        return "outside_window_or_withdrawn"
    values = f.get("forecast_values")
    if (
        not isinstance(values, list)
        or len(values) != 2
        or any(
            type(x) not in (int, float) or not math.isfinite(x) or not 0 < x < 1
            for x in values
        )
        or not math.isclose(sum(values), 1.0, abs_tol=1e-6)
    ):
        return "invalid_binary_vector"
    score(stats, values[1], int(raw.lower() == "yes"))
    return "paired"


def census(api, ident, cutoff, now):
    questions, skipped = metadata(api, ident)
    enrolled = [
        qid for qid, q in questions.items() if timestamp(q["open_time"]) >= cutoff
    ]
    hist = histories(api, enrolled)
    reasons = Counter()
    stats = new_stats()
    for qid, q in questions.items():
        reasons[classify(q, hist.get(qid, []), cutoff, now, stats)] += 1
    coverage = {
        "all_resolved_solo": len(questions),
        "non_solo_resolved_posts": skipped,
        "all_formats_post_registration": len(enrolled),
        "all_formats_own_post_registration": sum(bool(hist[q]) for q in enrolled),
        "binary_post_registration": sum(
            questions[q]["type"] == "binary" for q in enrolled
        ),
        "eligible_pairs": stats["n"],
    }
    return {"coverage": coverage, "exclusions": dict(reasons), "stats": stats}


def round_finished(api, ident):
    # Called only after the official alias has rotated. Never inspect the returned question objects.
    for status in ("upcoming", "open", "closed"):
        page = api.get(
            "/posts/",
            tournaments=ident,
            statuses=status,
            limit=1,
            with_cp="false",
            include_descriptions="false",
        )
        if page["results"]:
            return False
    return True


def descriptive(rounds):
    """Descriptive only until the fixed review threshold is reached."""
    cells = new_stats()
    for row in rounds.values():
        block = row["stats"]
        for k in ("n", "ln_base_sum", "ln_c_sum", "brier_base_sum", "brier_c_sum"):
            cells[k] += block[k]
        for name, buckets in block["bins"].items():
            for idx, bucket in enumerate(buckets):
                for j, value in enumerate(bucket):
                    cells["bins"][name][idx][j] += value
    n = cells["n"]
    result = {"pairs": n, "base": {}, "c": {}}
    for name in ("base", "c"):
        result[name] = {
            "mean_log_loss": cells[f"ln_{name}_sum"] / n if n else None,
            "brier": cells[f"brier_{name}_sum"] / n if n else None,
            "five_bin_ece": sum(abs(prob - yes) for _, prob, yes in cells["bins"][name])
            / n
            if n
            else None,
        }
    return result


def review(rounds):
    eligible = []
    total = 0
    # A once-only fixed analysis, using the first available chronological completed prefix.
    for row in sorted(rounds.values(), key=lambda r: (r["start"], r["project"])):
        if not row["complete"]:
            break
        s = row["stats"]
        if not s["n"]:
            continue
        eligible.append((s["ln_base_sum"] - s["ln_c_sum"], s["n"]))
        total += s["n"]
        if len(eligible) >= 12 and total >= 100:
            break
    if len(eligible) < 12 or total < 100:
        return None
    rng = random.Random(77411)

    def effect(blocks):
        return sum(v for v, _ in blocks) / sum(n for _, n in blocks)

    observed = effect(eligible)
    boot = sorted(effect(rng.choices(eligible, k=len(eligible))) for _ in range(8192))
    if len(eligible) <= 15:
        signs = itertools.product((-1, 1), repeat=len(eligible))
        offsets = (0, 0)
    else:
        signs = ([rng.choice((-1, 1)) for _ in eligible] for _ in range(16384))
        offsets = (1, 1)
    extreme, count = offsets
    for sign in signs:
        v = sum(s * b[0] for s, b in zip(sign, eligible)) / total
        count += 1
        extreme += int(abs(v) >= abs(observed) - 1e-14)
    p = extreme / count
    ci = [boot[int(0.025 * (len(boot) - 1))], boot[int(0.975 * (len(boot) - 1))]]
    return {
        "rounds": len(eligible),
        "questions": total,
        "gain_nats": observed,
        "ci95": ci,
        "p_two_sided": p,
        "new_replication_warranted": observed > 0 and ci[0] > 0 and p < 0.05,
        "automatic_promotion": False,
    }


def update(api, ledger, cutoff, now):
    verify_owner(api.get("/users/me/"))
    current = api.get("/projects/tournaments/minibench/")
    active = tournament_id(current)
    rounds = ledger["rounds"]
    key = str(active)
    if key not in rounds:
        rounds[key] = {
            "project": active,
            "start": current["start_date"],
            "slug": current["slug"],
            "complete": False,
            "stats": new_stats(),
        }
    for key, row in rounds.items():
        if row["complete"]:
            continue
        ident = int(key)
        if ident != active:
            meta = api.get(f"/projects/tournaments/{ident}/")
            if tournament_id(meta) != ident:
                raise RuntimeError("Previous round mismatch")
        row.update(census(api, ident, cutoff, now))
        row["complete"] = ident != active and round_finished(api, ident)
    if ledger.get("fixed_review") is None:
        ledger["fixed_review"] = review(rounds)
    ledger["descriptive"] = descriptive(rounds)
    ledger["evaluated_utc"] = now.isoformat()
    return ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, type=Path)
    path = parser.parse_args().ledger
    raw = Path(__file__).with_name("minibench_protocol.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROTOCOL:
        raise RuntimeError("Preregistered protocol changed")
    cutoff = timestamp(json.loads(raw)["frozen_before_first_open"])
    ledger = (
        json.loads(path.read_text())
        if path.exists()
        else {"protocol_sha256": PROTOCOL, "rounds": {}, "fixed_review": None}
    )
    if ledger.get("protocol_sha256") != PROTOCOL:
        raise RuntimeError("Unexpected ledger version")
    api = Reader(os.environ["METACULUS_TOKEN"])
    out = update(api, ledger, cutoff, datetime.now(UTC))
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(out, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)
    print(
        json.dumps(
            {
                "rounds": len(out["rounds"]),
                "paired_resolved": sum(r["stats"]["n"] for r in out["rounds"].values()),
                "fixed_review": out["fixed_review"],
            }
        )
    )


if __name__ == "__main__":
    main()
