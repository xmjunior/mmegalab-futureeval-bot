"""Market Pulse 26Q4 only. Historical validation, no-write rehearsal and live spot updates."""
import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from forecasting_tools import DiscreteQuestion, GeneralLlm, NumericDistribution, NumericQuestion, Percentile

import groq_free
import market_prices as prices
from main import FreeForecastLlm

MARKET_QUESTION_TYPES = {"numeric": NumericQuestion, "discrete": DiscreteQuestion}
MARKET_DISCRETE_GROUPS = {46010}
HY_STEP = 0.01
PROJECT = 33131
BASE = "https://www.metaculus.com/api"
BOT = "megalab-mario-bot"
UTC = timezone.utc
MODEL_NAMES = {
    "GEMINI_API_KEY": ["gemini/gemma-4-31b-it", "gemini/gemma-4-26b-a4b-it"],
    "OPENROUTER_API_KEY": ["openrouter/google/gemma-4-31b-it:free", "openrouter/google/gemma-4-26b-a4b-it:free"],
}
PRIMARY_ROUTES = frozenset(route for names in MODEL_NAMES.values() for route in names)
log = logging.getLogger("market_pulse")


def timestamp(value):
    if isinstance(value, (float, int)):
        return datetime.fromtimestamp(value, UTC)
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def discrete_geometry(row):
    """HY observations use decimal cent values at the centers of equally spaced bins."""
    scale = row["scaling"]
    edges = scale["continuous_range"]
    count = scale["inbound_outcome_count"]
    low, high = scale["range_min"], scale["range_max"]
    def near(left, right):
        return math.isclose(left, right, rel_tol=0, abs_tol=1e-10)

    if (type(count) is not int or count < 2 or row.get("inbound_outcome_count", count) != count
            or scale["zero_point"] is not None or len(edges) != count + 1
            or not near(high - low, count * HY_STEP)
            or not near(scale["nominal_min"], low + HY_STEP/2)
            or not near(scale["nominal_max"], high - HY_STEP/2)
            or not near(scale["nominal_min"] / HY_STEP, round(scale["nominal_min"] / HY_STEP))
            or not all(near(edge, low + i*HY_STEP) for i, edge in enumerate(edges))
            or any(row.get(name) is not True or scale.get(name) is not True
                   for name in ("open_lower_bound", "open_upper_bound"))):
        raise ValueError("Unqualified discrete Market Pulse scale")
    return count + 1


def parse_market_question(post, row):
    kind = row["type"]
    if kind not in MARKET_QUESTION_TYPES or (kind == "discrete" and post["id"] not in MARKET_DISCRETE_GROUPS):
        return None
    size = discrete_geometry(row) if kind == "discrete" else None
    question = MARKET_QUESTION_TYPES[kind].from_metaculus_api_json({**post, "question": row})
    if size is not None and (question.cdf_size != size or question.question_type != kind):
        raise ValueError("SDK dimension or kind differs from the verified discrete geometry")
    return question


class Account:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Token {os.environ['METACULUS_TOKEN']}", "Accept-Language": "en"})
        user = self.get("/users/me")
        if user.get("username", "").casefold() != BOT or user.get("is_bot") is not True:
            raise ValueError("Expected the registered bot account.")
        log.info("Authenticated bot=%s project=%s", BOT, PROJECT)

    def get(self, path):
        response = self.session.get(BASE + path, timeout=30)
        response.raise_for_status()
        return response.json()

    def group(self, post_id, *, question_ids=None):
        if post_id not in prices.SPECS:
            raise ValueError(f"Unvalidated group {post_id}")
        post = self.get(f"/posts/{post_id}/")
        if (post["id"] != post_id or post["projects"]["default_project"]["id"] != PROJECT
                or PROJECT not in {p["id"] for p in post["projects"]["tournament"]}):
            raise ValueError("Unexpected group tournament")
        group = post["group_of_questions"]
        questions = {}
        for row in group["questions"]:
            if question_ids is not None and row["id"] not in question_ids:
                continue
            question = parse_market_question(post, row)
            if question is not None:
                if question.id_of_question in questions:
                    raise ValueError("Duplicate Market Pulse subquestion")
                questions[question.id_of_question] = question
        return questions

    def submit_once(self, q, values):
        if len(values) != q.cdf_size or not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
            raise ValueError("Invalid Market Pulse CDF size or probability")
        if any(a > b for a, b in zip(values, values[1:])):
            raise ValueError("Unordered Market Pulse CDF")
        # Never retry an ambiguous POST. The next scheduled run will reconcile account history.
        response = self.session.post(BASE + "/questions/forecast/", timeout=40, json=[
            {"question": q.id_of_question, "source": "api", "continuous_cdf": values}])
        response.raise_for_status()
        fresh = self.group(q.id_of_post)[q.id_of_question]
        saved = latest_forecast(fresh)
        if not saved or not same_cdf(values, saved["forecast_values"]):
            raise RuntimeError(f"Forecast accepted but readback differs for {q.id_of_question}")
        log.info("Confirmed forecast by account readback question=%d", q.id_of_question)


def latest_forecast(q):
    record = q.api_json["question"].get("my_forecasts")
    if not isinstance(record, dict) or not isinstance(record.get("history"), list):
        raise RuntimeError(f"Missing authenticated forecast history for {q.id_of_question}")
    latest = record.get("latest")
    if not record["history"] and latest is None:
        return None
    if not isinstance(latest, dict) or not isinstance(latest.get("forecast_values"), list):
        raise RuntimeError(f"Latest forecast values unavailable for {q.id_of_question}")
    return latest


def same_cdf(left, right):
    return len(left) == len(right) and max(abs(a - b) for a, b in zip(left, right)) < 0.00001


def numeric_cdf(q, quantiles):
    percentiles = [Percentile(percentile=p, value=v) for p, v in sorted(quantiles.items())]
    forecast = NumericDistribution.from_question(percentiles, q)
    values = [p.percentile for p in forecast.get_cdf()]
    if len(values) != q.cdf_size or not all(0 <= a <= b <= 1 for a, b in zip(values, values[1:])):
        raise ValueError("Invalid SDK continuous CDF")
    return values


def options_for(estimate, widths, as_of, resolve):
    return {"observations": estimate.sample_size, "quote_age_days": (as_of.date() - estimate.source_date).days,
            "days_until_measurement": (resolve.date() - as_of.date()).days, "choices": [
        {"width": w, "quantiles": list(estimate.quantiles(w).values())} for w in widths]}


class ReviewerUnavailable(RuntimeError):
    """All attempted named routes failed before returning any reviewer answer."""


async def review(routes, rows):
    llm = None
    all_unavailable = True
    prompt = (
        "You calibrate interval forecasts. Each row is anonymous. Its options are validated "
        "alternative interval widths for the same statistical forecast. You can and should abstain if the evidence is insufficient for a trustworthy numeric distribution. Quantiles are ordered "
        "at 10%, 25%, 50%, 75%, 90%. Review the quantile spacing, data age, sample size and forecast horizon. Choose null for stale data, incoherent quantiles, or inadequate uncertainty coverage. Otherwise choose a listed width. Use 1.0 where "
        "available unless uncertainty warrants a wider choice. Output only a JSON array of "
        "numbers or null values, one per row in order. No prose.\n" + json.dumps(rows)
    )
    skip_openrouter = False
    for name in sorted(routes, key=lambda route: route == groq_free.ROUTE):
        if skip_openrouter and name.startswith("openrouter/"):
            continue
        if name == groq_free.ROUTE and not all_unavailable:
            continue
        try:
            if name == groq_free.ROUTE:
                result, _ = await asyncio.to_thread(groq_free.GroqFreeRoute().complete, prompt, max_tokens=512)
            else:
                if name not in PRIMARY_ROUTES:
                    raise ValueError("Unapproved Market Pulse reviewer")
                if llm is None:
                    llm = FreeForecastLlm()
                    llm.allow_cpu_fallback = False
                llm.routes = [GeneralLlm(name, temperature=0, timeout=90, allowed_tries=1, num_retries=0)]
                llm.available_at = {name: 0.0}
                result = await llm.invoke(prompt)
        except Exception as exc:
            # The inherited router marks recognized HTTP, quota and connection errors by
            # installing a cooldown. Programming, configuration and output errors must
            # never be mistaken for a reason to try a different forecaster.
            cooldown = llm.available_at.get(name, 0) if llm else 0
            if isinstance(exc, ValueError) or not (cooldown > 0 or isinstance(
                    exc, (groq_free.Unavailable, TimeoutError, ConnectionError))):
                all_unavailable = False
            log.warning("Free reviewer %s unavailable (%s)", name, type(exc).__name__)
            if name.startswith("openrouter/") and cooldown - time.monotonic() > 300:
                skip_openrouter = True
            continue
        try:
            choices = json.loads(result.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
            if (not isinstance(choices, list) or len(choices) != len(rows)
                    or any(w is not None and (type(w) not in (int, float) or w not in [x["width"] for x in row["choices"]])
                           for w, row in zip(choices, rows))):
                raise ValueError("Model selected invalid intervals")
            return name, choices
        except (ValueError, TypeError, AttributeError) as exc:
            all_unavailable = False
            log.warning("Invalid interval response from %s (%s)", name, type(exc).__name__)
    if all_unavailable:
        raise ReviewerUnavailable("Qualified free reviewers unavailable")
    raise RuntimeError("No valid free reviewer decision")


def configured_routes():
    names = [name for key, names in MODEL_NAMES.items() if os.getenv(key) for name in names]
    if groq_free.configured():
        names.append(groq_free.ROUTE)
    return names


async def qualify_reviewers(rows):
    async def qualify(names):
        decisions = []
        selected = names
        for row in rows:
            # Every decision in a qualification belongs to the same named model.
            try:
                name, [choice] = await review(selected, [row])
            except ReviewerUnavailable:
                if None in decisions:
                    raise RuntimeError("Unavailable after a recorded abstention") from None
                raise
            selected = [name]
            decisions.append(choice)
        return selected[0], decisions

    primaries = [names for key, names in MODEL_NAMES.items() if os.getenv(key)]
    trials = await asyncio.gather(*(qualify(names) for names in primaries), return_exceptions=True)
    if groq_free.configured() and all(isinstance(t, ReviewerUnavailable) for t in trials):
        # Restart the full sealed cohort. Results from an interrupted provider
        # cannot be combined with the fallback model's results.
        fallback = await asyncio.gather(qualify([groq_free.ROUTE]), return_exceptions=True)
        trials.extend(fallback)
    return trials


def checked_proof(proof):
    sha = os.getenv("MARKET_CODE_SHA", os.getenv("GITHUB_SHA"))
    if not sha or proof.get("code_sha") != sha or proof.get("project") != PROJECT or not proof.get("dry"):
        raise ValueError("Evidence missing or for a different project or commit")
    valid_names = set(configured_routes())
    routes = [name for name in proof["routes"] if name in valid_names]
    if not routes:
        raise RuntimeError("No tested named free route available")
    return routes


def cross_quarter_menu():
    evidence = json.loads(Path(__file__).with_name("market_pulse_q3.json").read_text())
    groups = {}
    for case in evidence["cases"]:
        if case.get("status") != "scored":
            continue
        group_id = str(case["model_mapped_q4"])
        for width, mass in case["candidate_mass"].items():
            groups.setdefault((group_id, float(width)), []).append(math.log(mass / case["baseline_mass"]))
    result = {}
    for (group, width), gains in groups.items():
        # Require all six Q3 periods, a positive average, and a majority of individual wins.
        if len(gains) == 6 and sum(gains) > 0 and sum(v > 0 for v in gains) > len(gains)/2:
            result.setdefault(group, []).append(width)
    log.info("Independent Q3 validation approved widths=%s. Community comparison=%s",
             result, evidence["summary"])
    return result


async def validate(account, history=None):
    history = prices.HISTORY if history is None else history
    # Spot-check the exact authenticated history shape before any hosted inference.
    sample = next((q for q in account.group(46015).values()
                   if q.api_json["question"]["status"] in ("open", "upcoming")), None)
    if sample:
        log.info("Authenticated group preflight question=%s cdf_size=%s previous=%s",
                 sample.id_of_question, sample.cdf_size, latest_forecast(sample) is not None)
    stats = prices.benchmark(history)
    external = cross_quarter_menu()
    widths = {str(g): sorted(set(stats["allowed_widths"][str(g)]) & set(external.get(str(g), [])))
              for g in stats["eligible"]}
    widths = {g: ws for g, ws in widths.items() if ws}
    log.info("Historical scorecard=%s", json.dumps(stats, default=str))
    if not widths:
        raise RuntimeError("No market family passed the historical gate")
    historical = sorted((h for h in history if str(h["post_id"]) in widths), key=lambda h: h["question_id"])
    # Reconstruct every eligible case with information available strictly before its scoring time.
    # Neither the identity, date nor resolution is sent to the reviewer.
    rows = []
    scored = []
    for h in historical:
        estimate = prices.model(h["post_id"], h["spot"] - timedelta(seconds=1), h["spot"], h["resolve"])
        rows.append(options_for(estimate, widths[str(h["post_id"])], h["spot"], h["resolve"]))
        scored.append((h, estimate))
    cases = {row["question_id"]: row for row in stats["cases"]}
    trials = await qualify_reviewers(rows)
    by_group = {}
    routes_by_group = {}
    for trial in trials:
        if isinstance(trial, BaseException):
            log.info("Provider failed closed: %s", type(trial).__name__)
            continue
        name, picks = trial
        chosen = {}
        abstained = set()
        for (h, est), width in zip(scored, picks):
            group = str(h["post_id"])
            if width is None:
                abstained.add(group)
                log.info("Hosted reviewer abstained on historical question=%s", h["question_id"])
                continue
            gain = cases[h["question_id"]]["widths"][width]
            chosen.setdefault(group, []).append((h["spot"], gain))
            by_group.setdefault(group, set()).add(h["question_id"])
            log.info("Blind historical model=%s question=%s data=%s actual=%s width=%s log_gain=%f",
                     name, h["question_id"], est.source_date, h["outcome"], width, gain)
        for group, gains in chosen.items():
            gains.sort()
            if group not in abstained and len(gains) >= 3 and sum(v for _,v in gains[:-1]) > 0 and gains[-1][1] > 0:
                routes_by_group.setdefault(group, []).append(name)
    if not any(not isinstance(trial, BaseException) for trial in trials):
        raise RuntimeError("No free hosted provider returned valid decisions; no forecasts can be published")
    if not routes_by_group:
        raise RuntimeError("Hosted forecasts failed the held-out baseline; publication disabled")
    proof = {"code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "project": PROJECT,
             "routes_by_group": routes_by_group, "routes": sorted({n for ns in routes_by_group.values() for n in ns}),
             "allowed_widths": {g:widths[g] for g in routes_by_group},
             "q3_evidence_sha256": hashlib.sha256(Path(__file__).with_name("market_pulse_q3.json").read_bytes()).hexdigest(),
             "historical": {g: sorted(by_group[g]) for g in routes_by_group}}
    # A second, fresh inference exercises the prospective path, with publication disabled.
    rehearsal = await forecast(account, proof, dry=True)
    if rehearsal != 1:
        raise RuntimeError("No forward dry run could be validated")
    proof["dry"] = True
    Path("market-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    log.info("MARKET_PROOF=%s", json.dumps(proof, separators=(",", ":")))


async def forecast(account, proof, dry=False):
    now = datetime.now(UTC)
    routes = proof["routes"] if dry else checked_proof(proof)
    candidates = []
    failed = 0
    for group_id in map(int, proof["allowed_widths"]):
        try:
            group = account.group(group_id)
        except (requests.RequestException, ValueError, KeyError) as exc:
            log.warning("Group %s unavailable (%s)", group_id, type(exc).__name__)
            failed += 1
            continue
        for q in group.values():
            raw = q.api_json["question"]
            if raw["status"] not in (("open", "upcoming") if dry else ("open",)):
                continue
            spot = timestamp(raw["spot_scoring_time"])
            deadline = min(spot, timestamp(raw["scheduled_close_time"]))
            if deadline <= now + timedelta(minutes=5) or (not dry and deadline-now > timedelta(days=7)):
                continue
            candidates.append((deadline, q, spot))
    completed = 0
    for deadline, q, spot in sorted(candidates, key=lambda item: (item[0], item[1].id_of_question)):
        try:
            completed += await forecast_one(account, proof, routes, q, spot, deadline, now, dry)
        except (ValueError, RuntimeError, requests.RequestException, OSError) as exc:
            log.warning("Group %s question %s failed closed (%s: %s)", q.id_of_post, q.id_of_question,
                        type(exc).__name__, str(exc)[:240])
            failed += 1
            continue
        if dry and completed:
            break
    if failed and (not dry or not completed):
        raise RuntimeError(f"{failed} eligible Market Pulse update(s) failed closed")
    log.info("Project=%s open eligible updates=%d published=%d cost=0", PROJECT, len(candidates), 0 if dry else completed)
    return completed


async def forecast_one(account, proof, routes, q, spot, deadline, now, dry):
    group = str(q.id_of_post)
    estimate = prices.model(q.id_of_post, now, spot, prices.period_end(q.group_question_option, spot))
    previous = latest_forecast(q)
    log.info("Authenticated history question=%s previous=%s", q.id_of_question, previous is not None)
    # If today's admissible CDFs include the stored forecast, no model call or update
    # is needed. This also works after process restarts without an external ledger.
    if previous and not dry and any(
        same_cdf(numeric_cdf(q, estimate.quantiles(w)), previous["forecast_values"])
        for w in proof["allowed_widths"][group]
    ):
        log.info("Previous CDF still current; skipped question=%s source_date=%s", q.id_of_question, estimate.source_date)
        return 0
    options = options_for(estimate, proof["allowed_widths"][group], now, prices.period_end(q.group_question_option, spot))
    name, [width] = await review([r for r in proof["routes_by_group"][group] if r in routes], [options])
    if width is None:
        log.info("Reviewer abstained question=%s; no publication", q.id_of_question)
        return 0
    quantiles = estimate.quantiles(width)
    values = numeric_cdf(q, quantiles)
    log.info("mode=%s group=%s question=%s spot=%s model=%s source_date=%s sources=%s "
             "cdf_points=%d cdf_sha256=%s",
             "dry" if dry else "live", group, q.id_of_question, spot.isoformat(), name,
             estimate.source_date, estimate.sources, len(values), digest(values))
    if dry:
        return 1
    # Recheck the individual authenticated history and deadline immediately before submission.
    fresh = account.group(q.id_of_post)[q.id_of_question]
    if (fresh.question_type != q.question_type
            or fresh.api_json["question"]["scaling"] != q.api_json["question"]["scaling"]):
        raise ValueError("Market Pulse geometry changed before submission")
    latest = latest_forecast(fresh)
    if fresh.api_json["question"]["status"] != "open" or datetime.now(UTC) >= deadline:
        log.info("Closed before submission, skipped question=%s", q.id_of_question)
        return 0
    if latest and (same_cdf(values, latest["forecast_values"]) or
                   (not previous or latest["start_time"] != previous["start_time"])):
        log.info("Identical or concurrently updated forecast, skipped question=%s", q.id_of_question)
        return 0
    account.submit_once(fresh, values)
    return 1


async def monitor(account):
    # The scheduled monitor never calls the publication function. Re-test after a new
    # numeric outcome lands, so a small historical sample need not stay frozen forever.
    history = []
    for group in prices.SPECS:
        for q in account.group(group).values():
            raw = q.api_json["question"]
            if raw["status"] != "resolved" or not q.open_lower_bound or not q.open_upper_bound or q.zero_point is not None:
                continue
            try:
                actual = float(raw["resolution"])
            except (TypeError, ValueError):
                continue
            spot = timestamp(raw["spot_scoring_time"])
            history.append(dict(post_id=group, question_id=q.id_of_question, spot=spot, label=q.group_question_option,
                                resolve=prices.period_end(q.group_question_option, spot), low=q.lower_bound,
                                high=q.upper_bound, bins=q.cdf_size-1, outcome=actual))
    if not ({h["question_id"] for h in history} - {h["question_id"] for h in prices.HISTORY}):
        log.info("No newly resolved cases; publication disabled; historical cohort size=%d", len(history))
        return
    await validate(account, history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["validate", "monitor", "live"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    log.setLevel(logging.INFO)
    if datetime.now(UTC) >= datetime(2027, 1, 1, tzinfo=UTC):
        log.info("Market Pulse 26Q4 has ended")
        return
    account = Account()
    if args.mode == "validate":
        asyncio.run(validate(account))
    elif args.mode == "monitor":
        asyncio.run(monitor(account))
    else:
        proof = json.loads(Path("market-proof.json").read_text())
        asyncio.run(forecast(account, proof))


if __name__ == "__main__":
    main()
