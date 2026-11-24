"""Issue and score original, prospective Treasury cases. No Metaculus endpoints."""
import argparse
import ast
import hashlib
import json
import math
import os
import re
import statistics
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_SHA = "0459814720cd51c3effa7cfa412cec8aefcdb105"
BASE_MAIN_HASH = "4fd58c14db32026ba3563d222cdf454a0bbefc31a2262dc3f122babbebbdb2fc"
TREASURY = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value="
ORIGIN = "https://openrouter.ai/api/v1/"
GOOGLE = "https://generativelanguage.googleapis.com/v1beta/models/"
GEMMAS = ("gemma-4-31b-it", "gemma-4-26b-a4b-it")
GOOGLE_PRICING = "https://aistudio.google.com/docs/pricing.md"
SOFR = "https://markets.newyorkfed.org/api/rates/secured/sofr/last/5.json"
CANDIDATE = "nvidia/nemotron-3-ultra-550b-a55b:free"
# Named base models appear on the 2026-11-24 FutureEval model leaderboard.
# A base listing, score, credit balance, or paid endpoint never enables these IDs.
FRONTIERS = (
    "anthropic/claude-opus-5:free",
    "openai/gpt-5.6-sol:free",
    "moonshotai/kimi-k3:free",
)
METHODS = ("champion", "A", "B", "C", "D", "E", "F", "coin")
HORIZONS = (3, 7, 14)


def now_utc():
    return datetime.now(timezone.utc)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def write_ledger(path, book):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(book, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


class NoAuthRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("credentialed_redirect_refused")


def request(url, body=None, headers=None, limit=3_000_000, timeout=30):
    hdr = {"User-Agent": "MEGALAB-public-economics-research/1", **(headers or {})}
    if body is not None:
        hdr["Content-Type"] = "application/json"
    req = urllib.request.Request(url, headers=hdr, data=None if body is None else json.dumps(body).encode())
    has_secret = any(name.lower() in {"authorization", "x-goog-api-key"} for name in hdr)
    opener = urllib.request.build_opener(NoAuthRedirect()).open if has_secret else urllib.request.urlopen
    with opener(req, timeout=timeout) as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("response_too_large")
    return raw


def parse_sheet(raw):
    result = {}
    for row in ET.fromstring(raw).findall(".//{*}properties"):
        day = date.fromisoformat(row.findtext("{*}NEW_DATE")[:10])
        ten, two = row.findtext("{*}BC_10YEAR"), row.findtext("{*}BC_2YEAR")
        if not ten or not two:
            continue
        values = tuple(Decimal(s) for s in (ten, two))
        if not all(v.is_finite() and -20 <= v <= 100 for v in values):
            raise ValueError("invalid_rate")
        if day in result and result[day] != values:
            raise ValueError("duplicate_conflicting_observation")
        result[day] = values
    if not result:
        raise ValueError("empty_year")
    return result


class Treasury:
    def __init__(self):
        self.years = {}
        self.hashes = {}

    def year(self, year):
        if year not in self.years:
            url = TREASURY + str(year)
            for attempt in range(2):
                try:
                    raw = request(url)
                    self.years[year] = parse_sheet(raw)
                    self.hashes[url] = digest(raw)
                    break
                except (urllib.error.URLError, TimeoutError):
                    if attempt:
                        raise
                    time.sleep(1)
        return self.years[year]


def sofr_evidence(today):
    try:
        raw = request(SOFR, timeout=10)
        rows = json.loads(raw)["refRates"]
        observations = [(r["effectiveDate"], float(r["percentRate"])) for r in rows if r["type"] == "SOFR"
                        and str(today-timedelta(days=10)) <= r["effectiveDate"] < str(today)]
        if not observations or not all(math.isfinite(rate) and -20 <= rate <= 100 for _, rate in observations):
            raise ValueError("invalid_sofr_observations")
        d, rate = max(observations)
        return {"url": SOFR, "document_sha256": digest(raw), "effective_date": d, "rate_percent": rate}
    except (OSError, ValueError, KeyError, TypeError):
        return {"url": SOFR, "status": "unavailable_at_issue"}


def issue(instant, source):
    today = instant.date()
    y, w, _ = today.isocalendar()
    case_id = f"ust10-v1-{y}-W{w:02d}"
    horizon = HORIZONS[(w - 1) % len(HORIZONS)]
    offset = Decimal("0.05") if w % 2 else Decimal("-0.05")
    history = source.year(today.year)
    used_years = [today.year]
    if sum(day < today for day in history) < 120:
        history = {**source.year(today.year - 1), **history}
        used_years.append(today.year - 1)
    days = sorted(d for d in history if d < today)[-252:]
    if len(days) < 100 or today - days[-1] > timedelta(days=7):
        raise ValueError("insufficient_or_stale_history")
    latest, two = history[days[-1]]
    threshold = latest + offset
    target = today + timedelta(days=horizon)
    lag = max(1, round(horizon * 5 / 7))
    # Disjoint historical intervals, oriented backward from the issue-date observation.
    ends = range(len(days) - 1, lag - 1, -lag)
    moves = [history[days[end]][0] - history[days[end-lag]][0] for end in ends]
    wins = sum(delta > offset for delta in moves)
    evidence = {
        "as_of": instant.isoformat(), "latest_observation": days[-1].isoformat(),
        "last_10y_percent": str(latest), "last_2y_percent": str(two),
        "10y_minus_2y_pp": str(latest - two), "lag_observations": lag,
        "five_observation_10y_change_pp": str(latest - history[days[-6]][0]),
        "disjoint_intervals": len(moves), "moves_above_threshold_offset": wins,
        "median_move_pp": str(statistics.median(moves)),
        "source_documents": {TREASURY + str(year): source.hashes[TREASURY + str(year)] for year in used_years},
        "window_start": days[0].isoformat(), "ny_fed_secured_overnight_rate": sofr_evidence(today),
    }
    q = {
        "question_text": f"Will the US Treasury 10-year par yield exceed {threshold}% on the first reported observation dated on or after {target}?",
        "background_info": f"The last available observation before issue was {latest}% on {days[-1]}. The US Treasury publishes the daily par yield curve.",
        "resolution_criteria": f"YES if the US Treasury daily BC_10YEAR is strictly above {threshold}% on the first available observation dated {target} through {target + timedelta(days=6)}, inclusive. Equality resolves NO.",
        "fine_print": "Use only the official US Treasury daily Treasury par yield curve XML. Cancel if the seven-calendar-day window has no observed rate. Lock the first outcome recorded after a two-day publication lag.",
    }
    return {"id": case_id, "issued_at": instant.isoformat(), "target": target.isoformat(),
            "source": TREASURY + str(target.year), "horizon_days": horizon, "threshold_percent": str(threshold),
            "question": q, "evidence": evidence, "strategies": {}, "attempts": [], "cache": {}, "resolution": None}


def production_prompts():
    # Read the template literals from the pinned production file. Only supported
    # substitutions are rendered. We never execute the source file or its expressions.
    source = (ROOT / "main.py").read_bytes()
    if digest(source) != BASE_MAIN_HASH:
        raise ValueError("production_prompt_source_hash_changed")
    methods = {"_get_research_prompt": "research", "_run_forecast_on_binary": "binary"}
    templates = {}
    for fn in ast.walk(ast.parse(source)):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or fn.name not in methods:
            continue
        for child in ast.walk(fn):
            if not isinstance(child, ast.Assign) or ast.unparse(child.targets[0]) != "prompt":
                continue
            literal = child.value.args[0]
            pieces = []
            for node in literal.values:
                if isinstance(node, ast.Constant):
                    pieces.append((None, node.value))
                else:
                    pieces.append((ast.unparse(node.value), ""))
            templates[methods[fn.name]] = pieces
    if set(templates) != {"research", "binary"}:
        raise ValueError("invalid_production_templates")
    return templates


def render(template, case, research=""):
    slots = {"question." + k: v for k, v in case["question"].items()}
    slots.update({"research": research, "datetime.now().strftime('%Y-%m-%d')": case["issued_at"][:10],
                  "self._get_conditional_disclaimer_if_necessary(question)": ""})
    raw = "".join(slots[variable] if variable else literal for variable, literal in template)
    lines = raw.split("\n")
    indent = max(len(line) - len(line.lstrip()) for line in lines[:2])
    return "\n".join(line[indent:] if len(line) - len(line.lstrip()) >= indent else line.lstrip() for line in lines)


def probability(text):
    last = text.strip().splitlines()[-1].strip().replace("**", "") if text.strip() else ""
    m = re.fullmatch(r"(?:Final answer:\s*)?Probability:\s*(\d+(?:\.\d+)?)%(?:,\s*0-100)?[.]?", last, re.I)
    if not m or not 0 <= float(m[1]) <= 100:
        raise ValueError("invalid_final_probability")
    return max(.01, min(.99, float(m[1]) / 100))


class Routes:
    def __init__(self):
        self.google_prices_checked = False
        self.free_catalog = None
        self.disabled = set()

    def ready(self, model):
        return bool(os.getenv("GEMINI_API_KEY" if model in GEMMAS else "OPENROUTER_API_KEY"))

    def verified_free(self, model):
        if model not in (CANDIDATE, *FRONTIERS):
            return False
        if self.free_catalog is None:
            items = json.loads(request(ORIGIN + "models", limit=8_000_000))["data"]
            self.free_catalog = {row["id"]: row.get("pricing", {}) for row in items}
        prices = self.free_catalog.get(model, {})
        if not isinstance(prices, dict) or not {"prompt", "completion"} <= prices.keys():
            return False
        try:
            return all(Decimal(str(v)).is_zero() for v in prices.values())
        except ArithmeticError:
            return False

    def call(self, model, prompt):
        if model in GEMMAS:
            if not self.google_prices_checked:
                raw = request(GOOGLE_PRICING, timeout=15).decode()
                marker = 'id="gemma-4"'
                if marker not in raw:
                    raise ValueError("zero_price_unverified")
                section = raw.split(marker, 1)[1].split("## ", 1)[0]
                for direction in ("Input", "Output"):
                    row = rf"\* {direction} price\s+\* Free of charge\s+\* Not available"
                    if not re.search(row, section):
                        raise ValueError("zero_price_unverified")
                self.google_prices_checked = True
            body = {"contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": .2, "maxOutputTokens": 1536}}
            reply = json.loads(request(GOOGLE + model + ":generateContent", body,
                                    {"x-goog-api-key": os.environ["GEMINI_API_KEY"]}, timeout=55))
            text = "\n".join(p["text"] for p in reply["candidates"][0]["content"]["parts"] if p.get("text") and not p.get("thought"))
            return text, reply.get("modelVersion", model), reply.get("usageMetadata", {})
        if model not in (CANDIDATE, *FRONTIERS) or not model.endswith(":free"):
            raise ValueError("model_outside_allowlist")
        if not self.verified_free(model):
            raise ValueError("zero_price_unverified")
        body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                "temperature": .2, "max_tokens": 1536,
                "provider": {"max_price": {"prompt": 0, "completion": 0, "request": 0}}}
        reply = json.loads(request(ORIGIN + "chat/completions", body,
                                  {"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]}, timeout=55))
        actual = reply["model"]
        if actual.removesuffix(":free") != model.removesuffix(":free"):
            raise ValueError("model_substitution")
        usage = reply.get("usage", {})
        if model in FRONTIERS and usage.get("cost") is None:
            raise ValueError("frontier_zero_cost_receipt_missing")
        if Decimal(str(usage.get("cost") or 0)) != 0:
            raise ValueError("provider_reported_nonzero_cost")
        return reply["choices"][0]["message"]["content"], actual, usage


def safe_status(exc):
    if isinstance(exc, urllib.error.HTTPError):
        return f"http_{exc.code}"
    if isinstance(exc, (TimeoutError, urllib.error.URLError, OSError)):
        return "network_or_timeout"
    if isinstance(exc, ValueError) and exc.args and re.fullmatch("[a-z_]+", str(exc.args[0])):
        return exc.args[0]
    return "invalid_provider_response"


def get_reply(case, model, key, prompt, routes, checkpoint):
    cache_key = f"{model}/{key}"
    if cache_key in case["cache"]:
        return case["cache"][cache_key]["text"]
    if not routes.ready(model) or model in routes.disabled:
        return None
    # Reports and forecasts are bounded jointly for each actual model; no loops on invalid output.
    if sum(a["model"] == model for a in case["attempts"]) >= 3:
        return None
    if any(a["model"] == model and a["status"] in {"http_429", "http_503", "network_or_timeout"}
           and a.get("retry_at") and now_utc() < datetime.fromisoformat(a["retry_at"]) for a in case["attempts"]):
        return None
    prior = [a for a in case["attempts"] if a["key"] == cache_key]
    if prior:
        last = prior[-1]
        if last["status"] not in {"http_429", "http_503", "network_or_timeout"}:
            return None
        retry_at = datetime.fromisoformat(last.get("retry_at", last["at"]))
        if now_utc() < retry_at:
            return None
    if now_utc().date().isoformat() != case["issued_at"][:10]:
        return None
    attempt = {"key": cache_key, "model": model, "at": now_utc().isoformat(),
               "prompt_sha256": digest(prompt.encode()), "status": "attempt_started"}
    case["attempts"].append(attempt)
    checkpoint()
    started = time.monotonic()
    try:
        reply, version, usage = routes.call(model, prompt)
        if not isinstance(reply, str) or not reply.strip() or len(reply) > 40_000:
            raise ValueError("empty_or_oversized_completion")
        if now_utc().date().isoformat() != case["issued_at"][:10]:
            raise ValueError("issue_window_ended")
        attempt.update(status="response", reported_model=version, output_sha256=digest(reply.encode()))
        attempt["usage"] = {k: v for k, v in usage.items() if k in {
            "cost", "prompt_tokens", "completion_tokens", "promptTokenCount", "candidatesTokenCount", "totalTokenCount"} and isinstance(v, (int, float))}
        # Cache text for deterministic replay, never publish free-form provider output in logs.
        case["cache"][cache_key] = {"text": reply}
    except (ValueError, KeyError, IndexError, TypeError, ArithmeticError, OSError) as exc:
        attempt["status"] = safe_status(exc)
        delay = 3600
        if isinstance(exc, urllib.error.HTTPError):
            requested = (exc.headers or {}).get("Retry-After", "")
            if str(requested).isdigit():
                delay = max(delay, int(requested))
        attempt["retry_at"] = (now_utc() + timedelta(seconds=min(delay, 86400))).isoformat()
        if attempt["status"] in {"http_429", "http_503", "http_401", "http_403", "network_or_timeout"}:
            routes.disabled.add(model)
    finally:
        attempt["seconds"] = round(time.monotonic() - started, 2)
        checkpoint()
    return case["cache"].get(cache_key, {}).get("text")


def record(case, name, p=None, reason="", **fields):
    old = case["strategies"].get(name, {})
    if "p" in old:
        return
    item = {"status": "issued" if p is not None else "unavailable", "reason": reason, **fields}
    if p is not None:
        if not math.isfinite(p) or not 0 < p < 1:
            raise ValueError("invalid_probability")
        item.update(p=p, timestamp=now_utc().isoformat())
    case["strategies"][name] = item


def adjudicate(case, routes, template, checkpoint):
    arms = case["strategies"]
    if "p" in arms.get("E", {}):
        return
    if any("p" not in arms.get(arm, {}) for arm in ("champion", "A")):
        record(case, "E", reason="disagreement_unobservable")
        return
    pair = [Decimal(str(arms[arm]["p"])) for arm in ("champion", "A")]
    if abs(pair[0] - pair[1]) < Decimal("0.20"):
        record(case, "E", reason="disagreement_below_0.20")
        return
    if now_utc().date().isoformat() != case["issued_at"][:10]:
        return
    # Budget is one inference per case, including failed requests and across reruns.
    prior = next((a for a in case["attempts"] if a["key"].endswith("/adjudication")), None)
    if prior:
        model = prior["model"]
        reply = case["cache"].get(prior["key"], {}).get("text")
    else:
        if not routes.ready(FRONTIERS[0]):
            record(case, "E", reason="frontier_key_unavailable")
            return
        try:
            model = next((m for m in FRONTIERS if routes.verified_free(m)), None)
        except (OSError, ValueError, KeyError, TypeError):
            record(case, "E", reason="frontier_free_catalog_unavailable")
            return
        if model is None:
            record(case, "E", reason="no_verified_free_frontier_route")
            return
        evidence = ("Adjudicate two forecasts for this same event. Consider the official as-of evidence and give your own final probability. "
                    f"Forecaster 1: {pair[0] * 100}%; forecaster 2: {pair[1] * 100}%. "
                    + json.dumps(case["evidence"], sort_keys=True))
        reply = get_reply(case, model, "adjudication", render(template, case, evidence), routes, checkpoint)
    try:
        p = probability(reply or "")
    except ValueError:
        record(case, "E", reason="adjudication_no_valid_free_response", model=model)
        return
    record(case, "E", p, "disagreement_triggered_frontier", model=model, inputs=["champion", "A"])


def forecast(case, routes, templates, checkpoint):
    e = case["evidence"]
    c = (e["moves_above_threshold_offset"] + 2) / (e["disjoint_intervals"] + 4)
    record(case, "coin", .5, "uninformative")
    record(case, "C", c, "disjoint_calendar_matched_changes_beta_2_2")
    r_prompt = render(templates["research"], case)
    for arm, models in (("champion", GEMMAS), ("A", (CANDIDATE,)), ("B", (CANDIDATE,))):
        if "p" in case["strategies"].get(arm, {}):
            continue
        status = "no_valid_hosted_response"
        for model in models:
            if arm == "B":
                research = "Verified official as-of Treasury data. Percentages are yield levels; changes are percentage points.\n" + json.dumps(e, sort_keys=True)
                stage = "grounded"
            else:
                research = get_reply(case, model, "research", r_prompt, routes, checkpoint)
                stage = "forecast"
            if not research:
                continue
            forecast_prompt = render(templates["binary"], case, research)
            reply = get_reply(case, model, stage, forecast_prompt, routes, checkpoint)
            if not reply:
                continue
            try:
                p = probability(reply)
            except ValueError:
                status = "invalid_final_probability"
                repair_prompt = (f'Question: {case["question"]["question_text"]}\n{case["question"]["resolution_criteria"]}\n'
                                 f'Analysis: {reply}\nSelect your final probability of YES as a whole percent.\n'
                                 'Return only a JSON array of 1 integers between 0 and 100, inclusive. Each number must be your own selection.')
                repair = get_reply(case, model, stage + "_repair", repair_prompt, routes, checkpoint)
                try:
                    values = json.loads((repair or "").strip().removeprefix("```json").removesuffix("```").strip())
                    if not isinstance(values, list) or len(values) != 1 or type(values[0]) is not int or not 0 <= values[0] <= 100:
                        raise ValueError("invalid_repair")
                    p = max(.01, min(.99, values[0] / 100))
                except (ValueError, TypeError):
                    continue
            record(case, arm, p, stage, model=model)
            break
        else:
            record(case, arm, reason=status)
    arms = case["strategies"]
    if "p" in arms["champion"] and "p" in arms["A"]:
        pair = [arms["champion"]["p"], arms["A"]["p"]]
        record(case, "D", statistics.mean(pair), "equal_linear_pool_two_independent_families", inputs=["champion", "A"])
    else:
        record(case, "D", reason="two_model_families_required")
    adjudicate(case, routes, templates["binary"], checkpoint)
    route = "C" if case["horizon_days"] <= 7 and e["disjoint_intervals"] >= 20 else "B"
    record(case, "F", arms[route].get("p"), "predeclared_horizon_and_analog_count", input=route)
    checkpoint()


def resolve(case, source, instant):
    if case["resolution"] is not None:
        return
    target = date.fromisoformat(case["target"])
    last = target + timedelta(days=6)
    today = instant.date()
    if today < target + timedelta(days=2):
        return
    # Read both years only if needed; don't dereference a future-year endpoint in December.
    rates = dict(source.year(target.year))
    if last.year != target.year and last.year <= today.year:
        rates.update(source.year(last.year))
    observed = sorted(d for d in rates if target <= d <= min(last, today - timedelta(days=2)))
    if observed:
        d = observed[0]
        value = rates[d][0]
        case["resolution"] = {"observed_date": str(d), "value_percent": str(value),
                              "outcome": int(value > Decimal(case["threshold_percent"])),
                              "retrieved_at": instant.isoformat(), "raw_sha256": source.hashes[TREASURY + str(d.year)]}
    elif today > last + timedelta(days=14) and max(rates) > last:
        case["resolution"] = {"void": True, "reason": "no_published_observation_in_window", "retrieved_at": instant.isoformat()}


def summarize(book):
    resolved = [c for c in book["cases"].values() if "outcome" in (c["resolution"] or {})]
    summary = {}
    for arm in METHODS:
        valid = [c for c in resolved if "p" in c["strategies"].get(arm, {})]
        scores = [(c["strategies"][arm]["p"], c["resolution"]["outcome"]) for c in valid]
        paired = [c for c in valid if "p" in c["strategies"].get("champion", {})]
        pairs = [((c["strategies"]["champion"]["p"] - c["resolution"]["outcome"])**2 -
                  (c["strategies"][arm]["p"] - c["resolution"]["outcome"])**2) for c in paired]
        summary[arm] = {
            "resolved_n": len(scores), "paired_with_champion_n": len(pairs),
            "brier": statistics.mean((p-y)**2 for p,y in scores) if scores else None,
            "nats_log_loss": statistics.mean(-(math.log(p) if y else math.log1p(-p)) for p,y in scores) if scores else None,
            "mean_forecast_minus_event_rate": statistics.mean(p-y for p,y in scores) if scores else None,
            "paired_brier_improvement": statistics.mean(pairs) if pairs else None,
        }
    return {"cases_issued": len(book["cases"]), "cases_resolved": len(resolved), "arms": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    path = parser.parse_args().ledger
    book = json.loads(path.read_text()) if path.exists() else {"schema": 1, "baseline_commit": BASE_SHA, "cases": {}}
    if book.get("schema") != 1 or book.get("baseline_commit") != BASE_SHA:
        raise ValueError("ledger_version_mismatch")
    instant = now_utc()
    source = Treasury()
    for case in book["cases"].values():
        resolve(case, source, instant)
    y, w, _ = instant.date().isocalendar()
    key = f"ust10-v1-{y}-W{w:02d}"
    if key not in book["cases"]:
        book["cases"][key] = issue(instant, source)
    case = book["cases"][key]
    if instant.date().isoformat() == case["issued_at"][:10]:
        forecast(case, Routes(), production_prompts(), lambda: write_ledger(path, book))
    book["summary"] = summarize(book)
    write_ledger(path, book)
    print(json.dumps({"case": key, "issued_at": case["issued_at"], "target": case["target"],
                      "statuses": {k: v["status"] for k,v in case["strategies"].items()},
                      "attempts": [{k: a[k] for k in ("model", "status")} for a in case["attempts"]],
                      "cases_resolved": book["summary"]["cases_resolved"]}, sort_keys=True))


if __name__ == "__main__":
    main()
