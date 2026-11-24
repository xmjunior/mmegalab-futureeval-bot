"""Prove provider access without inference. Optional, once-only Treasury search."""
import argparse
import base64
import http.client
import json
import os
import re
import ssl
from datetime import datetime, timezone

REPO = "xmjunior/mmegalab-futureeval-bot"
VERIFY = "verify/openai-access-audit"
RECORDS = "research/champion-observations"
TRIAL = "records/trials/20261124-treasury-tavily.json"
DEADLINE = "2026-11-24T23:59:59+00:00"
QUERY = "October 2026 FOMC statement October Federal Open Market Committee target range federal funds rate site:federalreserve.gov"


def request(host, path, token=None, method="GET", data=None):
    """Single HTTPS request to an explicit host. No redirects, retries or body logging."""
    headers = {"Accept": "application/json", "User-Agent": "megalab-independent-research"}
    if token:
        headers["Authorization"] = "Bearer " + token
    if data is not None:
        headers["Content-Type"] = "application/json"
    conn = http.client.HTTPSConnection(host, timeout=18, context=ssl.create_default_context())
    try:
        conn.request(method, path, None if data is None else json.dumps(data), headers)
        response = conn.getresponse()
        raw = response.read(1_500_001)
        if not 200 <= response.status < 300 or len(raw) > 1_500_000:
            return response.status, {}
        return response.status, json.loads(raw) if raw else {}
    finally:
        conn.close()


def value(name):
    return os.getenv(name, "").strip()


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def quota(payload):
    a, k = payload.get("account", {}), payload.get("key", {})
    return {"plan": a.get("current_plan"), "used": a.get("plan_usage"), "included": a.get("plan_limit"),
            "paygo_used": a.get("paygo_usage"), "paygo_limit": a.get("paygo_limit"),
            "key_used": k.get("usage"), "key_limit": k.get("limit")}


def free(q):
    # Check prepaid headroom directly. A null PAYGO cap does not establish that PAYGO is off.
    # The entire key allowance plus 250 other account credits must still fit inside the free plan.
    numbers = ("used", "included", "key_limit", "key_used", "paygo_used")
    if any(type(q.get(field)) is not int for field in numbers):
        return False
    return (str(q["plan"]).lower() in {"researcher", "free"} and q["paygo_used"] == 0
            and q["paygo_limit"] in (0, None) and 0 <= q["used"] < q["included"] <= 1000
            and 0 <= q["key_used"] < q["key_limit"] <= 700
            and q["included"] - q["used"] >= q["key_limit"] - q["key_used"] + 250)



def github(method, path, data=None):
    return request("api.github.com", "/repos/" + REPO + "/" + path, value("GH_TOKEN"), method, data)


def write_record(record, previous=None):
    data = json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n"
    body = {"branch": RECORDS, "message": "Record single Tavily Treasury trial: " + record["status"],
            "content": base64.b64encode(data.encode()).decode()}
    if previous:
        body["sha"] = previous
    return github("PUT", "contents/" + TRIAL, body)


def treasury_search(key, before):
    now = datetime.now(timezone.utc).isoformat()
    if not free(before) or now >= DEADLINE:
        emit("search_skipped", reason="free_quota_or_deadline", requests=0)
        return
    ref_status, ref = github("GET", "git/ref/heads/" + VERIFY)
    if ref_status != 200 or ref.get("object", {}).get("sha") != value("GITHUB_SHA"):
        raise ValueError("audit_ref_mismatch")
    opts = {"query": QUERY, "search_depth": "basic", "auto_parameters": False,
            "include_domains": ["federalreserve.gov"], "topic": "general",
            "start_date": "2026-10-01", "end_date": "2026-11-23",
            "include_usage": True, "include_answer": False, "max_results": 3}
    record = {"status": "reserved", "purpose": "Retrieve an official dated monetary-policy source for the prospective Treasury panel.",
              "research_only": True, "cutoff": "2026-11-23", "query": opts,
              "run": value("GITHUB_RUN_ID"), "commit": value("GITHUB_SHA"), "at": now,
              "free_credit_cap": 1, "before": before}
    status, reservation = write_record(record)
    if status != 201:
        existing_status, existing = github("GET", "contents/" + TRIAL + "?ref=" + RECORDS)
        if existing_status == 200 and existing.get("sha"):
            emit("search_skipped", reason="previous_reservation", requests=0)
            return
        raise ValueError("reservation_unavailable")
    prior = reservation["content"]["sha"]
    try:
        quota_status, current = request("api.tavily.com", "/usage", key)
        if quota_status != 200 or not free(quota(current)):
            record["status"] = "skipped_free_quota"
            persisted, _ = write_record(record, prior)
            if persisted != 200:
                raise ValueError("quota_stop_record_failed")
            emit("search_skipped", reason="free_quota_changed", requests=0)
            return
        status, result = request("api.tavily.com", "/search", key, "POST", opts)
        record.update(status="provider_error", http=status)
        if status == 200:
            # Only official, dated FOMC release links, no raw snippets or third-party text.
            urls = [r.get("url", "") for r in result.get("results", []) if isinstance(r, dict)]
            pattern = r"https://(?:www\.)?federalreserve\.gov/newsevents/pressreleases/monetary(2026\d{4})a\.htm"
            matches = sorted({u for u in urls if (m := re.fullmatch(pattern, u)) and "20261001" <= m[1] <= "20261123"})
            record.update(status="retrieved" if matches else "no_eligible_source", sources=matches,
                          candidates=len(urls), search_credits=result.get("usage", {}).get("credits"))
        usage_status, accounting = request("api.tavily.com", "/usage", key)
        record["usage_after_http"] = usage_status
        if usage_status == 200:
            record["after"] = quota(accounting)
    except (OSError, ValueError, KeyError, http.client.HTTPException) as err:
        record.update(status="indeterminate", error_type=type(err).__name__)
    persisted, _ = write_record(record, prior)
    if persisted != 200:
        raise ValueError("result_write_failed_reservation_retained")
    emit("treasury_trial", status=record["status"], source_count=len(record.get("sources", [])),
         search_credits=record.get("search_credits"), after=record.get("after"))
    if record["status"] in {"indeterminate", "provider_error"}:
        raise ValueError("search_provider_unavailable")


def run(provider, search=False):
    if provider == "asknews":
        key = value("ASKNEWS_API_KEY")
        if not key:
            emit("asknews", configured=False)
            return
        code, res = request("api.asknews.app", "/v1/profiles/me/limits", key)
        ok = code == 200 and "request_rate_capacity" in res
        emit("asknews", configured=True, http=code, authenticated=ok, news_queries=0)
        if not ok:
            raise ValueError("asknews_limits_unavailable")
    elif provider == "openai":
        key = value("OPENAI_API_KEY")
        if not key:
            emit("openai", configured=False, inference_requests=0)
            return
        code, res = request("api.openai.com", "/v1/models", key)
        ok = code == 200 and isinstance(res.get("data"), list)
        emit("openai", configured=True, http=code, authenticated=ok, inference_requests=0)
        if not ok:
            raise ValueError("openai_model_list_unavailable")
    else:
        key = value("TAVILY_API_KEY")
        source = value("TAVILY_SOURCE")
        if not key:
            emit("tavily", configured=False, search_requests=0)
            return
        if not key.startswith("tvly-"):
            raise ValueError("unexpected_tavily_credential_format")
        code, res = request("api.tavily.com", "/usage", key)
        ok = code == 200 and isinstance(res.get("key"), dict) and isinstance(res.get("account"), dict)
        emit("tavily", source=source, configured=True, authenticated=ok, http=code,
             quota=quota(res) if ok else None)
        if not ok:
            raise ValueError("tavily_usage_unavailable")
        if search:
            treasury_search(key, quota(res))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("provider", choices=["asknews", "openai", "tavily"])
    parser.add_argument("--search", action="store_true")
    args = parser.parse_args()
    run(args.provider, args.search)
