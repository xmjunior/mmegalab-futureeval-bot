"""Independent checks of additional research sources with redacted receipts."""
import argparse
import csv
import hashlib
import http.client
import json
import os
import re
import time
from datetime import date
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlencode

from resource_access import QUERY

HERE = Path(__file__).resolve().parent
GROQ_MODEL = "openai/gpt-oss-120b"
EXA_MCP = "/mcp?tools=web_search_exa"
OFFICIAL = re.compile(r"https://(?:www\.)?federalreserve\.gov/newsevents/pressreleases/monetary(2026\d{4})a\.htm")


def fetch(host, path, headers=None, body=None):
    # Never follow a redirect, print a URL with a key or include remote text in exceptions.
    headers = {"Accept": "application/json", "User-Agent": "MEGALAB-source-trials", **(headers or {})}
    if body is not None:
        headers["Content-Type"] = "application/json"
    conn = http.client.HTTPSConnection(host, timeout=25)
    try:
        conn.request("GET" if body is None else "POST", path,
                     json.dumps(body) if body is not None else None, headers)
        response = conn.getresponse()
        raw = response.read(2_000_001)
        if not 200 <= response.status < 300:
            raise RuntimeError("http_" + str(response.status))
        if len(raw) > 2_000_000:
            raise ValueError("response_limit")
        if "text/event-stream" in response.getheader("Content-Type", ""):
            msgs = [json.loads(line[5:]) for line in raw.splitlines() if line.startswith(b"data:")]
            data = next(row for row in msgs if row.get("id") == body.get("id"))
        else:
            data = json.loads(raw) if raw else {}
        return data, {k.lower():v for k,v in response.getheaders()}
    finally:
        conn.close()


def fred_vintage(series, asof, start):
    params = urlencode(dict(series_id=series, realtime_start=asof, realtime_end=asof,
                            observation_start=start, observation_end=asof, file_type="json",
                            api_key=os.environ["FRED_API_KEY"]))
    data, _ = fetch("api.stlouisfed.org", "/fred/series/observations?" + params)
    if data.get("realtime_start") != asof or data.get("realtime_end") != asof:
        raise ValueError("unexpected_vintage")
    rows = data["observations"]
    if not rows or len(rows) != data["count"]:
        raise ValueError("incomplete_vintage")
    valid = {}
    for row in rows:
        d = date.fromisoformat(row["date"]).isoformat()
        if not start <= d <= asof or d in valid:
            raise ValueError("unexpected_observation")
        if row["value"] != ".":
            v = Decimal(row["value"])
            if not v.is_finite():
                raise ValueError("invalid_yield")
            valid[d] = v
    return valid


def fred_trial():
    cutoff = "2026-11-23"
    with (HERE / "promo/treasury-asof.csv").open() as f:
        frozen = {r["observation_date"]:r for r in csv.DictReader(f)}
    stats = {}
    for series, field in [("DGS2", "two_year_percent"), ("DGS10", "ten_year_percent")]:
        data = fred_vintage(series, cutoff, min(frozen))
        common = sorted(data.keys() & frozen.keys())
        if not common:
            raise ValueError("empty_comparison")
        diffs = [abs(data[d] - Decimal(frozen[d][field])) for d in common]
        stats[series] = dict(comparable=len(common), identical=diffs.count(0),
                             max_difference_bp=str(100 * max(diffs)),
                             first=common[0], last=common[-1], missing=len(frozen)-len(common))
    return dict(as_of=cutoff, series=stats)


def exa_auth():
    data, _ = fetch("api.exa.ai", "/websets/v0/websets?limit=1",
                    {"x-api-key":os.environ["EXA_API_KEY"]})
    if not isinstance(data.get("data"), list):
        raise ValueError("invalid_list")
    return {"authenticated":True, "billable_searches":0}


def exa_free_research():
    headers = {"Accept":"application/json, text/event-stream"}

    def rpc(method, params, n=None):
        req = dict(jsonrpc="2.0", method=method, params=params)
        if n is not None:
            req["id"] = n
        msg, hdr = fetch("mcp.exa.ai", EXA_MCP, headers, req)
        if "mcp-session-id" in hdr:
            headers["Mcp-Session-Id"] = hdr["mcp-session-id"]
        if n is None:
            return {}
        if msg.get("id") != n or "result" not in msg:
            raise ValueError("mcp_call_failed")
        return msg["result"]

    result = rpc("initialize", {"protocolVersion":"2025-03-26", "capabilities":{},
                  "clientInfo":{"name":"megalab-research","version":"1"}}, 1)
    headers["MCP-Protocol-Version"] = result["protocolVersion"]
    rpc("notifications/initialized", {})
    result = rpc("tools/list", {}, 2)
    if not any(t["name"] == "web_search_exa" for t in result["tools"]):
        raise ValueError("search_unavailable")
    result = rpc("tools/call", {"name":"web_search_exa","arguments":{"query":QUERY,"numResults":3}}, 3)
    if result.get("isError"):
        raise ValueError("free_search_error")
    raw = " ".join(x.get("text","") for x in result["content"] if x.get("type") == "text")
    urls = sorted({m[0] for m in OFFICIAL.finditer(raw) if "20261001" <= m[1] <= "20261123"})
    return {"sources":urls, "eligible_official_source":bool(urls), "credentialed_searches":0}


def groq_catalog():
    data, hdr = fetch("api.groq.com","/openai/v1/models",
                      {"Authorization":"Bearer "+os.environ["GROQ_API_KEY"]})
    models = {m["id"] for m in data["data"] if m.get("active", True)}
    if GROQ_MODEL not in models:
        raise ValueError("reasoning_model_unavailable")
    return {"model":GROQ_MODEL, "authenticated":True, "inferences":0,
            "account_tier":hdr.get("x-groq-tier") if hdr.get("x-groq-tier") in ("free","developer") else None}


if __name__ == "__main__":
    jobs = {"fred":fred_trial,"exa_auth":exa_auth,"exa_search":exa_free_research,"groq":groq_catalog}
    p = argparse.ArgumentParser()
    p.add_argument("provider", choices=jobs)
    name = p.parse_args().provider
    start = time.monotonic()
    try:
        record = {"result":"ok", **jobs[name]()}
    except Exception as err:
        reason = str(err) if isinstance(err, (RuntimeError, ValueError)) and re.fullmatch(r"[a-z0-9_]{1,55}",str(err)) else type(err).__name__
        record = {"result":"failed", "reason":reason}
    print(json.dumps({"provider":name,"elapsed_s":round(time.monotonic()-start,3),**record},sort_keys=True))
    raise SystemExit(record["result"] != "ok")
