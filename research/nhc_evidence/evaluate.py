"""Closed-source audit; generate artifacts containing only observations and hashes."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener

from evidence import DISCUSSION, NoRedirect, Unsupported, explicit_urls, fetch_public, parse_bulletin

YEAR = 2025
DECK_BASE = f"https://ftp.nhc.noaa.gov/atcf/archive/{YEAR}/"


def download(url, limit, cache):
    name = hashlib.sha256(url.encode()).hexdigest()
    path = cache / name
    if path.is_file():
        return path.read_bytes()
    for attempt in range(2):
        try:
            data = fetch_public(url, limit)
        except (OSError, URLError) as exc:
            if attempt or isinstance(exc, HTTPError) and exc.code not in (408, 429, 500, 502, 503, 504):
                raise
            time.sleep(2)
        else:
            path.write_bytes(data)
            return data


def truth_at(data, storm, when):
    wind = set()
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
        text = stream.read(2_000_001)
    if len(text) > 2_000_000:
        raise Unsupported("oversized decompressed best track")
    for row in csv.reader(text.decode("ascii").splitlines(), skipinitialspace=True):
        if len(row) < 9 or row[0] != "AL" or row[1] != storm[2:4] or row[4].strip() != "BEST":
            continue
        if row[2] != when.strftime("%Y%m%d%H") or when.minute or int(row[3].strip() or 0) != 0 or int(row[5]) != 0:
            continue
        v = int(row[8])
        if v > 0:
            wind.add(v)
    if len(wind) != 1:
        raise Unsupported("missing or ambiguous exact-time best-track intensity")
    return wind.pop()


def stats(rows, complete):
    cases = [r for r in rows if "verified_wind_kt" in r]
    n = len(cases)
    if not n:
        return {"paired_storms": 0, "scientific_gate": "insufficient"}
    delta = [abs(r["initial_wind_kt"] - r["verified_wind_kt"]) -
             abs(r["official_24h_wind_kt"] - r["verified_wind_kt"]) for r in cases]
    wins, losses = sum(x > 0 for x in delta), sum(x < 0 for x in delta)
    informative = wins + losses
    p = sum(math.comb(informative, j) for j in range(wins, informative + 1)) / 2**informative
    return {"paired_storms": n, "wins": wins, "ties": n-wins-losses, "losses": losses, "sign_p_one_sided": p,
            "mae_persistence_kt": statistics.mean(abs(r["initial_wind_kt"] - r["verified_wind_kt"]) for r in cases),
            "mae_official_kt": statistics.mean(abs(r["official_24h_wind_kt"] - r["verified_wind_kt"]) for r in cases),
            "mean_paired_improvement_kt": statistics.mean(delta),
            "scientific_gate": "pass" if complete and n >= 8 and sum(delta) > 0 and p <= .05 else "not_passed"}


def closed_season(cache):
    archive = download(DECK_BASE, 900_000, cache)
    filenames = sorted(set(re.findall(fr'href="(bal[0-4]\d{YEAR}\.dat\.gz)"', archive.decode(), re.I)))
    if not filenames:
        raise Unsupported("official season index returned no numbered Atlantic systems")
    rows = []
    complete = True
    for filename in filenames:
        num = filename[3:5]
        storm = "AL" + num + str(YEAR)
        url = f"https://www.nhc.noaa.gov/archive/{YEAR}/al{num}/al{num}{YEAR}.discus.001.shtml"
        row = {"storm": storm, "url": url, "verifier_url": DECK_BASE + filename}
        rows.append(row)
        try:
            raw = download(url, 250000, cache)
            row.update(parse_bulletin(url, raw))
            verified = download(row["verifier_url"], 600000, cache)
            row["verifier_sha256"] = hashlib.sha256(verified).hexdigest()
            row["verified_wind_kt"] = truth_at(verified, storm, datetime.fromisoformat(row["target_utc"]))
        except (OSError, ValueError) as exc:
            if isinstance(exc, HTTPError) and exc.code in (404, 410):
                why = "archived endpoint absent"
            elif isinstance(exc, Unsupported):
                why = str(exc)
                if not why.startswith("missing or ambiguous exact-time best-track") and "24H wind" not in why and why != "undated correction":
                    complete = False
            else:
                complete = False
                why = f"{type(exc).__name__} {getattr(exc, 'code', '')}".strip()
            row["exclusion"] = why
    return {"year": YEAR, "index_url": DECK_BASE, "index_sha256": hashlib.sha256(archive).hexdigest(),
            "universe": len(rows), "technical_retrieval_complete": complete, "rows": rows,
            "statistics": stats(rows, complete), "spend_usd": 0, "promote_to_tournament": False}


def applicability():
    # Select only a registered, resolved sample. Never retain question content, predictions or crowd aggregates.
    pid = 33121
    url = f"https://www.metaculus.com/api/posts/?tournaments={pid}&statuses=resolved&include_descriptions=true&with_cp=false&limit=100"
    headers = {"Accept-Language": "en", "User-Agent": "MEGALAB-closed-evidence-audit"}
    if token := os.getenv("METACULUS_TOKEN"):
        headers["Authorization"] = "Token " + token
    with build_opener(NoRedirect()).open(Request(url, headers=headers), timeout=30) as response:
        payload = json.load(response)
    if not isinstance(payload.get("results"), list):
        raise Unsupported("unsupported closed question list")
    items = {}
    for post in payload["results"]:
        tournaments = (post.get("projects") or {}).get("tournament", [])
        if pid not in {p["id"] for p in tournaments}:
            raise Unsupported("tournament identity missing or incorrect")
        for row in [post.get("question")] + (post.get("group_of_questions") or {}).get("questions", []):
            if not row or row.get("status") != "resolved":
                continue
            if not isinstance(row.get("resolution_criteria"), str):
                raise Unsupported("resolved question criteria absent")
            contract = " ".join(str(row.get(k) or "") for k in ("title", "resolution_criteria", "fine_print"))
            items[row["id"]] = bool(len(explicit_urls(contract)) == 1)
    return {"url": url, "first_page_posts": len(payload["results"]), "resolved_distinct_question_ids": sorted(items),
            "evaluated_resolved": len(items), "explicit_eligible_contracts": sum(items.values()),
            "sample": "First page of up to 100 returned resolved posts, unique resolved direct and grouped questions only; conditionals excluded.",
            "next_page": bool(payload.get("next")), "checked_utc": datetime.now(timezone.utc).isoformat(),
            "status": "measured" if items else "unknown"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("season", "applicability"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/nhc-2025-public"))
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    try:
        result = closed_season(args.cache) if args.mode == "season" else applicability()
    except (ValueError, OSError) as exc:
        result = {"status": "unknown", "error": f"{type(exc).__name__}: {exc}", "promote_to_tournament": False}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("rows", "resolved_distinct_question_ids")}))
    if result.get("status") == "unknown":
        raise SystemExit(1)
