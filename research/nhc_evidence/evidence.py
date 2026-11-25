"""Retrieve a dated NHC wind forecast only from an explicitly cited discussion."""
from __future__ import annotations

import calendar
import hashlib
import re
from datetime import datetime, timedelta, timezone
from html import unescape
from urllib.error import URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

UTC = timezone.utc
PREFIX = "https://www.nhc.noaa.gov/archive/"
DISCUSSION = re.compile(
    r"https://www\.nhc\.noaa\.gov/archive/(20\d{2})/al(\d{2})/al\2\1\.discus\.(\d{3})\.shtml"
)
ZONES = {"AST": -4, "EDT": -4, "EST": -5, "CDT": -5, "CST": -6, "GMT": 0, "UTC": 0}


class Unsupported(ValueError):
    """Missing, ambiguous or out-of-scope official data."""


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def fetch_public(url, limit):
    req = Request(url, headers={"User-Agent": "MEGALAB-closed-data-evaluation/1.0"})
    with build_opener(NoRedirect()).open(req, timeout=12) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise Unsupported("oversized response")
    return data


def wind_row(text, label, issue):
    matches = re.findall(
        rf"(?m)^\s*{label}\s+(\d{{2}})/(\d{{2}})(\d{{2}})Z\s+\d+(?:\.\d+)?[NS]\s+"
        r"\d+(?:\.\d+)?[WE]\s+(\d+)\s+KT\b", text)
    if len(matches) != 1:
        raise Unsupported(f"missing or ambiguous {label} wind")
    day, hour, minute, wind = map(int, matches[0])
    valid = []
    for delta in range(-1, 4):
        date = (issue + timedelta(days=delta)).date()
        if date.day == day:
            valid.append(datetime(date.year, date.month, day, hour, minute, tzinfo=UTC))
    if len(valid) != 1 or not 0 < wind <= 250:
        raise Unsupported("implausible time or wind")
    return valid[0], wind


def parse_bulletin(url, data):
    match = DISCUSSION.fullmatch(url)
    if not match or "000" == match[3]:
        raise Unsupported("unrecognized publisher URL")
    year, number, advisory = map(int, match.groups())
    # The archive page includes navigation and other dates outside the bulletin.
    decoded = data.decode("ascii", errors="replace")
    blocks = re.findall(r"<pre\b[^>]*>(.*?)</pre>", decoded, re.S | re.I)
    if len(blocks) != 1:
        raise Unsupported("bulletin body missing or duplicated")
    body = unescape(re.sub(r"<[^>]+>", "", blocks[0])).replace("\r", "")
    if re.search(r"\bCORRECT(?:ED|ION)\b", body, re.I):
        raise Unsupported("undated correction")
    storm_id = f"AL{number:02}{year}"
    if not re.search(rf"(?m)^NWS National Hurricane Center[^\n]*\b{storm_id}\s*$", body, re.I):
        raise Unsupported("issuer or storm id mismatch")
    adv_match = re.search(r"(?im)^[^\n]*\bDiscussion Number\s+(\d+)\s*$", body)
    if not adv_match or int(adv_match[1]) != advisory:
        raise Unsupported("advisory sequence mismatch")
    header = re.search(r"(?im)^\s*(\d{1,4})\s+(AM|PM)\s+([A-Z]{3})\s+"
                       r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Z]{3})\s+(\d{1,2})\s+(\d{4})\s*$", body)
    if not header or header[3].upper() not in ZONES or int(header[6]) != year:
        raise Unsupported("ambiguous bulletin issue time")
    digits = header[1].zfill(3)
    hour, minute = int(digits[:-2]), int(digits[-2:])
    if not 1 <= hour <= 12:
        raise Unsupported("invalid clock")
    hour = hour % 12 + (12 if header[2].upper() == "PM" else 0)
    month = list(calendar.month_abbr).index(header[4].title())
    tz = timezone(timedelta(hours=ZONES[header[3].upper()]))
    issue = datetime(year, month, int(header[5]), hour, minute, tzinfo=tz).astimezone(UTC)
    initial, observed = wind_row(body, "INIT", issue)
    target, forecast = wind_row(body, "24H", issue)
    if abs(initial - issue) > timedelta(hours=3) or not timedelta(hours=15) <= target-issue <= timedelta(hours=27):
        raise Unsupported("bulletin time conflicts with forecast")
    return {"storm": storm_id, "advisory": advisory, "url": url,
            "source_sha256": hashlib.sha256(data).hexdigest(), "issue_utc": issue.isoformat(),
            "available_asof_assumption": (max(initial, issue) + timedelta(hours=1)).isoformat(),
            "target_utc": target.isoformat(), "initial_utc": initial.isoformat(),
            "initial_wind_kt": observed, "official_24h_wind_kt": forecast}


def explicit_urls(contract):
    if not re.search(r"\bwind(?:s)?\b", contract, re.I):
        return []
    tokens = re.findall(r"https?://[^\s<>\]\)]+", contract)
    urls = {t.rstrip(".,;") for t in tokens if DISCUSSION.fullmatch(t.rstrip(".,;"))}
    return sorted(urls)


def dossier(contract, asof, read=fetch_public):
    """Return one attributable source record or abstain; never substitute a later edition."""
    urls = explicit_urls(contract)
    if len(urls) != 1 or asof.tzinfo is None:
        return None
    try:
        packet = parse_bulletin(urls[0], read(urls[0], 250000))
        if datetime.fromisoformat(packet["available_asof_assumption"]) <= asof < datetime.fromisoformat(packet["target_utc"]):
            return packet
    except (OSError, URLError, ValueError):
        pass
    return None
