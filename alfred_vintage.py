"""Read a complete FRED/ALFRED API vintage without redirecting a credential."""
import http.client
import json
import os
from datetime import date
from decimal import Decimal
from urllib.parse import urlencode


def fred_vintage(series, asof, start):
    params = urlencode(dict(series_id=series, realtime_start=asof, realtime_end=asof,
                            observation_start=start, observation_end=asof, file_type="json",
                            api_key=os.environ["FRED_API_KEY"]))
    conn = http.client.HTTPSConnection("api.stlouisfed.org", timeout=25)
    try:
        conn.request("GET", "/fred/series/observations?" + params,
                     headers={"Accept":"application/json", "User-Agent":"MEGALAB-vintage"})
        response = conn.getresponse()
        raw = response.read(2_000_001)
        if response.status != 200:
            raise RuntimeError("http_" + str(response.status))
        if len(raw) > 2_000_000:
            raise ValueError("response_limit")
        data = json.loads(raw)
    finally:
        conn.close()
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
