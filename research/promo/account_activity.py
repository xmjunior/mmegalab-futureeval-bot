"""Read organization billing activity without requesting inference or logging billing data."""
import argparse
import json
import os
from datetime import datetime, timezone
from decimal import Decimal
from urllib.error import HTTPError
from urllib.parse import urlencode

import runner

SNAPSHOT = datetime.fromisoformat(runner.OWNER_CHECKED)
COSTS = "https://api.openai.com/v1/organization/costs"


def costs_since_snapshot(key, now):
    # Whole days over-count activity before the snapshot. Never subtract credits/refunds.
    start = SNAPSHOT.replace(hour=0, minute=0, second=0, microsecond=0)
    if now <= SNAPSHOT:
        raise ValueError("clock_before_owner_snapshot")
    query = {"start_time": int(start.timestamp()), "end_time": int(now.timestamp()),
             "bucket_width": "1d", "limit": 31}
    total = Decimal(0)
    seen = set()
    buckets = 0
    while True:
        raw = runner.shadow.request(COSTS + "?" + urlencode(query),
                                    headers={"Authorization": "Bearer " + key}, timeout=20)
        page = json.loads(raw, parse_float=Decimal)
        if page.get("object") != "page" or not isinstance(page.get("data"), list):
            raise ValueError("unexpected_costs_response")
        for bucket in page["data"]:
            if bucket.get("object") != "bucket" or not isinstance(bucket.get("results"), list):
                raise ValueError("unexpected_costs_bucket")
            buckets += 1
            for item in bucket["results"]:
                amount = item["amount"]
                if str(amount["currency"]).upper() != "USD":
                    raise ValueError("non_usd_costs")
                value = Decimal(str(amount["value"]))
                if not value.is_finite():
                    raise ValueError("nonfinite_costs")
                total += max(0, value)
        if page.get("has_more") is False:
            break
        cursor = page.get("next_page")
        if not isinstance(cursor, str) or not cursor or cursor in seen or len(seen) >= 64:
            raise ValueError("invalid_costs_pagination")
        seen.add(cursor)
        query["page"] = cursor
    if not buckets:
        raise ValueError("no_usage_buckets")
    return total


def inspect(now):
    key = os.getenv("OPENAI_ADMIN_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        return {"organization_activity": "unavailable_no_configured_credential"}
    try:
        total = costs_since_snapshot(key, now)
    except HTTPError as exc:
        return {"organization_activity": f"http_{exc.code}"}
    except (OSError, ValueError, TypeError, KeyError, ArithmeticError) as exc:
        return {"organization_activity": runner.shadow.safe_status(exc)}
    return {"organization_activity": "positive_costs_reported" if total > 0 else "no_positive_costs_reported",
            "pending_external_usage_upper_bound": "unavailable_from_costs_endpoint"}


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps({**inspect(datetime.now(timezone.utc)), "inference_requests": 0}))
