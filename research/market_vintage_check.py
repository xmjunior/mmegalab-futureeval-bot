"""Replay only six closed, frozen Market Pulse inputs using ALFRED's API."""
import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from unittest.mock import patch

from source_trials import fred_vintage

TOLERANCE = 1e-8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("market_path", type=Path)
    parser.add_argument("--fallback-smoke", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.market_path.resolve()))
    import market_prices as mp

    reference = dict(mp._FRED_REPLAY)
    if len(reference) != 6:
        raise ValueError("unexpected_frozen_cohort")
    results = []
    for (pid, a, s, end), row in sorted(reference.items()):
        source = parse_qs(urlsplit(row[0]).query)
        series = source["id"][0]
        vintage = source["vintage_date"][0]
        start = source["cosd"][0]
        stamp = datetime.fromisoformat(a)
        cutoff = stamp.astimezone(mp.NY).date()
        if date.fromisoformat(vintage) >= cutoff or datetime.fromisoformat(end).date() >= date(2026,11,24):
            raise ValueError("invalid_vintage_or_unresolved_case")
        if args.fallback_smoke:
            with patch.object(mp, "_FRED_REPLAY", {}), patch.object(mp, "_get", side_effect=OSError("chart unavailable")):
                restored = mp.model(pid,stamp,datetime.fromisoformat(s),datetime.fromisoformat(end))
            if "vintage_date=" + vintage not in restored.sources[0] or "api_key" in restored.sources[0]:
                raise ValueError("unexpected_fallback_source")
        else:
            try:
                data = fred_vintage(series, vintage, start)
            except (RuntimeError, ValueError, OSError, KeyError) as err:
                results.append({"series": series, "vintage": vintage, "matched":False,
                                "unavailable":str(err) if isinstance(err, RuntimeError)
                                and str(err).startswith("http_") else type(err).__name__})
                continue
            def feed(code, day):
                if code != series or day.isoformat() != vintage:
                    raise ValueError("model_requested_a_different_vintage")
                return {date.fromisoformat(k):float(v) for k,v in data.items()}, "FRED_API"
            with patch.object(mp, "_FRED_REPLAY", {}), patch.object(mp, "_fred", feed):
                restored = mp.model(pid,stamp,datetime.fromisoformat(s),datetime.fromisoformat(end))
        largest = max(abs(x-y) for x,y in zip(restored.base_quantiles,row[5], strict=True))
        baseline_gap = max(abs(x-y) for x,y in zip(restored.naive_quantiles,row[6], strict=True))
        match = (restored.source_date.isoformat() == row[2] and restored.sample_size == row[3]
                 and abs(restored.center-row[4]) <= TOLERANCE and max(largest,baseline_gap) <= TOLERANCE)
        results.append({"series":series,"vintage":vintage,"matched":bool(match),
                        "sealed_source_date":row[2],"api_source_date":restored.source_date.isoformat(),
                        "max_abs_quantile_difference":round(max(largest,baseline_gap),8)})
        if args.fallback_smoke:
            if not match:
                raise ValueError("fallback_changed_distribution")
            break
    print(json.dumps({"cases":len(results),"matches":sum(r["matched"] for r in results),
                      "results":results,"forecasts_published":0},sort_keys=True))


if __name__ == "__main__":
    main()
