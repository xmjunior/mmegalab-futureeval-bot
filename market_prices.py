"""Point-in-time Market Pulse 26Q4 market data and numeric spot-score benchmark."""

import csv
import io
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np

SPECS = {
    46007: ("spread", "CL=F", "ES=F"),
    46008: ("spread", "GC=F", "ES=F"),
    46010: ("level", "BAMLH0A0HYM2"),
    46011: ("spread", "NQ=F", "ES=F"),
    46012: ("spread", "NVDA", "AAPL"),
    46013: ("spread", "NVDA", "MSFT"),
    46014: ("level", "DGS10"),
    46015: ("maximum", "VIX"),
}
WIDTHS = (1.0, 1.25, 1.5)
NY = ZoneInfo("America/New_York")
CBOE = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9)


def period_end(label: str, spot: datetime) -> datetime:
    match = re.fullmatch(
        r"([A-Za-z]{3})\s+(\d{1,2})\s*[-–—]\s*(?:([A-Za-z]{3})\s+)?(\d{1,2})(?:\s*\([^)]*\))?",
        label.strip(),
    )
    if not match or spot.tzinfo is None:
        raise ValueError(f"Unrecognized period: {label!r}")
    left, start, right, end = match.groups()
    start_month = datetime.strptime(left, "%b").month
    end_month = datetime.strptime(right or left, "%b").month
    start_year = spot.year + (1 if spot.month == 12 and start_month == 1 else 0)
    start_day = date(start_year, start_month, int(start))
    end_day = date(start_year + int(end_month < start_month), end_month, int(end))
    if not 5 <= (end_day - start_day).days <= 20 or abs((start_day - spot.date()).days) > 7:
        raise ValueError(f"Unexpected dates for {label!r}, spot={spot.isoformat()}")
    return datetime.combine(end_day, time(16), NY).astimezone(timezone.utc)


# Frozen outcome labels, extracted from the corresponding public Metaculus group post
# (https://www.metaculus.com/questions/<post_id>/) on 2026-11-24. This is the entire
# resolved numeric cohort for the eight price series on that date, including losses.
# Results are only read by the scorer after the forecast has been built.
_CLOSED = [
    (46007, 46200, 'Oct 12 - Oct 23', '2026-10-12T03:00:00Z', -25, 25, 200, 14.98),
    (46007, 46201, 'Oct 26 - Nov 6', '2026-10-26T03:00:00Z', -25, 25, 200, -8.68),
    (46007, 46202, 'Nov 9 - Nov 20', '2026-11-09T04:00:00Z', -25, 25, 200, -1.23),
    (46008, 46206, 'Oct 12 - Oct 23', '2026-10-12T03:00:00Z', -10, 10, 200, -2.398),
    (46008, 46207, 'Oct 26 - Nov 6', '2026-10-26T03:00:00Z', -10, 10, 200, 5.037),
    (46008, 46208, 'Nov 9 - Nov 20', '2026-11-09T04:00:00Z', -10, 10, 200, -7.421),
    (46010, 46212, 'Oct 12 - Oct 23', '2026-10-12T03:00:00Z', 1.745, 3.745, 200, 3.38),
    (46010, 46213, 'Oct 26 - Nov 6', '2026-10-26T03:00:00Z', 1.745, 3.745, 200, 3.42),
    (46010, 46709, 'Nov 9-20 (wider range)', '2026-11-06T13:00:00Z', 1.745, 4.255, 251, 3.76),
    (46011, 46218, 'Oct 12 - Oct 23', '2026-10-12T03:00:00Z', -5, 5, 200, -3.856),
    (46011, 46219, 'Oct 26 - Nov 6', '2026-10-26T03:00:00Z', -5, 5, 200, -3.615),
    (46011, 46220, 'Nov 9 - Nov 20', '2026-11-09T04:00:00Z', -5, 5, 200, -0.923),
    (46012, 46224, 'Oct 12 - Oct 23', '2026-10-12T03:00:00Z', -15, 15, 200, -11.28),
    (46012, 46225, 'Oct 26 - Nov 6', '2026-10-26T03:00:00Z', -15, 15, 200, -9.99),
    (46013, 46230, 'Oct 12 - Oct 23', '2026-10-12T03:00:00Z', -15, 15, 200, -6.02),
    (46013, 46231, 'Oct 26 - Nov 6', '2026-10-26T03:00:00Z', -15, 15, 200, -12.69),
    (46014, 46236, 'Oct 12 - Oct 23', '2026-10-12T03:00:00Z', 4.255, 6.255, 200, 5.0),
    (46014, 46237, 'Oct 26 - Nov 6', '2026-10-26T03:00:00Z', 4.255, 6.255, 200, 4.86),
    (46014, 46238, 'Nov 9 - Nov 20', '2026-11-09T04:00:00Z', 4.255, 6.255, 200, 4.92),
    (46015, 46242, 'Oct 12 - Oct 23', '2026-10-12T03:00:00Z', 10, 50, 200, 22.31),
    (46015, 46243, 'Oct 26 - Nov 6', '2026-10-26T03:00:00Z', 10, 50, 200, 28.36),
    (46015, 46244, 'Nov 9 - Nov 20', '2026-11-09T04:00:00Z', 10, 50, 200, 40.03),
]
HISTORY = []
for _post, _id, _label, _spot, _low, _high, _bins, _result in _CLOSED:
    _utc = datetime.fromisoformat(_spot)
    HISTORY.append(dict(post_id=_post, question_id=_id, label=_label, spot=_utc,
                        resolve=period_end(_label, _utc), low=_low, high=_high,
                        outcome=_result, bins=_bins, source=f"https://www.metaculus.com/questions/{_post}/"))


# Only exact, already-resolved windows can match these derived ALFRED / Cboe inputs.
# Level-series records store the unscaled empirical percentiles and both volatility
# estimates. VIX records retain the historical unconditional and conditional quantiles.
@lru_cache(maxsize=1)
def _closed_replays():
    rows = json.loads(Path(__file__).with_name("market_pulse_vintages.json").read_text())
    records = {(r["post_id"], r["asof"], r["spot"], r["resolve"]): r for r in rows}
    if len(records) != len(rows):
        raise ValueError("Duplicate archived market window")
    return records


def _get(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; MarketPulse26Q4/1.0)"})
    with urlopen(request, timeout=18) as response:
        data = response.read(8_000_001)
    if len(data) > 8_000_000:
        raise ValueError("Market source exceeds expected size")
    return data


@lru_cache(maxsize=64)
def _yahoo(symbol: str):
    path = "/v8/finance/chart/" + quote(symbol, safe="") + "?range=5y&interval=1d"
    last_error = None
    for host in ("query2.finance.yahoo.com", "query1.finance.yahoo.com"):
        try:
            result = json.loads(_get("https://" + host + path))["chart"]["result"][0]
            if result["meta"]["symbol"] != symbol:
                raise ValueError("Instrument identity mismatch")
            quotes = result["indicators"]
            closes = quotes["quote"][0]["close"] if symbol.endswith("=F") else quotes["adjclose"][0]["adjclose"]
            tz = ZoneInfo(result["meta"]["exchangeTimezoneName"])
            series = {datetime.fromtimestamp(t, tz).date(): float(v)
                      for t, v in zip(result["timestamp"], closes, strict=True)
                      if v is not None and math.isfinite(v) and v > 0}
            return series, "https://finance.yahoo.com/quote/" + quote(symbol, safe="") + "/history/"
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            last_error = exc
    raise RuntimeError(f"Historical quote data unavailable for {symbol}") from last_error


@lru_cache(maxsize=64)
def _fred(code: str, vintage: date):
    qs = urlencode({"id": code, "cosd": (vintage - timedelta(days=810)).isoformat(),
                    "vintage_date": vintage.isoformat()})
    url = "https://alfred.stlouisfed.org/graph/alfredgraph.csv?" + qs
    rows = csv.DictReader(io.StringIO(_get(url).decode()))
    field = code + "_" + vintage.strftime("%Y%m%d")
    if not rows.fieldnames or field not in rows.fieldnames:
        raise ValueError(f"ALFRED did not deliver the requested {code} vintage")
    return {date.fromisoformat(row["observation_date"]): float(row[field])
            for row in rows if row.get(field) not in (None, "", ".")}, url


@lru_cache(maxsize=8)
def _current_fred(code: str, cutoff: date):
    # Day-before-only observations, plus one extra calendar day for release lag.
    # A current-vintage release must never substitute for a historical vintage.
    eligible_day = cutoff - timedelta(days=1)
    query = urlencode({"id": code, "cosd": (cutoff-timedelta(days=810)).isoformat(),
                       "coed": eligible_day.isoformat()})
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?" + query
    rows = csv.DictReader(io.StringIO(_get(url).decode()))
    if not rows.fieldnames or code not in rows.fieldnames:
        raise ValueError(f"FRED returned the wrong series for {code}")
    return {date.fromisoformat(row["observation_date"]): float(row[code])
            for row in rows if row.get(code) not in (None, "", ".")
            and date.fromisoformat(row["observation_date"]) <= eligible_day}, url


@lru_cache(maxsize=1)
def _vix():
    rows = csv.DictReader(io.StringIO(_get(CBOE).decode("utf-8-sig")))
    return {datetime.strptime(r["DATE"], "%m/%d/%Y").date(): (float(r["CLOSE"]), float(r["HIGH"]))
            for r in rows if r.get("CLOSE") and r.get("HIGH")}, CBOE


@dataclass
class MarketEstimate:
    source_date: date
    sources: tuple[str, ...]
    sample_size: int
    center: float
    base_quantiles: np.ndarray
    naive_quantiles: np.ndarray

    def quantiles(self, width: float = 1.0) -> dict[float, float]:
        if width not in WIDTHS:
            raise ValueError("Unvalidated distribution width")
        values = self.center + width * (self.base_quantiles - self.center)
        return dict(zip(LEVELS, map(float, values), strict=True))


def _regime_estimate(source_date, sources, sample_size, center, naive, short_vol, sample_vol):
    if not np.isfinite(sample_vol) or sample_vol <= 0 or not np.isfinite(short_vol) or short_vol < 0:
        raise ValueError("Missing return variance")
    multiplier = float(np.clip(np.sqrt(short_vol / sample_vol), 0.7, 2.0))
    adjusted = center + multiplier * (naive - center)
    return MarketEstimate(source_date, sources, sample_size, center, adjusted, naive)


def _weekdays(first: date, last: date) -> int:
    return sum((first + timedelta(days=n)).weekday() < 5 for n in range((last-first).days + 1))


def model(post_id: int, as_of: datetime, spot: datetime, resolve: datetime) -> MarketEstimate:
    if any(dt.tzinfo is None for dt in (as_of, spot, resolve)) or not as_of <= spot < resolve:
        raise ValueError("Invalid as-of / spot / target order")
    if post_id not in SPECS:
        raise ValueError(f"No verified market feed for {post_id}")
    kind, *symbols = SPECS[post_id]
    # Excluding today's entire bar prevents using a session's eventual closing price
    # at an intraday cutoff. At a Sunday-night/Monday-UTC spot, Friday is available.
    cutoff = as_of.astimezone(NY).date() - timedelta(days=1)
    now = datetime.now(timezone.utc)
    key = (post_id, *(dt.astimezone(timezone.utc).isoformat() for dt in (as_of, spot, resolve)))
    replay = _closed_replays().get(key) if resolve < now else None
    if replay:
        source_day = date.fromisoformat(replay["source_date"])
        naive = np.asarray(replay["naive_quantiles"])
        if kind == "maximum":
            sources = (replay["url"], "prefix-sha256:" + replay["prefix_sha256"])
            return MarketEstimate(source_day, sources, replay["sample_size"], replay["center"],
                                  np.asarray(replay["conditional_quantiles"]), naive)
        sources = (replay["url"], "sha256:" + replay["sha256"])
        return _regime_estimate(source_day, sources, replay["sample_size"], replay["center"],
                                naive, replay["short_vol"], replay["sample_vol"])
    if kind == "level":
        # The current public FRED chart is suitable only for an actual present-time
        # prediction. An arbitrary past as-of always needs its original ALFRED vintage.
        if timedelta(0) <= now-as_of <= timedelta(minutes=30) and now < spot:
            try:
                quotes, source = _current_fred(symbols[0], cutoff)
                if not quotes or _weekdays(max(quotes)+timedelta(days=1), cutoff) > 2:
                    raise ValueError("Current FRED observations stale")
            except (OSError, ValueError, KeyError):
                quotes, source = _fred(symbols[0], cutoff)
        else:
            quotes, source = _fred(symbols[0], cutoff)
        dates = sorted(d for d in quotes if d <= cutoff)[-505:]
        values = np.array([quotes[d] for d in dates])
        sources = (source,)
    elif kind == "maximum":
        quotes, source = _vix()
        dates = sorted(d for d in quotes if d <= cutoff)[-505:]
        values = np.array([quotes[d][0] for d in dates])
        highs = np.array([quotes[d][1] for d in dates])
        sources = (source,)
    else:
        a, source_a = _yahoo(symbols[0])
        b, source_b = _yahoo(symbols[1])
        dates = sorted(d for d in a.keys() & b.keys() if d <= cutoff)[-505:]
        values = np.array([a[d] for d in dates])
        other = np.array([b[d] for d in dates])
        sources = (source_a, source_b)
    if len(dates) < 160 or _weekdays(dates[-1] + timedelta(days=1), cutoff) > 2:
        raise ValueError(f"Insufficient or stale prices for {post_id} as of {as_of.isoformat()}")
    target_day = resolve.astimezone(NY).date()
    # The reporting periods start on the UTC spot date for regular groups. For a
    # replaced FRED question, the level horizon is measured from the last vintage.
    start = spot.astimezone(timezone.utc).date()
    horizon = _weekdays(start, target_day)
    lag = _weekdays(dates[-1] + timedelta(days=1), start - timedelta(days=1))
    if kind == "level":
        horizon += lag
    if not 4 <= horizon <= 25 or lag > 15:
        raise ValueError("Outside the supported biweekly horizon")
    left = np.arange(len(values) - horizon - (lag if kind == "maximum" else 0))
    right = left + horizon
    if kind == "spread":
        historical = 100 * (values[right]/values[left] - other[right]/other[left])
        daily = 100 * (np.diff(np.log(values)) - np.diff(np.log(other)))
        center = 0.0
    elif kind == "level":
        historical = values[right] - values[left]
        daily = np.diff(values)
        center = float(values[-1])
    else:
        offsets = np.arange(1 + lag, horizon + lag + 1)
        historical = np.max(highs[left[:, None] + offsets], axis=1) / values[left]
        # Both candidates start at the latest VIX. Compare conditioning on similar
        # starting levels against using peak multiples from the full rolling sample.
        naive = historical * values[-1]
        nearest = np.argsort(np.abs(np.log(values[left] / values[-1])))[:max(80, len(left)//3)]
        outcomes = historical[nearest] * values[-1]
        return MarketEstimate(dates[-1], sources, len(nearest), float(np.median(outcomes)),
                              np.quantile(outcomes, LEVELS), np.quantile(naive, LEVELS))
    # The empirical horizon changes and their volatility denominator use the same
    # available observations (up to 505 closes, hence up to 504 daily changes).
    changes = historical - np.median(historical)
    naive = np.quantile(center + changes, LEVELS)
    return _regime_estimate(dates[-1], sources, len(left), center, naive,
                            np.std(daily[-30:], ddof=1), np.std(daily, ddof=1))


def numeric_cdf(question, quantiles):
    from continuous_math import student_t_distribution

    distribution = student_t_distribution(question, sorted(quantiles.items()))
    return [point.percentile for point in distribution.get_cdf()]


def _mass(quantiles: dict[float, float], low: float, high: float, outcome: float, bins: int,
          zero_point: float | None = None) -> float:
    from continuous_math import continuous_bucket

    # Market Pulse archives include both linear and logarithmic axes.
    question = SimpleNamespace(lower_bound=low, upper_bound=high, zero_point=zero_point, cdf_size=bins+1,
                               open_lower_bound=True, open_upper_bound=True)
    cdf = numeric_cdf(question, quantiles)
    bucket = continuous_bucket(outcome, low, high, bins, zero_point)
    if bucket == 0:
        return cdf[0]
    if bucket == bins + 1:
        return 1 - cdf[-1]
    return cdf[bucket] - cdf[bucket-1]


def benchmark(history=None) -> dict:
    cohort = HISTORY if history is None else history
    cases = []
    def evaluate(row):
        estimate = model(row['post_id'], row['spot'] - timedelta(seconds=1), row['spot'], row['resolve'])
        args = row['low'], row['high'], row['outcome'], row['bins'], row.get('zero_point')
        naive = dict(zip(LEVELS, estimate.naive_quantiles))
        baseline = _mass(naive, *args)
        return dict(post_id=row['post_id'], question_id=row['question_id'], spot=row['spot'].isoformat(),
                    source_date=estimate.source_date.isoformat(), outcome=row['outcome'], sample_size=estimate.sample_size,
                    baseline_mass=baseline, widths={w: math.log(_mass(estimate.quantiles(w), *args)/baseline) for w in WIDTHS})
    failed = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [(row, pool.submit(evaluate, row)) for row in cohort]
        for row, future in futures:
            try:
                cases.append(future.result())
            except Exception as error:
                failed.setdefault(str(row['post_id']), []).append(
                    dict(question_id=row['question_id'], error=f"{type(error).__name__}: {error}")
                )
    by_group = {}
    allowed = {}
    for post_id in SPECS:
        subset = sorted((x for x in cases if x['post_id']==post_id), key=lambda x:x['spot'])
        if str(post_id) in failed or len(subset) < 3:
            allowed[str(post_id)] = []
            by_group[str(post_id)] = dict(n=len(subset), omitted=failed.get(str(post_id), []))
            continue
        development = subset[:-1]
        gains = {w: sum(x['widths'][w] for x in development)/len(development) for w in WIDTHS}
        # Freeze candidate widths from development only. The downstream named model
        # is blinded to dates, asset identities and labels, then scored on the last spot.
        allowed[str(post_id)] = [w for w in WIDTHS if gains[w] > 0]
        by_group[str(post_id)] = dict(n=len(subset), development_mean_log_gain=gains,
                                      held_out_question=subset[-1]['question_id'])
    return dict(cases=cases, by_group=by_group, eligible=[int(k) for k,v in allowed.items() if v],
                allowed_widths=allowed, failed=failed)


if __name__ == '__main__':
    print(json.dumps(benchmark(), indent=2))
