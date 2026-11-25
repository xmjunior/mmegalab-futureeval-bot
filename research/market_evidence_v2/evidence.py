"""Factual metadata for a blind review of a cent-grid end-of-period credit spread."""
from datetime import timedelta
import math

import numpy as np

import market_prices as prices


def prior_coverage(values, horizon, width):
    """Disjoint historical targets, estimated solely from their respective prefixes.

    All observations belong to the vintage available to the current reviewer. An
    earlier observation or label may have been revised by this vintage.
    """
    values = np.asarray(values, dtype=float)
    if horizon < 4 or horizon > 25 or not np.isfinite(values).all():
        raise ValueError("Unsupported horizon or series")
    hits = trials = 0
    for target in range(len(values) - 1, 159 + horizon - 1, -horizon):
        origin = target - horizon
        prefix = values[max(0, origin - 504):origin + 1]
        daily = np.diff(prefix)
        long = np.std(daily[-252:], ddof=1)
        if long <= 0:
            raise ValueError("Undefined historical regime")
        ratio = np.clip(np.sqrt(np.std(daily[-30:], ddof=1) / long), 0.7, 2.0)
        changes = prefix[horizon:] - prefix[:-horizon]
        lo, hi = np.quantile(changes - np.median(changes), [.1, .9])
        lo, hi = prefix[-1] + width * ratio * np.array([lo, hi])
        hits += int(lo <= values[target] <= hi)
        trials += 1
    return {"interval_level": 0.8, "prior_disjoint_targets": trials, "inside_interval": hits}


def grounded_row(row, observations, as_of, spot, resolve, estimate):
    cutoff = as_of.astimezone(prices.NY).date() - timedelta(days=1)
    dates = sorted(d for d in observations if d <= cutoff)[-505:]
    if len(dates) < 160 or dates[-1] != estimate.source_date:
        raise ValueError("Inconsistent vintage evidence")
    if not math.isclose(observations[dates[-1]], estimate.center, abs_tol=1e-10):
        raise ValueError("Forecast and evidence have different last observations")
    horizon = prices._weekdays(spot.date(), resolve.astimezone(prices.NY).date())
    horizon += prices._weekdays(dates[-1] + timedelta(days=1), spot.date() - timedelta(days=1))
    if len(dates) - horizon != estimate.sample_size:
        raise ValueError("Mismatched horizon or sample count")
    values = [observations[d] for d in dates]
    metadata = {
        "measurement": "end-of-period level of a credit spread, in percentage points; cent grid",
        "data": "daily observations in one historical release available at forecast origin",
        "weekdays_since_latest_in_eligible_vintage": prices._weekdays(dates[-1] + timedelta(days=1), cutoff),
        "model_horizon_weekday_steps": horizon,
        "sampling": "the original observations count is overlapping historical horizon changes",
        "coverage_method": "same estimator on earlier prefixes of this vintage; prior targets do not overlap; past coverage can differ from future coverage",
        "prior_interval_checks": [
            {"width": option["width"], **prior_coverage(values, horizon, option["width"])} for option in row["choices"]
        ],
    }
    return {**row, "evidence": metadata}
