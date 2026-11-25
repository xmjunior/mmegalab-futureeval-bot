"""Score new examples on the economic study's linear, two-open-tail geometry."""
import argparse
import json
import math

import numpy as np

from paired_economic import masses as archived_masses


def masses(series, quantiles):
    """Nearest admissible PMF in squared distance to the frozen study's PMF."""
    if series not in {"credit", "vix"}:
        raise ValueError("unknown series")
    q = np.asarray(quantiles, dtype=float)
    if q.shape != (5,) or not np.isfinite(q).all() or np.any(np.diff(q) < 0):
        raise ValueError("require five finite nondecreasing quantiles")
    edges, original = archived_masses(series, q)
    bins = len(edges) - 1
    # The two open exterior masses have their own minimum and no interior cap.
    lower = np.r_[0.001, np.full(bins, 0.01 / bins), 0.001]
    upper = np.r_[1.0, np.full(bins, min(1.0, 40.0 / bins)), 1.0]
    if np.all((lower <= original) & (original <= upper)):
        return edges, original
    # The Euclidean projection has the form clip(original + offset, lower, upper).
    low, high = -1., 1.
    for _ in range(64):
        offset = (low + high) / 2
        if np.clip(original + offset, lower, upper).sum() < 1:
            low = offset
        else:
            high = offset
    return edges, np.clip(original + (low + high) / 2, lower, upper)


def score(series, quantiles, outcome):
    if not math.isfinite(outcome):
        raise ValueError("nonfinite outcome")
    edges, pmf = masses(series, quantiles)
    bins = len(edges) - 1
    if outcome < edges[0]:
        bucket = 0
    elif outcome > edges[-1]:
        bucket = bins + 1
    else:
        location = (outcome - edges[0]) / (edges[-1] - edges[0])
        bucket = max(1, int(location * bins + 1 - 1e-10))
    return {"bucket": bucket, "log_mass_nats": math.log(pmf[bucket])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("series", choices=("credit", "vix"))
    parser.add_argument("outcome", type=float)
    parser.add_argument("quantiles", nargs=5, type=float)
    args = parser.parse_args()
    print(json.dumps(score(args.series, args.quantiles, args.outcome), allow_nan=False))
