"""Check whether a 30-trading-session volatility window is observable in a series."""
import numpy as np


def session_volatility_ratio(levels, *, observation_unit):
    """Return the bounded short/full volatility multiplier, or abstain on monthly data.

    Callers must supply consecutive, pre-origin levels in the declared native unit.
    Thirty session differences require 31 levels; monthly levels cannot identify them.
    """
    if observation_unit == "month":
        return None
    if observation_unit != "trading_session":
        raise ValueError("unknown observation unit")
    levels = np.asarray(levels, dtype=float)
    if levels.ndim != 1 or not np.isfinite(levels).all():
        raise ValueError("require a finite one-dimensional series")
    if len(levels) < 31:
        return None
    changes = np.diff(levels)
    full = np.std(changes, ddof=1)
    if full == 0:
        return 1.0
    recent = np.std(changes[-30:], ddof=1)
    return float(np.clip(np.sqrt(recent / full), 0.7, 2.0))


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observation_unit", choices=("month", "trading_session"))
    parser.add_argument("levels", nargs="+", type=float)
    args = parser.parse_args()
    ratio = session_volatility_ratio(args.levels, observation_unit=args.observation_unit)
    print(json.dumps({"window_trading_sessions": 30, "observation_unit": args.observation_unit,
                      "status": "unsupported" if ratio is None else "available",
                      "volatility_multiplier": ratio}, allow_nan=False))
