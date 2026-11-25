"""Robust Student-t + Filtered Historical Simulation + Gibbs-Candes Online ACI.

Addresses Defect 3 across Market Pulse (`market_prices.py`) and OMEGA Continuous
(`research/omega/continuous/evaluate.py`):
  - D3a: Volatility-denominator window mismatch (`505`-day empirical sample scaled by
    `252`-day `long_vol` in `market_prices.py`; `30`-day window translated verbatim as
    `30` months = 2.5 years on monthly data in `omega/continuous/evaluate.py`).
  - D3b: Unstandardized raw-difference pooling across calm and crisis regimes, plus
    zero-drift bias on serially correlated macro changes.
  - D3c: Non-adaptive static Great-Moderation split-conformal expansion (`train` only,
    clamped >= 0) failing 80% coverage (`62.96%` champion, `68.52%` CQR) on `UNRATE`.

All parameters below are frozen strictly prior to out-of-sample `test` evaluation:
  - Damped AR(1) momentum weight `mom_weight = 0.50`, decay `phi = 0.65`,
  - Huber outlier bound `huber_k = 3.0` MADs on monthly innovations (prevents linear
    extrapolation of one-off exogenous jump shocks while preserving business-cycle drift),
  - Truncated symmetric Student-t (`df = 3.5`) on `[0, 100]` with blended robust MAD
    long/short (9-month vs full-window) scale,
  - Strictly causal Gibbs & Candes (2021) Online Adaptive Conformal Inference (`lr = 0.06`,
    target miscoverage `alpha = 0.20`) updating ONLY from past forecasts whose first
    official ALFRED release date satisfies `published_date <= current_origin`.
"""
from __future__ import annotations

import math
import numpy as np
from scipy import stats


class TruncatedStudentT:
    """Truncated location-scale Student-t distribution on [low_bound, high_bound]."""

    def __init__(self, loc: float, scale: float, df: float = 3.5,
                 low_bound: float = 0.0, high_bound: float = 100.0):
        if not (math.isfinite(loc) and math.isfinite(scale) and math.isfinite(df)) or df <= 1.0:
            raise ValueError("invalid Student-t parameters")
        self.loc = float(loc)
        self.scale = max(float(scale), 0.05)
        self.df = float(df)
        self.low_bound = float(low_bound)
        self.high_bound = float(high_bound)
        self.low = float(stats.t.cdf((self.low_bound - self.loc) / self.scale, self.df))
        self.mass = float(stats.t.cdf((self.high_bound - self.loc) / self.scale, self.df) - self.low)
        if not self.mass > 1e-12:
            raise ValueError("lost physical support")

    def cdf(self, x):
        z = (np.asarray(x, dtype=float) - self.loc) / self.scale
        return np.clip((stats.t.cdf(z, self.df) - self.low) / self.mass, 0.0, 1.0)

    def log_density(self, x: float) -> float:
        if not self.low_bound <= x <= self.high_bound:
            return -math.inf
        z = (x - self.loc) / self.scale
        return float(stats.t.logpdf(z, self.df) - math.log(self.scale * self.mass))

    def quantile(self, p):
        rp = np.clip(self.low + np.asarray(p, dtype=float) * self.mass, 1e-12, 1.0 - 1e-12)
        return np.clip(self.loc + self.scale * stats.t.ppf(rp, self.df), self.low_bound, self.high_bound)


def _mad_sd(x: np.ndarray, floor: float = 0.05) -> float:
    x = np.asarray(x, dtype=float)
    med = float(np.median(x))
    return max(1.4826 * float(np.median(np.abs(x - med))), floor)


def huber_damped_location_and_scale(hist: np.ndarray, h: int, mom_weight: float = 0.50,
                                    phi: float = 0.65, huber_k: float = 3.0,
                                    short_window: int = 9) -> tuple[float, float]:
    """Compute Huber-bounded damped momentum center and robust forecast residual scale."""
    hist = np.asarray(hist, dtype=float)
    if len(hist) < max(24, h + 6) or not np.all(np.isfinite(hist)):
        raise ValueError("insufficient or non-finite history")
    diff = np.diff(hist)
    mad_d = _mad_sd(diff, floor=0.05)
    clipped_diff = np.clip(diff, -huber_k * mad_d, huber_k * mad_d)
    damped_sum = sum(phi ** k for k in range(1, h + 1))

    recent_slope = float(np.mean(clipped_diff[-3:]))
    loc = float(hist[-1]) + mom_weight * recent_slope * damped_sum

    # Historical h-step pseudo-forecasts inside the known causal window hist:
    slopes_hist = np.array([np.mean(clipped_diff[i - 3:i]) for i in range(3, len(hist) - h)])
    preds_hist = hist[3:-h] + mom_weight * slopes_hist * damped_sum
    res_h = hist[3 + h:] - preds_hist
    mad_full = _mad_sd(res_h, floor=0.05 * math.sqrt(h))
    mad_short = _mad_sd(clipped_diff[-short_window:], floor=0.05) * math.sqrt(h)
    base_sd = 0.65 * mad_full + 0.35 * mad_short
    return loc, base_sd


def build_robust_t_distribution(hist: np.ndarray, h: int, aci_log_mult: float = 0.0,
                                mom_weight: float = 0.50, df: float = 3.5) -> TruncatedStudentT:
    loc, base_sd = huber_damped_location_and_scale(hist, h, mom_weight=mom_weight)
    t90_ratio = 1.2815515655446004 / float(stats.t.ppf(0.90, df))
    scale = base_sd * t90_ratio * math.exp(aci_log_mult)
    return TruncatedStudentT(loc, scale, df=df)


def run_causal_aci_sequence(cases: list, mom_weight: float = 0.50, df: float = 3.5,
                            aci_lr: float = 0.06, target_miscover: float = 0.20) -> list[TruncatedStudentT]:
    """Run strictly-causal Gibbs-Candes Online ACI across chronological cases.

    Each element of `cases` starts with `(origin, last, h, target, published_date, actual_y, hist, ...)`.
    An outcome `actual_y` enters the ACI state ONLY when `published_date <= current_origin`.
    """
    pending: list[list] = []  # [published_date, h, err_outside_80, consumed_bool]
    log_mult: dict[int, float] = {}
    out: list[TruncatedStudentT] = []
    p_check = np.array([0.10, 0.90])

    for c in cases:
        origin, _, h, _, published, y, hist = c[:7]
        # Release strictly already-published past errors in chronological publication order
        ready = [item for item in pending if item[0] <= origin and not item[3]]
        for item in sorted(ready, key=lambda r: (r[0], r[1])):
            item[3] = True
            hh = item[1]
            cur = log_mult.get(hh, 0.0)
            log_mult[hh] = float(np.clip(cur + aci_lr * (item[2] - target_miscover), -0.35, 0.60))

        dist = build_robust_t_distribution(hist, h, aci_log_mult=log_mult.get(h, 0.0),
                                           mom_weight=mom_weight, df=df)
        q10, q90 = dist.quantile(p_check)
        outside = float(not (q10 <= y <= q90))
        pending.append([published, h, outside, False])
        out.append(dist)
    return out


def fhs_standardized_quantiles(values: np.ndarray, horizon: int, levels=(0.10, 0.25, 0.50, 0.75, 0.90),
                               short_win: int = 30, ewma_lambda: float = 0.90) -> np.ndarray:
    """Filtered Historical Simulation (Hull & White 1998) fixing Defect D3a/D3b.

    Standardizes 1-step innovations `r_t / sigma_t` using a causal EWMA scale and forms
    filtered h-step standardized paths rescaled by current local volatility.
    """
    v = np.asarray(values, dtype=float)
    diff = np.diff(v)
    if len(diff) < short_win + horizon + 20:
        raise ValueError("insufficient series length for FHS")
    var_ewma = float(np.var(diff[:short_win], ddof=1))
    z_1 = []
    for r in diff[short_win:]:
        # Causal two-speed variance update (adapts promptly at structural regime breaks)
        var_fast = 0.75 * var_ewma + 0.25 * float(r) ** 2
        sd = math.sqrt(max(var_ewma, 1e-16))
        if abs(float(r)) <= 3.5 * sd:
            z_1.append(float(r) / sd)
        var_ewma = ewma_lambda * var_ewma + (1.0 - ewma_lambda) * float(r) ** 2
        if abs(float(r)) > 3.5 * sd:
            var_ewma = var_fast
    z_arr = np.asarray(z_1, dtype=float)
    z_arr = z_arr - np.median(z_arr)
    csum = np.concatenate([[0.0], np.cumsum(z_arr)])
    h_shocks = csum[horizon:] - csum[:-horizon]
    h_sd = float(np.std(h_shocks, ddof=1))
    if h_sd > 1e-12:
        h_shocks = h_shocks * (math.sqrt(horizon) / h_sd)
    cur_sd = 0.5 * math.sqrt(max(var_ewma, 1e-16)) + 0.5 * float(np.std(diff[-short_win:], ddof=1))
    return float(v[-1]) + cur_sd * np.quantile(h_shocks, levels)


def holt_damped_t_distribution(hist: np.ndarray, h: int, alpha: float = 0.70,
                               beta: float = 0.20, phi: float = 0.70, df: float = 4.0) -> TruncatedStudentT:
    """Published reproducible macro baseline: Gardner-McKenzie (1985) damped Holt trend + Student-t."""
    hist = np.asarray(hist, dtype=float)
    level = float(hist[0])
    trend = float(hist[1] - hist[0])
    errors_1 = []
    for y in hist[1:]:
        pred = level + phi * trend
        errors_1.append(y - pred)
        prev_level = level
        level = alpha * float(y) + (1.0 - alpha) * (level + phi * trend)
        trend = beta * (level - prev_level) + (1.0 - beta) * phi * trend
    damped_h = sum(phi ** k for k in range(1, h + 1))
    loc = level + damped_h * trend
    sd = _mad_sd(np.asarray(errors_1[-36:]), floor=0.05) * math.sqrt(h)
    scale = sd * (1.2815515655446004 / float(stats.t.ppf(0.90, df)))
    return TruncatedStudentT(loc, scale, df=df)
