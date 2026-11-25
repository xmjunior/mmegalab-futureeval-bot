"""Candidate quantile -> Metaculus-CDF constructions.

All candidates share the production standardisation (`sdk_standardize`) so that
comparisons isolate the *construction* step, which is where the audit found the
information loss.

TQ ("tail-consistent quantile") construction
--------------------------------------------
1. Map declared values to the question's own unscaled axis u (linear or log).
2. Map declared probabilities to Student-t(nu) scores z_i = T_nu^{-1}(p_i).
3. u(z) is the monotone piecewise-linear interpolant through (z_i, u_i), and the
   outermost segments are extended linearly (optionally widened by `tail_scale`).
   Inside the declared quantiles the density is a piecewise t-density; beyond them
   the tail decays like the reference t-tail instead of the SDK's flat, range-
   dependent ramp to a hard-coded 5%/95% at open bounds.
4. Closed bounds are *censored*: probability beyond a closed bound is assigned to
   the edge bucket that contains the bound (Metaculus bucket 1 / bucket N), which
   is exactly where an outcome "at the bound" is scored. This also fixes the SDK's
   displacement of explicit point masses at a closed bound.
5. Optional defensive uniform mixture `eps` inside the range.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

try:
    from .metaculus_math import Geometry, sdk_standardize
except ImportError:
    from metaculus_math import Geometry, sdk_standardize

_TIE = 1e-9


def _scores(p, nu):
    p = np.asarray(p, dtype=float)
    return stats.norm.ppf(p) if nu is None else stats.t.ppf(p, nu)


def _cdf_of_scores(z, nu):
    return stats.norm.cdf(z) if nu is None else stats.t.cdf(z, nu)


def _strict(u):
    u = np.array(u, dtype=float)
    for i in range(1, len(u)):
        if u[i] <= u[i - 1]:
            u[i] = u[i - 1] + _TIE
    return u


def tq_raw_cdf(percentiles, geo: Geometry, nu: float | None = 3.0, tail_scale: float = 1.0,
               eps: float = 0.0) -> np.ndarray:
    ps = np.array([p for p, _ in percentiles], dtype=float)
    xs = np.array([v for _, v in percentiles], dtype=float)
    if (
        len(ps) < 2
        or not (np.all(np.isfinite(ps)) and np.all(np.isfinite(xs)))
        or np.any(ps <= 0.0)
        or np.any(ps >= 1.0)
        or np.any(np.diff(ps) <= 0)
        or np.any(np.diff(xs) < 0)
    ):
        raise ValueError("percentiles must be finite, strictly inside (0,1), increasing in level and non-decreasing in value")
    if geo.zero_point is not None and np.any(xs <= geo.zero_point):
        raise ValueError("value at or below zero point on a log-scaled question")
    u = _strict(geo.to_loc(xs))
    z = _scores(ps, nu)
    s_lo = (u[1] - u[0]) / (z[1] - z[0]) * tail_scale
    s_hi = (u[-1] - u[-2]) / (z[-1] - z[-2]) * tail_scale
    grid = np.linspace(0.0, 1.0, geo.cdf_size)
    zz = np.interp(grid, u, z)
    lo, hi = grid < u[0], grid > u[-1]
    zz[lo] = z[0] + (grid[lo] - u[0]) / s_lo
    zz[hi] = z[-1] + (grid[hi] - u[-1]) / s_hi
    F = _cdf_of_scores(zz, nu)
    # Censor at closed bounds: mass beyond them is scored in the edge buckets.
    if not geo.open_lower:
        F[0] = 0.0
    if not geo.open_upper:
        F[-1] = 1.0
    if eps:
        F[1:-1] = (1 - eps) * F[1:-1] + eps * (F[0] + (F[-1] - F[0]) * grid[1:-1])
    return np.maximum.accumulate(F)


def tq_cdf(percentiles, geo: Geometry, **kw) -> np.ndarray:
    return sdk_standardize(tq_raw_cdf(percentiles, geo, **kw), geo)


def normal_fit_cdf(percentiles, geo: Geometry) -> np.ndarray:
    """Simple published reference: least-squares normal on the u-axis quantiles."""
    ps = np.array([p for p, _ in percentiles])
    u = geo.to_loc(np.array([v for _, v in percentiles], dtype=float))
    z = stats.norm.ppf(ps)
    A = np.vstack([np.ones_like(z), z]).T
    (mu, sd), *_ = np.linalg.lstsq(A, u, rcond=None)
    sd = max(sd, 1e-4)
    grid = np.linspace(0, 1, geo.cdf_size)
    F = stats.norm.cdf((grid - mu) / sd)
    if not geo.open_lower:
        F[0] = 0.0
    if not geo.open_upper:
        F[-1] = 1.0
    return sdk_standardize(np.maximum.accumulate(F), geo)


def metalog_cdf(percentiles, geo: Geometry, terms: int = 4) -> np.ndarray:
    """Keelin (2016) unbounded metalog fitted by least squares on the u-axis.

    Published, reproducible quantile-parameterised alternative. Infeasible
    (non-monotone) fits raise ValueError and are counted as failures.
    """
    ps = np.array([p for p, _ in percentiles])
    u = geo.to_loc(np.array([v for _, v in percentiles], dtype=float))

    def basis(y):
        L = np.log(y / (1 - y))
        cols = [np.ones_like(y), L, (y - 0.5) * L, (y - 0.5)][:terms]
        return np.vstack(cols).T

    a, *_ = np.linalg.lstsq(basis(ps), u, rcond=None)
    yy = np.concatenate([np.logspace(-9, np.log10(0.5), 4000), 1 - np.logspace(np.log10(0.5), -9, 4000)[1:]])
    qq = basis(yy) @ a
    if np.any(np.diff(qq) <= 0):
        raise ValueError("infeasible metalog")
    grid = np.linspace(0, 1, geo.cdf_size)
    F = np.interp(grid, qq, yy, left=0.0, right=1.0)
    if not geo.open_lower:
        F[0] = 0.0
    if not geo.open_upper:
        F[-1] = 1.0
    return sdk_standardize(np.maximum.accumulate(F), geo)
