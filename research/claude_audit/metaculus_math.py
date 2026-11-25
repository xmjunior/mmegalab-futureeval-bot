"""Exact Metaculus continuous-question geometry and scoring, independent of the bot.

Sources (fetched 2026-11-24 with `gh api`, blob SHAs recorded in EVIDENCE.md):
  * Metaculus/metaculus utils/the_math/formulas.py   (location <-> bucket mapping)
  * Metaculus/metaculus scoring/score_math.py        (baseline / peer log score)
  * https://www.metaculus.com/api/  "CDF generation details" (validity rules)

Continuous baseline score  = 100 * ln(pmf[bucket] / baseline) / 2
   baseline = 0.05 outside the range, (1 - 0.05 * open_bounds) / N inside.
Continuous peer score      = 100 * n/(n-1) * ln(pmf[bucket] / gm[bucket]) / 2
Hence for a fixed field of opponents, Delta(peer) = 50 * Delta(ln pmf[bucket]) exactly,
so every improvement in expected ln pmf maps 1:1 onto both official scores.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Geometry:
    lower: float
    upper: float
    open_lower: bool
    open_upper: bool
    zero_point: float | None = None
    cdf_size: int = 201

    def __post_init__(self):
        if not (math.isfinite(self.lower) and math.isfinite(self.upper)) or self.upper <= self.lower:
            raise ValueError("invalid range")
        if self.zero_point is not None and self.lower <= self.zero_point:
            raise ValueError("lower bound must exceed zero point")

    @property
    def n(self) -> int:  # inbound outcome count
        return self.cdf_size - 1

    @property
    def open_count(self) -> int:
        return int(self.open_lower) + int(self.open_upper)

    def to_loc(self, x):
        """Nominal value -> unscaled location (0 at lower, 1 at upper). Vectorised."""
        x = np.asarray(x, dtype=float)
        L, U, z = self.lower, self.upper, self.zero_point
        if z is None:
            return (x - L) / (U - L)
        r = (U - z) / (L - z)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (np.log((x - L) * (r - 1) + (U - L)) - np.log(U - L)) / np.log(r)

    def from_loc(self, u):
        u = np.asarray(u, dtype=float)
        L, U, z = self.lower, self.upper, self.zero_point
        if z is None:
            return L + (U - L) * u
        r = (U - z) / (L - z)
        return L + (U - L) * (r ** u - 1) / (r - 1)

    def edges(self) -> np.ndarray:
        return self.from_loc(np.linspace(0.0, 1.0, self.cdf_size))

    def bucket(self, x: float) -> int:
        """Metaculus unscaled_location_to_bucket_index (formulas.py)."""
        if x < self.lower:
            return 0
        if x > self.upper:
            return self.n + 1
        u = float(self.to_loc(x))
        if u >= 1:
            return self.n
        return max(int(u * self.n + 1 - 1e-10), 1)

    def baseline(self, bucket: int) -> float:
        if bucket in (0, self.n + 1):
            return 0.05
        return (1 - 0.05 * self.open_count) / self.n


def pmf_from_cdf(cdf) -> np.ndarray:
    cdf = np.asarray(cdf, dtype=float)
    return np.diff(cdf, prepend=0.0, append=1.0)


def baseline_score(cdf, geo: Geometry, outcome: float) -> float:
    pmf = pmf_from_cdf(cdf)
    b = geo.bucket(outcome)
    return 100.0 * math.log(pmf[b] / geo.baseline(b)) / 2.0


def expected_baseline_score(cdf, geo: Geometry, true_bucket_probs) -> float:
    """Exact expectation of the official continuous baseline score."""
    pmf = pmf_from_cdf(cdf)
    p = np.asarray(true_bucket_probs, dtype=float)
    base = np.full(len(pmf), (1 - 0.05 * geo.open_count) / geo.n)
    base[0] = base[-1] = 0.05
    mask = p > 0
    return float(50.0 * np.sum(p[mask] * np.log(pmf[mask] / base[mask])))


def true_bucket_probs(true_cdf, geo: Geometry) -> np.ndarray:
    """Probability of each of the N+2 Metaculus buckets under a true nominal CDF.

    Bucket 1 is [lower, edge1]; buckets k>1 are (edge_{k-1}, edge_k]. On closed bounds
    outcomes strictly outside [lower, upper] are impossible on Metaculus and any mass at
    or censored onto the closed boundary belongs in the edge inbound bucket (1 or N).
    """
    F = np.asarray(true_cdf(geo.edges()), dtype=float)
    p = np.diff(F, prepend=0.0, append=1.0)
    p[0] = F[0]
    p[-1] = 1.0 - F[-1]
    if not geo.open_lower:
        p[1] += p[0]
        p[0] = 0.0
    if not geo.open_upper:
        p[-2] += p[-1]
        p[-1] = 0.0
    p = np.clip(p, 0.0, None)
    return p / p.sum()


def validate_cdf(cdf, geo: Geometry) -> list[str]:
    """Return the list of Metaculus API rule violations (empty list == accepted)."""
    errors = []
    c = np.asarray(cdf, dtype=float)
    if len(c) != geo.cdf_size:
        errors.append(f"size {len(c)} != {geo.cdf_size}")
        return errors
    if not np.all(np.isfinite(c)):
        errors.append("non-finite values")
        return errors
    steps = np.diff(c)
    min_step = 0.01 / geo.n
    max_step = 0.2 * 200 / geo.n
    if steps.min() < min_step - 1e-10:
        errors.append(f"min step {steps.min():.3g} < {min_step:.3g}")
    if steps.max() > max_step + 1e-10:
        errors.append(f"max step {steps.max():.3g} > {max_step:.3g}")
    if geo.open_lower:
        if c[0] < 0.001 - 1e-12:
            errors.append(f"open lower bound mass {c[0]:.4g} < 0.001")
    elif c[0] != 0.0:
        errors.append(f"closed lower bound cdf[0]={c[0]!r} != 0")
    if geo.open_upper:
        if c[-1] > 0.999 + 1e-12:
            errors.append(f"open upper bound mass {1 - c[-1]:.4g} < 0.001")
    elif c[-1] != 1.0:
        errors.append(f"closed upper bound cdf[-1]={c[-1]!r} != 1")
    return errors


# --------------------------------------------------------------------------------------
# Production path: forecasting-tools 0.3.5 NumericDistribution (used by main.py and by
# the Market Pulse release via NumericDistribution.from_question(...).get_cdf()).
# --------------------------------------------------------------------------------------

def sdk_cdf(percentiles: list[tuple[float, float]], geo: Geometry) -> np.ndarray:
    from forecasting_tools import NumericDistribution, Percentile

    dist = NumericDistribution(
        declared_percentiles=[Percentile(percentile=p, value=v) for p, v in percentiles],
        open_upper_bound=geo.open_upper,
        open_lower_bound=geo.open_lower,
        upper_bound=geo.upper,
        lower_bound=geo.lower,
        zero_point=geo.zero_point,
        cdf_size=geo.cdf_size,
    )
    return np.array([pt.percentile for pt in dist.get_cdf()], dtype=float)


def sdk_standardize(raw_cdf, geo: Geometry) -> np.ndarray:
    """Apply the *identical* SDK/Metaculus standardisation to an arbitrary raw CDF.

    Re-using the SDK routine isolates the effect of CDF *construction* from the
    (shared, mandatory) standardisation step in every comparison.
    """
    from forecasting_tools import NumericDistribution

    dist = NumericDistribution.model_construct(
        declared_percentiles=[], open_upper_bound=geo.open_upper, open_lower_bound=geo.open_lower,
        upper_bound=geo.upper, lower_bound=geo.lower, zero_point=geo.zero_point,
        cdf_size=geo.cdf_size, standardize_cdf=True, strict_validation=True, is_date=False,
    )
    return np.asarray(dist._standardize_cdf(list(map(float, raw_cdf))), dtype=float)
