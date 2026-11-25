"""Student-t quantile interpolation on the Metaculus axis (Antigravity 49d433af).

The pinned forecasting-tools SDK standardizes the CDF exactly once. Its normal
quantile interpolation must not be applied to the resulting submission grid.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Literal
import math

import numpy as np
from forecasting_tools import NumericDistribution, Percentile
from pydantic import model_validator


def _number(value):
    if isinstance(value, datetime):
        if value.utcoffset() is None:
            raise ValueError("A date forecast requires a timezone")
        value = value.timestamp()
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("A continuous scale needs finite values")
    return value


@dataclass(frozen=True)
class _Axis:
    lower: float
    upper: float
    zero: float | None
    size: int

    def __post_init__(self):
        if not self.upper > self.lower or not math.isfinite(self.upper - self.lower):
            raise ValueError("Invalid continuous bounds")
        if self.size is None or self.size < 2 or not isinstance(self.size, int):
            raise ValueError("At least two CDF grid points are required")
        if self.zero is not None and self.lower <= self.zero <= self.upper:
            raise ValueError("The log origin must be outside the question bounds")

    def _log_ratio(self, value):
        # log((value-zero)/(lower-zero)), with log1p near a linear scale.
        distance = value - self.zero
        base = self.lower - self.zero
        if distance == 0 or (distance > 0) != (base > 0):
            raise ValueError("Outcome crosses the question's logarithmic origin")
        offset = (value - self.lower) / base
        if abs(offset) < 0.5:
            return math.log1p(offset)
        return math.log(abs(distance)) - math.log(abs(base))

    def location(self, value):
        if self.zero is None:
            return (value - self.lower) / (self.upper - self.lower)
        return self._log_ratio(value) / self._log_ratio(self.upper)

    def nominal_grid(self):
        u = np.linspace(0.0, 1.0, self.size)
        if self.zero is None:
            values = self.lower + (self.upper - self.lower) * u
        else:
            log_r = self._log_ratio(self.upper)
            # Reflect decreasing derivative ratios so expm1 uses the small end
            # of the interval. For enormous ratios avoid computing exp(log_r).
            at = u if log_r > 0 else 1 - u
            k = abs(log_r)
            if k < 700:
                weight = np.expm1(k * at) / math.expm1(k)
            else:
                weight = np.exp(k * (at - 1)) * -np.expm1(-k * at) / -math.expm1(-k)
            if log_r > 0:
                values = self.lower + (self.upper - self.lower) * weight
            else:
                values = self.upper - (self.upper - self.lower) * weight
        values[0], values[-1] = self.lower, self.upper
        if np.any(np.diff(values) <= 0) or not np.all(np.isfinite(values)):
            raise ValueError("Question grid cannot be represented in floating point")
        return values


def continuous_bucket(value, lower, upper, bins, zero_point=None):
    """Official interior edge convention (left, right], exterior bins 0 and N+1."""
    axis = _Axis(_number(lower), _number(upper), None if zero_point is None else _number(zero_point), bins + 1)
    value = _number(value)
    if value < axis.lower:
        return 0
    if value > axis.upper:
        return bins + 1
    return min(bins, max(1, int(axis.location(value) * bins + 1 - 1e-10)))


def _t3_tail(magnitude):
    # With theta=atan(sqrt(3)/|x|), SF(x)=(theta-sin(2*theta)/2)/pi.
    theta = math.atan2(math.sqrt(3), magnitude)
    if theta < 0.01:
        square = theta * theta
        return theta * square * (2/3 - square * (2/15 - square * 4/315)) / math.pi
    return (theta - 0.5 * math.sin(2 * theta)) / math.pi


@lru_cache(maxsize=128)
def _t3_score(p):
    if p == 0.5:
        return 0.0
    tail = min(p, 1 - p)
    low, high = 0.0, math.sqrt(3) / tail ** (1/3)
    for _ in range(64):
        mid = (low + high) / 2
        if _t3_tail(mid) > tail:
            low = mid
        else:
            high = mid
    return math.copysign((low + high) / 2, p - 0.5)


class GridDistribution(NumericDistribution):
    """An already standardized submission grid, including after Pydantic reload."""

    grid_format: Literal["megalab-cdf-v1"] = "megalab-cdf-v1"

    @model_validator(mode="after")
    def validate_percentiles(self):
        axis = _Axis(_number(self.lower_bound), _number(self.upper_bound),
                     None if self.zero_point is None else _number(self.zero_point), self.cdf_size)
        points = self.declared_percentiles
        if len(points) != axis.size or [p.value for p in points] != axis.nominal_grid().tolist():
            raise ValueError("Distribution does not match the question's submission grid")
        h = np.array([p.percentile for p in points])
        if not np.all(np.isfinite(h)) or np.any((h < 0) | (h > 1)):
            raise ValueError("Invalid CDF probabilities")
        if (h[0] < 0.001 if self.open_lower_bound else h[0] != 0.0):
            raise ValueError("Invalid lower tail")
        if (h[-1] > 0.999 if self.open_upper_bound else h[-1] != 1.0):
            raise ValueError("Invalid upper tail")
        steps = np.diff(h)
        n = len(steps)
        if np.any(steps < 0.01/n - 1.01e-10) or np.any(steps > 40/n + 1.01e-10):
            raise ValueError("Invalid CDF step height")
        return self

    def get_cdf(self):
        return self.declared_percentiles


def _axis(question):
    return _Axis(_number(question.lower_bound), _number(question.upper_bound),
                 None if question.zero_point is None else _number(question.zero_point), question.cdf_size)


def outcome_grid(question, size=101):
    """Nominal values at equally spaced locations for index elicitation."""
    axis = _Axis(_number(question.lower_bound), _number(question.upper_bound),
                 None if question.zero_point is None else _number(question.zero_point), size)
    return axis.nominal_grid().tolist()


def grid_distribution(question, probabilities):
    """Wrap a final grid, including an already standardized median, without refitting."""
    axis = _axis(question)
    points = [Percentile(percentile=p, value=v)
              for p, v in zip(probabilities, axis.nominal_grid(), strict=True)]
    return GridDistribution(declared_percentiles=points, lower_bound=axis.lower,
                            upper_bound=axis.upper, zero_point=axis.zero, cdf_size=axis.size,
                            open_lower_bound=question.open_lower_bound, open_upper_bound=question.open_upper_bound,
                            is_date=isinstance(question.lower_bound, datetime), standardize_cdf=False)


def student_t_distribution(question, points):
    """Antigravity's fixed t(3) interpolation; duplicate outcomes retain their mass."""
    axis = _axis(question)
    points = [(p.percentile, p.value) if hasattr(p, "percentile") else p for p in points]
    if len(points) < 2:
        raise ValueError("At least two quantiles are required")
    ps, xs = zip(*points)
    ps = np.array([_number(p) for p in ps])
    xs = np.array([_number(x) for x in xs])
    if np.any(np.diff(ps) <= 0) or ps[0] <= 0 or ps[-1] >= 1 or np.any(np.diff(xs) < 0):
        raise ValueError("Quantile levels must increase inside (0,1) with nondecreasing outcomes")
    if (not question.open_lower_bound and xs[0] < axis.lower) or (not question.open_upper_bound and xs[-1] > axis.upper):
        raise ValueError("A quantile lies beyond a closed bound")
    # Retain both pinned SDK unit/scale checks and spacing guard before fitting.
    elicited = [Percentile(percentile=p, value=x) for p, x in zip(ps, xs)]
    standardizer = NumericDistribution.model_construct(
        declared_percentiles=elicited, lower_bound=axis.lower, upper_bound=axis.upper,
        open_lower_bound=question.open_lower_bound, open_upper_bound=question.open_upper_bound,
    )
    standardizer._check_percentile_spacing()
    standardizer._check_too_far_from_bounds(elicited)

    locations = np.array([axis.location(x) for x in xs])
    scores = np.array([_t3_score(float(p)) for p in ps])
    n = axis.size - 1
    grid = np.arange(axis.size) / n
    # Match the official bucket tolerance at interior edges, including log
    # transforms that round an exact nominal edge slightly to the right.
    scaled = locations * n
    nearest = np.rint(scaled)
    on_edge = (nearest > 0) & (nearest < n) & (np.abs(scaled-nearest) <= 1e-10)
    locations[on_edge] = nearest[on_edge] / n
    # np.interp selects the last of matching knots (right-continuous CDF).
    z = np.interp(grid, locations, scores)
    for i, j, mask, infinity in ((0, 1, grid < locations[0], -math.inf),
                                 (-1, -2, grid > locations[-1], math.inf)):
        width = locations[j] - locations[i]
        if width:
            z[mask] = scores[i] + (grid[mask] - locations[i]) * (scores[j] - scores[i]) / width
        else:
            z[mask] = infinity
    # The last flat extrapolated quantile segment is a point mass at the last knot.
    if locations[-1] == locations[-2]:
        z[grid == locations[-1]] = math.inf
    raw = [_t3_tail(abs(x)) if x < 0 else 1 - _t3_tail(x) for x in z]
    if not question.open_lower_bound:
        raw[0] = 0.0
    elif np.any(xs == axis.lower):
        # At this edge the API wants P(X < lower), not P(X <= lower).
        first = int(np.searchsorted(xs, axis.lower))
        raw[0] = 0.0 if first == 0 and xs[1] == xs[0] else float(ps[first])
    if not question.open_upper_bound:
        raw[-1] = 1.0
    return grid_distribution(question, standardizer._standardize_cdf(raw))
