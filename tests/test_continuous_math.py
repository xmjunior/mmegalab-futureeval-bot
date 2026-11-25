import hashlib
import unittest
from datetime import datetime, timedelta, timezone
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from forecasting_tools import NumericDistribution

from continuous_math import GridDistribution, continuous_bucket, grid_distribution, outcome_grid, student_t_distribution


def question(lower=0, upper=100, zero=None, size=201, open_lower=True, open_upper=True):
    return SimpleNamespace(lower_bound=lower, upper_bound=upper, zero_point=zero, cdf_size=size,
                           open_lower_bound=open_lower, open_upper_bound=open_upper)


def heights(dist):
    return [point.percentile for point in dist.get_cdf()]


class ContinuousMathTests(unittest.TestCase):
    points = [(.075, 8), (.25, 23), (.5, 50), (.9, 66)]

    def test_antigravity_reference_linear_and_log(self):
        # Frozen output from 49d433af tq_cdf(..., nu=3) with SciPy and the locked SDK.
        for q, expected in [
            (question(), [.0427337455, .044223356, .2652555126, .5, .9553555167, .9916179765, .9918678166]),
            (question(1, 1000, 0, 23), [.0131123081, .0166102193, .0472283279, .3431122231, .9949853437, .998274184, .9988048476]),
        ]:
            with self.subTest(q=q):
                dist = student_t_distribution(q, self.points)
                n = q.cdf_size
                indices = [0, 1, n//4, n//2, 3*n//4, n-2, n-1]
                self.assertEqual([heights(dist)[i] for i in indices], expected)
                sdk = NumericDistribution.model_construct(lower_bound=q.lower_bound,
                            upper_bound=q.upper_bound, zero_point=q.zero_point)
                sdk_grid = [sdk._cdf_location_to_nominal_location(i/(n-1)) for i in range(n)]
                np.testing.assert_allclose([p.value for p in dist.get_cdf()], sdk_grid, rtol=1e-13)

    def test_complete_antigravity_grids(self):
        # SHA-256 of 49d433af builders.tq_cdf(self.points, Geometry(...), nu=3)
        # for these 24 geometries. Canonical form: one 10-decimal CDF value per line.
        digest = hashlib.sha256()
        for (lo, hi, zero), size, lo_open, hi_open in product(
            [(0, 100, None), (1, 1000, 0)], [8, 23, 201], [True, False], [True, False]
        ):
            q = question(lo, hi, zero, size, lo_open, hi_open)
            for h in heights(student_t_distribution(q, self.points)):
                digest.update(f"{h:.10f}\n".encode())
        self.assertEqual(digest.hexdigest(), "ea67e1ebb07e89ff688de571305ab07a205b8d40c82d06d4fa74ed34f247c24a")

    def test_reflected_log_grid_and_percentiles(self):
        # Reflect [1,100] around 0, exchanging left and right tails.
        a = student_t_distribution(question(1, 100, 0, 21), self.points)
        pts = [(1-p, -x) for p,x in reversed(self.points)]
        b = student_t_distribution(question(-100, -1, 0, 21), pts)
        np.testing.assert_allclose(heights(b), 1-np.array(heights(a)[::-1]), atol=1e-10)
        np.testing.assert_allclose([p.value for p in b.get_cdf()],
                                   [-p.value for p in reversed(a.get_cdf())], rtol=1e-13)

    def test_discrete_atoms_and_open_closed_tails(self):
        for lower_open in (True, False):
            for upper_open in (True, False):
                q = question(0, 10, size=11, open_lower=lower_open, open_upper=upper_open)
                for value in (0, 5, 10):
                    with self.subTest(value=value, lower_open=lower_open, upper_open=upper_open):
                        h = heights(student_t_distribution(q, [(0.1,value), (0.9,value)]))
                        masses = np.diff(h)
                        self.assertEqual(int(np.argmax(masses)) + 1, max(1, value))
                        self.assertGreater(max(masses), .98)
                        self.assertEqual(h[0], .001 if lower_open else 0.)
                        self.assertEqual(h[-1], .999 if upper_open else 1.)

    def test_atom_on_fractional_non_decimal_grid_edge(self):
        for q in [question(0, 7, size=8), question(1, 31, 0, size=8),
                  question(-31, -1, 0, size=8)]:
            edges = outcome_grid(q, size=8)
            x = edges[5]
            h = heights(student_t_distribution(q, [(.1,x),(.9,x)]))
            self.assertEqual(continuous_bucket(x, q.lower_bound, q.upper_bound, 7, q.zero_point), 5)
            self.assertEqual(int(np.argmax(np.diff(h))) + 1, 5)
            self.assertEqual((edges[0], edges[-1]), (q.lower_bound, q.upper_bound))

    def test_date_units_and_naive_rejection(self):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        delta = timedelta(days=1)
        d = question(start, start + 100*delta)
        distribution = student_t_distribution(d, [(p, start + x*delta) for p,x in self.points])
        self.assertTrue(distribution.is_date)
        self.assertEqual(heights(distribution), heights(student_t_distribution(question(), self.points)))
        with self.assertRaises(ValueError):
            student_t_distribution(d, [(0.2,start.replace(tzinfo=None)), (0.8,start+delta)])

    def test_grid_roundtrip_and_aggregate_no_interpolation(self):
        d = student_t_distribution(question(), self.points)
        with patch.object(NumericDistribution, 'get_cdf', side_effect=AssertionError), \
             patch.object(NumericDistribution, '_standardize_cdf', side_effect=AssertionError):
            copy = GridDistribution.model_validate_json(d.model_dump_json())
            agg = grid_distribution(question(), heights(copy))
            self.assertEqual(copy.model_dump(), d.model_dump())
            self.assertEqual(heights(agg), heights(d))
        for change in ('percentile', 'value'):
            bad = d.model_dump()
            bad['declared_percentiles'][5][change] = 0
            with self.assertRaises(ValueError):
                GridDistribution.model_validate(bad)

    def test_input_protections(self):
        self.assertRaises(ValueError, student_t_distribution, question(), [(.2, 50),(.3,49)])
        self.assertRaises(ValueError, student_t_distribution, question(), [(.2, 5),(.20001,49)])
        self.assertRaises(ValueError, student_t_distribution, question(), [(.2, 5),(.8,500)])
        self.assertRaises(ValueError, student_t_distribution, question(), [(.2, 140),(.8,150)])
        self.assertRaises(ValueError, student_t_distribution, question(open_lower=False), [(.1,-1),(.9,50)])
        self.assertRaises(ValueError, student_t_distribution, question(1, 100, 0), [(.2,-1),(.8,49)])
        self.assertRaises(ValueError, student_t_distribution, question(0, 100, 50), self.points)
        self.assertRaises(ValueError, student_t_distribution, question(), [(.2,5),(.8,float('nan'))])

    def test_resolved_bucket_boundaries_and_scale(self):
        self.assertEqual([continuous_bucket(x, 0, 100, 10) for x in [-1,0,10,10.001,100,101]],
                         [0,1,1,2,10,11])
        self.assertEqual(continuous_bucket(10, 1, 100, 2, 0), 1)
        self.assertEqual(continuous_bucket(-10, -100, -1, 2, 0), 1)
        self.assertEqual(continuous_bucket(-9.99, -100, -1, 2, 0), 2)
