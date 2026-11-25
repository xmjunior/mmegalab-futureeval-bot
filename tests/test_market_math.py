"""Regressions for the live Market Pulse price window, bucket and CDF paths."""
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import socket
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import market_prices as prices


class MarketMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.enterClassContext(patch.object(socket.socket, "connect", side_effect=AssertionError("unexpected network")))
        import market_pulse

        cls.market = market_pulse

    def test_scorer_uses_submission_cdf_and_official_edges(self):
        q = SimpleNamespace(lower_bound=0, upper_bound=10, zero_point=None, cdf_size=11,
                            open_lower_bound=True, open_upper_bound=True)
        quantiles = dict(zip(prices.LEVELS, [2, 3, 4, 5, 6]))
        cdf = self.market.numeric_cdf(q, quantiles)
        for outcome, mass in [(-0.1, cdf[0]), (0, cdf[1]-cdf[0]),
                              (4, cdf[4]-cdf[3]), (4.00001, cdf[5]-cdf[4]),
                              (10, cdf[-1]-cdf[-2]), (10.1, 1-cdf[-1])]:
            with self.subTest(outcome=outcome):
                self.assertEqual(prices._mass(quantiles, 0, 10, outcome, 10), mass)

    def test_log_scale_archive_and_exact_boundary_use_submission_mass(self):
        q = SimpleNamespace(lower_bound=10, upper_bound=40, zero_point=0, cdf_size=11,
                            open_lower_bound=True, open_upper_bound=True)
        quantiles = dict(zip(prices.LEVELS, [15, 17, 19, 24, 30]))
        cdf = self.market.numeric_cdf(q, quantiles)
        self.assertAlmostEqual(prices._mass(quantiles, 10, 40, 20, 10, 0), cdf[5]-cdf[4])
        self.assertAlmostEqual(prices._mass(quantiles, 10, 40, 20.001, 10, 0), cdf[6]-cdf[5])

    def test_full_window_scales_empirical_changes_from_505_closes(self):
        dates = [date(2021, 1, 1) + timedelta(days=i) for i in range(505)]
        # The final 252 changes all share the same regime. The old denominator
        # discards the calm observations present in the historical quantiles.
        daily = np.tile([-1., 1.], 252) * np.repeat([0.2, 2.], 252)
        values = np.r_[12., 12.+np.cumsum(daily)]
        spot = datetime.combine(dates[-1]+timedelta(days=2), datetime.min.time(), timezone.utc)
        asof = spot-timedelta(seconds=1)
        resolve = spot+timedelta(days=12)
        with patch.object(prices, '_fred', return_value=(dict(zip(dates, values)), 'synthetic')):
            est = prices.model(46014, asof, spot, resolve)
        expected = np.sqrt(daily[-30:].std(ddof=1)/daily.std(ddof=1))
        self.assertGreater(expected, 1.1)
        np.testing.assert_allclose(est.base_quantiles-est.center, expected*(est.naive_quantiles-est.center))
        self.assertGreater(np.ptp(est.naive_quantiles), 0)

    def test_archived_vintage_scores_closed_case_without_network(self):
        row = next(r for r in prices.HISTORY if r['question_id'] == 46212)
        with patch.object(prices, '_get', side_effect=AssertionError('unexpected network')):
            estimate = prices.model(row['post_id'], row['spot']-timedelta(seconds=1), row['spot'], row['resolve'])
        self.assertEqual(estimate.source_date, date(2026, 10, 9))
        self.assertEqual(estimate.sources[1], 'sha256:9b62266bdf5a990e57c0cf9a4cbce4edb4548398cd25d630ada2bbd0f861ee3e')
        self.assertEqual(estimate.sample_size, 495)
        self.assertAlmostEqual(estimate.naive_quantiles[0], 2.914)
        self.assertNotAlmostEqual(estimate.base_quantiles[0], 2.9033250711645926, places=6)

    def test_archived_q3_rescored_even_if_old_masses_change(self):
        archive = json.loads(Path(self.market.__file__).with_name("market_pulse_q3.json").read_text())
        actual = self.market.cross_quarter_menu()
        self.assertEqual(actual, {"46010": [1.25]})
        for row in archive["cases"]:
            row["baseline_mass"] = 1e-15
            row["candidate_mass"] = {str(width): 1.0 for width in archive["widths"][str(row["post_id"])]}
        # Intercept just the archived scorecard, leaving the frozen, checked
        # ALFRED validation inputs and their loaded stats intact.
        with patch.object(Path, "read_text", return_value=json.dumps(archive)):
            self.assertEqual(self.market.cross_quarter_menu(), actual)

    def test_closed_q4_levels_cannot_qualify_against_q3_menu(self):
        levels = [r for r in prices.HISTORY if r["post_id"] in (46010, 46014)]
        stats = prices.benchmark(levels)
        self.assertEqual(stats["failed"], {})
        self.assertEqual(len(stats["cases"]), 6)
        self.assertEqual(stats["eligible"], [46014])
        self.assertEqual(stats["allowed_widths"]["46014"], [1.25])
        q3 = self.market.cross_quarter_menu()
        self.assertFalse(set(stats["eligible"]) & {int(group) for group in q3})
        holdout = next(r for r in stats["cases"] if r["question_id"] == 46238)
        self.assertLess(holdout["widths"][1.25], 0)


if __name__ == "__main__":
    unittest.main()
