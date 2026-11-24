import json
import math
import unittest
from datetime import date
from pathlib import Path

import numpy as np

import evaluate as e


class VintageTests(unittest.TestCase):
    def test_supported_archive_includes_future_revisions_but_query_does_not(self):
        obs = date(2016, 1, 1)
        rows = e.archive()
        self.assertEqual(e.snapshot(rows, date(2016, 3, 31))[obs], 4.9)
        self.assertEqual(e.snapshot(rows, e.LATEST)[obs], 4.8)
        self.assertEqual(e.snapshot(rows, date(2016, 1, 1)).get(obs), None)

    def test_split_depends_on_label_release_not_forecast_date_alone(self):
        self.assertEqual(e.split(date(2006, 12, 31), date(2007, 1, 2)), "purged")
        self.assertEqual(e.split(date(2015, 12, 31), date(2016, 1, 1)), "purged")
        self.assertEqual(e.split(date(2015, 9, 30), date(2015, 11, 2)), "valid")

    def test_input_revisions_and_future_labels_cannot_affect_training_calibrator(self):
        cases = e.cohort(e.archive())
        self.assertTrue(all(known > origin for origin,_,_,_,known,_,_,_,_ in cases))
        expected = e.calibration(cases)
        corrupted = [row if row[-1] == "train" else (*row[:5], 99., np.zeros(120), np.zeros(120), row[-1])
                     for row in cases]
        self.assertEqual(expected, e.calibration(corrupted))

    def test_cases_have_distinct_keys_and_horizons(self):
        cases=e.cohort(e.archive())
        self.assertEqual(len({(r[0], r[2]) for r in cases}), len(cases))
        self.assertTrue(all(e.month_shift(r[1], r[2]) == r[3] for r in cases))
        self.assertTrue(all(r[1] <= r[0] < r[4] <= e.LATEST for r in cases))


class DistributionTests(unittest.TestCase):
    def test_truncation_continuity_quantiles_and_density(self):
        x = np.linspace(0, 100, 50_001)
        for dist in (e.Distribution(2, 1), e.Distribution(2, 1,[.1, .5, 1, 2, 3]),
                     e.Distribution(2, 1,[.1,.5,1,2,3],.03)):
            q = dist.quantile(e.P)
            self.assertTrue(np.all(np.diff(q)>0))
            np.testing.assert_allclose(dist.cdf(q), e.P, atol=1e-8)
            self.assertEqual(dist.cdf(-1),0)
            self.assertEqual(dist.cdf(101),1)
            self.assertEqual(dist.log_density(101), -math.inf)
            integral = np.trapezoid([math.exp(dist.log_density(y)) for y in x], x)
            self.assertAlmostEqual(integral, 1., delta=.004)

    def test_qcrps_midpoint_approximation_matches_gaussian_control(self):
        dist=e.Distribution(50,1)
        score=e.metrics(dist,50,50)
        exact=(math.sqrt(2)-1)/math.sqrt(math.pi)
        self.assertAlmostEqual(score["crps_q99"], exact, delta=.001)

    def test_ties_in_reported_rates_have_increasing_quantiles(self):
        for dist in e.arms(np.ones(120)*4., 3, 0).values():
            self.assertTrue(np.all(np.diff(dist.quantile(e.P))>0))
            self.assertTrue(np.isfinite(e.metrics(dist, 4.4, 4.2)["nll"]))


if __name__ == "__main__":
    unittest.main()


class ProspectiveTests(unittest.TestCase):
    def test_gap_rule_matches_full_rule_when_there_is_no_gap_and_excludes_missing_endpoints(self):
        from shadow import gap_arms
        hist=np.linspace(4.,6.,120) + .18*np.sin(np.arange(120))
        for h in e.HORIZONS:
            original=e.arms(hist,h,0.)["champion_rule"]
            np.testing.assert_allclose(gap_arms(hist,h)["champion_rule_gaps"].quantile(e.P),original.quantile(e.P))
            broken=hist.copy()
            broken[90]=np.nan
            self.assertTrue(np.all(np.isfinite(gap_arms(broken,h)["empirical_gaps"].quantile(e.P))))
            broken[0]=np.nan
            with self.assertRaises(ValueError):
                gap_arms(broken,h)

    def test_settlement_avoids_early_network_and_scores_first_release_once(self):
        import settle as s
        from datetime import datetime,timezone
        from unittest.mock import Mock
        state={"seal_sha256":s.SEAL_SHA,"cases":{}}
        fake=Mock(side_effect=AssertionError("early network"))
        state=s.advance(state,datetime(2026,12,4,13,29,tzinfo=timezone.utc),fake)
        fake.assert_not_called()
        raw=("period_start_date,UNRATE,realtime_start_date,realtime_end_date\n"
             "2026-11-01,4.3,2026-12-04,2027-01-01\n"
             "2026-11-01,4.1,2027-01-02,\n").encode()
        state=s.advance(state,datetime(2027,1,5,tzinfo=timezone.utc),lambda _day:raw)
        self.assertEqual(state["scored"],1)
        self.assertEqual(state["cases"]["2026-11-01"]["value"],4.3)
        self.assertEqual(state["distributions_scored"],2)
        newer=s.advance(state,datetime(2027,1,6,tzinfo=timezone.utc),fake)
        self.assertEqual(newer,state)
        fake.assert_not_called()

    def test_wrong_seal_fails_without_network(self):
        import settle as s
        from datetime import datetime,timezone
        with self.assertRaises(ValueError):
            s.advance({"seal_sha256":"changed","cases":{}},datetime.now(timezone.utc),
                      lambda _:self.fail("unexpected network"))
