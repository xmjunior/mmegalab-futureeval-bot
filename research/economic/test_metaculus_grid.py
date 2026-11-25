import math
import unittest
import subprocess
import sys
import json

import numpy as np
import metaculus_grid as grid
from paired_economic import masses as archived_masses


class EconomicGridTests(unittest.TestCase):
    def test_open_tails_and_step_bounds_with_narrow_distributions(self):
        cases = [("vix", [14, 15, 16, 17, 18]), ("vix", [45] * 5), ("credit", [4] * 5),
                 ("credit", [-100] * 5), ("vix", [180] * 5)]
        self.assertLess(archived_masses(*cases[0])[1][-1], .001)
        for series, quantiles in cases:
            with self.subTest(series=series, quantiles=quantiles):
                edges, mass = grid.masses(series, quantiles)
                n = len(edges) - 1
                self.assertAlmostEqual(mass.sum(), 1.0, places=13)
                self.assertGreaterEqual(min(mass[0], mass[-1]), .001)
                self.assertGreaterEqual(mass[1:-1].min(), .01 / n)
                self.assertLessEqual(mass[1:-1].max(), 40 / n)

    def test_boundary_assignment_and_nearby_observations(self):
        q = [5, 9, 12, 18, 27]
        edges, mass = grid.masses("vix", q)
        for value, bucket in ((-1., 0), (0., 1), (10., 10), (10 + 1e-6, 11),
                              (10 - 1e-6, 10), (120., 120), (121., 121)):
            result = grid.score("vix", q, value)
            self.assertEqual(result["bucket"], bucket)
            self.assertAlmostEqual(result["log_mass_nats"], math.log(mass[bucket]))

    def test_cli_emits_one_json_record(self):
        output = subprocess.check_output([sys.executable, grid.__file__, "vix", "10",
                                          "5", "9", "12", "18", "27"], text=True)
        self.assertEqual(len(output.splitlines()), 1)
        self.assertEqual(json.loads(output), grid.score("vix", [5, 9, 12, 18, 27], 10))

    def test_invalid_inputs(self):
        for q in ([1, 2], [1, 2, 3, 5, 4], [1, 2, 3, 4, np.nan]):
            with self.assertRaises(ValueError):
                grid.masses("vix", q)
        with self.assertRaises(ValueError):
            grid.score("vix", [1]*5, math.inf)
        with self.assertRaises(ValueError):
            grid.masses("wrong", [1]*5)
