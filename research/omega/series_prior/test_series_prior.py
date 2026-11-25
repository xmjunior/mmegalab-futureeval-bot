import unittest
from evaluate import fit, predict
from data import assigned_targets


class SeriesPriorTests(unittest.TestCase):
    def test_shrinkage_counts_rounds_and_leaves_unseen_series_exact(self):
        def row(rd, y):
            return ["train", rd, "fred", "spread", "2025-12-30", "[]", 0.2,0.3,0.4,y,"fred|spread"]
        train = [row("2025-10-26",1), row("2025-10-26",1), row("2025-11-09",0)]
        model = fit(train)
        self.assertEqual(model["fred","spread"], (1.,2))
        self.assertAlmostEqual(predict(model,"fred","spread",.2)[0], (1+4*.2)/6)
        self.assertEqual(predict(model,"fred","new",.2),(.2,False))

    def test_holdout_and_disallowed_training_target(self):
        row = ["test","2026-03-01","fred","spread","2026-03-08","[]",0,0,0,1,"fred|spread"]
        self.assertEqual(fit([row]), {})
        row[:2] = ["train","2025-10-26"]
        with self.assertRaises(ValueError):
            fit([row])

    def test_issue_time_embargo_includes_even_unresolved_targets(self):
        item = {"source":"fred","id":"spread","question":"Historical fixture",
                "resolution_dates":["2025-11-23","2026-05-10"]}
        self.assertEqual(set(assigned_targets([item])),
                         {("fred","spread","2025-11-23"),("fred","spread","2026-05-10")})
        self.assertEqual(assigned_targets([{**item,"direction":["conditional"]}]),{})
        self.assertEqual(assigned_targets([{**item,"id":["composite","target"]}]),{})

if __name__ == "__main__":
    unittest.main()
