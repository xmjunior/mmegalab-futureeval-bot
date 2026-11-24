import math
import sys
import unittest
from copy import deepcopy
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import settle

PANEL = settle.read_json(settle.HERE / "panel.json")
ARMS = {
    "C": {
        "probabilities": {k: v["baseline_C"] for k, v in PANEL["cases"].items()},
        "issued_at": "2026-11-24T19:33:50+00:00",
    }
}


def instant(d):
    return datetime(2026, 12, d, 12, tzinfo=UTC)


class Settlement(unittest.TestCase):
    def test_pending_first_official_ties_then_immutable_scores(self):
        source = Mock()
        source.observations.side_effect = AssertionError(
            "no premature Treasury request"
        )
        book = settle.settle(settle.new_state(), PANEL, ARMS, source, instant(3))
        self.assertEqual(book["coverage"]["resolved"], 0)
        self.assertEqual(book["coverage"]["target_dates"], 2)
        self.assertIsNone(book["scores"]["C"]["mean_brier"])
        source.observations.side_effect = None
        d2, d3, d9 = (date(2026, 12, i) for i in (2, 3, 9))
        # Literal decimal equality for both first targets. A later larger value must not replace them.
        source.observations.return_value = (
            {
                d2: {"2y": Decimal("4.65"), "10y": Decimal("4.95")},
                d3: {"2y": Decimal("8.0"), "10y": Decimal("9")},
                d9: {"2y": Decimal("4.66"), "10y": Decimal("4.97")},
            },
            "abc",
        )
        book = settle.settle(book, PANEL, ARMS, source, instant(4))
        self.assertEqual({q["outcome"] for q in book["resolutions"].values()}, {0})
        self.assertEqual(book["coverage"]["resolved"], 2)
        locked = deepcopy(book["resolutions"])
        source.observations.return_value[0][d2]["2y"] = Decimal("7")
        book = settle.settle(book, PANEL, ARMS, source, instant(11))
        self.assertEqual(book["status"], "finished")
        self.assertTrue(all(book["resolutions"][k] == v for k, v in locked.items()))
        self.assertEqual(book["coverage"]["resolved"], 4)
        score = book["scores"]["C"]
        key = "ust-promo-v1-20261202-2y"
        p = ARMS["C"]["probabilities"][key]
        self.assertAlmostEqual(score["scores_by_case"][key]["brier"], p * p)
        self.assertAlmostEqual(
            score["scores_by_case"][key]["log_loss_nats"], -math.log1p(-p)
        )
        source.observations.side_effect = AssertionError(
            "terminal state does not fetch"
        )
        self.assertEqual(
            settle.settle(deepcopy(book), PANEL, ARMS, source, instant(18)), book
        )

    def test_transport_and_seal_errors_remain_visible_and_missing_values_not_imputed(
        self,
    ):
        source = Mock()
        source.observations.side_effect = TimeoutError("transport failed")
        with self.assertRaises(TimeoutError):
            settle.settle(settle.new_state(), PANEL, ARMS, source, instant(10))
        source.observations.side_effect = None
        source.observations.return_value = (
            {
                date(2026, 12, 2): {"2y": Decimal("4.70")},
                date(2026, 12, 9): {"2y": Decimal("4.68"), "10y": Decimal("5.03")},
            },
            "abc",
        )
        book = settle.settle(settle.new_state(), PANEL, ARMS, source, instant(10))
        self.assertEqual(book["coverage"]["resolved"], 1)
        self.assertEqual(book["coverage"]["void"], 1)
        self.assertFalse(
            "outcome" in book["resolutions"]["ust-promo-v1-20261202-spread"]
        )
        altered = deepcopy(ARMS)
        altered["C"]["probabilities"]["ust-promo-v1-20261202-2y"] = 0.123
        with self.assertRaisesRegex(ValueError, "sealed forecast changed"):
            settle.settle(book, PANEL, altered, source, instant(12))


if __name__ == "__main__":
    unittest.main()
