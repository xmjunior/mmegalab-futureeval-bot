import copy
import unittest
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock

import live


class Trials(unittest.TestCase):
    def setUp(self):
        self.book = {"schema":live.SCHEMA, "cases":{}}
        self.day = datetime(2026,11,24,15,tzinfo=UTC)
        self.source = Mock()
        dates = [date(2026,1,1)+timedelta(days=i) for i in range(365)]
        self.source.observations.return_value = ({d:{"10y":Decimal(i%9)/100,"2y":Decimal(0)}
                                                   for i,d in enumerate(dates)}, "fixture")
        self.call = Mock(return_value=('{"p":0.6}',{"prompt_tokens":100,"completion_tokens":40}))
        self.saved = []
        self.save = lambda _: self.saved.append(copy.deepcopy(self.book))

    def go(self):
        live.advance(self.book,self.source,lambda:self.day,self.call,self.save)

    def test_future_rows_not_in_prompt_and_assets_persist(self):
        self.go()
        case = next(iter(self.book["cases"].values()))
        self.assertLess(case["design"]["evidence"]["observations"][-1][0],"2026-11-24")
        self.assertNotIn("empirical reference probability",self.call.call_args_list[0].args[0])
        self.assertIn("empirical reference probability",self.call.call_args_list[1].args[0])
        self.go()
        self.assertEqual(self.call.call_count,2)
        self.assertEqual(live.summary(self.book)["paired"],0)

    def test_ambiguous_crash_cannot_reinfer_and_settlement_waits(self):
        self.go()
        self.book = self.saved[1]
        self.call.reset_mock()
        self.go()
        self.assertEqual(self.call.call_count,1)
        self.day += timedelta(days=9)
        self.go()
        res = live.summary(self.book)
        self.assertEqual((res["issued"],res["resolved"],res["paired"]),(1,1,0))

    def test_sealed_forecasts_score_after_publication_only(self):
        self.go()
        self.day += timedelta(days=8)
        self.go()
        self.assertEqual(live.summary(self.book)["paired"],0)
        self.day += timedelta(days=1)
        self.go()
        self.assertEqual(live.summary(self.book)["scores"]["coin"]["brier"],.25)
        self.assertEqual(live.summary(self.book)["paired"],1)

    def test_changed_arm_or_outcome_fails_before_scoring(self):
        self.go()
        self.day += timedelta(days=9)
        self.go()
        case = next(iter(self.book["cases"].values()))
        old = case["models"]["blind"]["p"]
        case["models"]["blind"]["p"] = .9
        with self.assertRaisesRegex(ValueError, "invalid sealed arm"): live.summary(self.book)
        case["models"]["blind"]["p"] = old
        case["resolution"]["outcome"] ^= 1
        with self.assertRaisesRegex(ValueError, "changed resolution"): live.summary(self.book)

    def test_no_backfill_or_rescore_after_last_window(self):
        self.day += timedelta(days=2)
        self.go()
        self.source.observations.assert_not_called()
        self.call.assert_not_called()

    def test_checkpoint_failure_prevents_inference_and_mutation_is_detected(self):
        self.save=Mock(side_effect=RuntimeError("journal unavailable"))
        with self.assertRaises(RuntimeError): self.go()
        self.call.assert_not_called()
        next(iter(self.book["cases"].values()))["design"]["reference"]=.99
        with self.assertRaisesRegex(ValueError,"changed frozen"): self.go()

if __name__ == "__main__":
    unittest.main()
