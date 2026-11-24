"""Forecast safety and reproducibility checks; every model response here is synthetic."""
import json
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError

import shadow as s

ISSUE = datetime(2026, 11, 24, 18, tzinfo=timezone.utc)


class Source:
    def __init__(self):
        self.history = {ISSUE.date() - timedelta(days=i): (Decimal("4.50") + Decimal(i % 7)/100,
                         Decimal("4.20")) for i in range(145)}
        self.history[ISSUE.date() + timedelta(days=1)] = (Decimal("99.99"), Decimal("0"))
        self.hashes = {s.TREASURY + "2026": "ab" * 32}

    def year(self, year):
        return self.history


class FakeRoutes:
    def __init__(self):
        self.disabled = set()
        self.calls = []

    def ready(self, model):
        return True

    def verified_free(self, model):
        return False

    def call(self, model, prompt):
        self.calls.append((model,prompt))
        if "You do not produce forecasts yourself" in prompt:
            return "Synthetic evidence.", model, {}
        p = 70 if model in s.GEMMAS else (40 if "Verified official" in prompt else 40)
        return f"Probability: {p}%", model, {}


class ShadowTests(unittest.TestCase):
    def setUp(self):
        self.clock = mock.patch.object(s, "now_utc", return_value=ISSUE)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.source = Source()
        with mock.patch.object(s,"sofr_evidence", return_value={"status": "fixture"}):
            self.case = s.issue(ISSUE,self.source)
        self.templates = s.production_prompts()

    def test_future_data_and_equality(self):
        e = self.case["evidence"]
        self.assertEqual(e["last_10y_percent"],"4.51")
        self.assertEqual(e["latest_observation"],"2026-11-23")
        self.assertEqual(self.case["threshold_percent"],"4.46")
        target = date.fromisoformat(self.case["target"])
        self.source.history[target] = (Decimal("4.46"), Decimal("4.20"))
        later = ISSUE + timedelta(days=20)
        s.resolve(self.case, self.source, later)
        self.assertEqual(self.case["resolution"]["outcome"],0)
        self.source.history[target] = (Decimal("8"),Decimal("7"))
        s.resolve(self.case, self.source,later)
        self.assertEqual(self.case["resolution"]["outcome"],0)

    def test_exact_prompt_and_independent_families(self):
        r = FakeRoutes()
        s.forecast(self.case,r,self.templates,lambda:None)
        self.assertEqual(len(r.calls),5)
        self.assertEqual(self.case["strategies"]["D"]["p"],.55)
        self.assertEqual(self.case["strategies"]["E"]["reason"], "no_verified_free_frontier_route")
        self.assertIn("Today is 2026-11-24.", r.calls[1][1])
        self.assertIn("The last thing you write is your final answer", r.calls[1][1])
        self.assertIn("Your research assistant says:\nSynthetic evidence.",r.calls[1][1])
        self.assertIn(s.TREASURY, r.calls[4][1])
        original = json.dumps(self.case,sort_keys=True)
        s.forecast(self.case,r,self.templates,lambda:None)
        self.assertEqual(json.dumps(self.case,sort_keys=True), original)
        self.assertEqual(len(r.calls),5)

    def test_status_saturation_no_duplicate_no_late_call(self):
        r = FakeRoutes()
        def saturated(*_):
            raise HTTPError("https://example.test", 429, "test", {"Retry-After":"7400"},None)
        r.call = mock.Mock(side_effect=saturated)
        s.forecast(self.case,r,self.templates,lambda:None)
        self.assertEqual(r.call.call_count,3)
        self.assertNotIn("p",self.case["strategies"]["D"])
        self.assertEqual(self.case["attempts"][0]["retry_at"],(ISSUE+timedelta(seconds=7400)).isoformat())
        fresh = FakeRoutes()
        s.forecast(self.case,fresh,self.templates,lambda:None)
        self.assertEqual(fresh.calls,[])
        with mock.patch.object(s,"now_utc",return_value=ISSUE+timedelta(days=1)):
            s.get_reply(self.case,s.CANDIDATE,"research","late",fresh,lambda:None)
        self.assertEqual(fresh.calls,[])

    def test_free_routes_fail_closed(self):
        r = s.Routes()
        with mock.patch.object(s,"request",return_value=b'{"data":[{"id":"nvidia/nemotron-3-ultra-550b-a55b:free","pricing":{"prompt":"0","completion":"0.12"}}]}') as calls:
            with self.assertRaisesRegex(ValueError,"zero_price_unverified"):
                r.call(s.CANDIDATE,"test")
            self.assertEqual(calls.call_count,1)
        with mock.patch.object(s,"request",return_value=b'pricing not in expected format') as calls:
            with self.assertRaisesRegex(ValueError,"zero_price_unverified"):
                r.call(s.GEMMAS[0],"test")
            self.assertEqual(calls.call_count,1)
        with self.assertRaisesRegex(ValueError,"outside_allowlist"):
            r.call("paid_model","test")

    def test_frontier_adjudicates_exactly_once_on_verified_free_route(self):
        m = s.FRONTIERS[0]
        self.case["strategies"] = {"champion": {"p": .7}, "A": {"p": .5}}
        prices = {"id": m, "pricing": {"prompt": "0", "completion": "0", "request": "0"}}
        reply = {"model": m, "choices": [{"message": {"content": "Probability: 58%"}}], "usage": {"cost": 0}}
        with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "unit-test-only"}), mock.patch.object(
            s, "request", side_effect=[json.dumps({"data":[prices]}).encode(), json.dumps(reply).encode()]
        ) as calls:
            s.adjudicate(self.case,s.Routes(),self.templates["binary"],lambda:None)
            self.assertEqual(self.case["strategies"]["E"]["p"],.58)
            self.assertEqual(calls.call_args_list[1].args[1]["provider"]["max_price"],{"prompt":0,"completion":0,"request":0})
            self.assertEqual(calls.call_args_list[1].args[1]["model"],m)
            s.adjudicate(self.case,s.Routes(),self.templates["binary"],lambda:None)
            self.assertEqual(calls.call_count,2)

    def test_frontier_abstains_on_price_missing_comparator_or_small_disagreement(self):
        self.case["strategies"] = {"champion":{"p": .7}, "A":{"p": .4}}
        m = s.FRONTIERS[0]
        paid = [{"id": m, "pricing":{"prompt":"0","completion":"0","request":".00001"}},
                {"id": m.removesuffix(":free"), "pricing":{"prompt":"0","completion":"0"}}]
        with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY":"unit-test-only"}), mock.patch.object(
            s, "request",return_value=json.dumps({"data":paid}).encode()
        ) as calls:
            s.adjudicate(self.case,s.Routes(),self.templates["binary"],lambda:None)
            self.assertEqual(self.case["strategies"]["E"]["reason"],"no_verified_free_frontier_route")
            self.case["strategies"]["A"] = {"p": .66}
            s.adjudicate(self.case,s.Routes(),self.templates["binary"],lambda:None)
            self.assertEqual(self.case["strategies"]["E"]["reason"],"disagreement_below_0.20")
            self.case["strategies"].pop("A")
            s.adjudicate(self.case,s.Routes(),self.templates["binary"],lambda:None)
            self.assertEqual(self.case["strategies"]["E"]["reason"],"disagreement_unobservable")
            self.assertEqual(calls.call_count,1)
            self.assertEqual(self.case["attempts"],[])

    def test_frontier_single_failed_call_and_credential_redirect(self):
        self.case["strategies"] = {"champion":{"p": .7}, "A":{"p": .4}}
        m = s.FRONTIERS[0]
        r = s.Routes()
        r.free_catalog = {m:{"prompt":"0","completion":"0"}}
        with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY":"unit-test-only"}), mock.patch.object(
            s, "request",side_effect=HTTPError("https://example.test",503,"fixture",{},None)
        ) as calls:
            s.adjudicate(self.case,r,self.templates["binary"],lambda:None)
            s.adjudicate(self.case,s.Routes(),self.templates["binary"],lambda:None)
            self.assertEqual(calls.call_count,1)
            self.assertNotIn("p",self.case["strategies"]["E"])
        with self.assertRaisesRegex(ValueError,"credentialed_redirect_refused"):
            s.NoAuthRedirect().redirect_request(None,None,307,"fixture",{},"https://example.test")

    def test_invalid_bounds_and_no_label_yet(self):
        for txt in ("Probability: NaN%","Probability: 103%","I guess it is 42%.",""):
            with self.assertRaises(ValueError):
                s.probability(txt)
        s.resolve(self.case,self.source,ISSUE + timedelta(days=15))
        self.assertIsNone(self.case["resolution"])
        self.assertEqual(s.probability("Probability: 0%"),.01)
        self.assertEqual(s.probability("Final answer: Probability: 100%"),.99)

    def test_pairs_do_not_impute_missing(self):
        r = FakeRoutes()
        s.forecast(self.case,r,self.templates,lambda:None)
        self.case["resolution"] = {"outcome": 1}
        self.case["strategies"]["B"] = {"status": "unavailable"}
        summary = s.summarize({"cases":{"fixture":self.case}})
        self.assertEqual(summary["arms"]["B"]["paired_with_champion_n"],0)
        self.assertEqual(summary["arms"]["D"]["paired_with_champion_n"],1)
        self.assertAlmostEqual(summary["arms"]["champion"]["brier"],.09)
        self.assertLess(summary["arms"]["D"]["paired_brier_improvement"],0)

    def test_state_persistence(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"a"/"ledger.json"
            s.write_ledger(path, self.case)
            self.assertEqual(json.loads(path.read_bytes()),self.case)
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
