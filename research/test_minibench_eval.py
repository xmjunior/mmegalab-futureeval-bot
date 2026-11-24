import json
import math
import unittest
from copy import deepcopy
from pathlib import Path

import minibench_eval as mb

T0 = mb.timestamp("2026-11-25T00:00:00Z")
NOW = mb.timestamp("2026-12-01T00:00:00Z")
Q = {
    "id": 3,
    "status": "resolved",
    "type": "binary",
    "open_time": "2026-11-25T12:00:00Z",
    "actual_close_time": "2026-11-25T15:00:00Z",
    "actual_resolve_time": "2026-11-29T10:00:00Z",
    "resolution": "yes",
}
F = {
    "question_id": 3,
    "author_id": mb.BOT_ID,
    "start_time": mb.timestamp("2026-11-25T12:01:00Z").timestamp(),
    "end_time": None,
    "forecast_values": [0.2, 0.8],
}
ME = {"id": mb.BOT_ID, "is_bot": True, "is_primary_bot": True, "username": mb.BOT}


class Api:
    def __init__(self):
        self.current = 41
        self.me = ME
        self.calls = []
        self.own = [F]

    def get(self, path, **params):
        self.calls.append((path, params))
        if path == "/users/me/":
            return self.me
        if path.startswith("/projects/tournaments/"):
            i = (
                self.current
                if path.endswith("/minibench/")
                else int(path.split("/")[-2])
            )
            return {
                "id": i,
                "slug": f"minibench-2026-11-{i - 20}",
                "start_date": f"2026-11-{i - 20}T00:00:00Z",
            }
        if path == "/posts/":
            if params["statuses"] != "resolved":
                return {"results": []}
            return {
                "results": [
                    {
                        "status": "resolved",
                        "projects": {"default_project": {"id": params["tournaments"]}},
                        "title": "NEVER EMIT",
                        "question": dict(
                            Q,
                            title="PRIVATE TEXT",
                            aggregations={"private": "SHOULD NOT EMIT"},
                        ),
                    }
                ],
                "next": None,
            }
        if path == "/questions/bulk-forecast-read/":
            return {"results": [{"question_id": 3, "forecasts": self.own}]}
        raise AssertionError(path)


class Mini(unittest.TestCase):
    def test_protocol_digest(self):
        raw = Path(mb.__file__).with_name("minibench_protocol.json").read_bytes()
        self.assertEqual(mb.hashlib.sha256(raw).hexdigest(), mb.PROTOCOL)
        self.assertEqual(mb.timestamp(json.loads(raw)["frozen_before_first_open"]), T0)

    def test_metrics_and_calibration(self):
        s = mb.new_stats()
        self.assertEqual(mb.classify(Q, [F], T0, NOW, s), "paired")
        self.assertEqual(s["n"], 1)
        self.assertAlmostEqual(s["ln_base_sum"], -math.log(0.8))
        self.assertAlmostEqual(s["ln_c_sum"], -math.log(0.77))
        self.assertAlmostEqual(s["brier_c_sum"], 0.23**2)
        self.assertEqual(s["bins"]["base"][-1][0], 1)
        self.assertEqual(s["bins"]["c"][3][0], 1)

    def test_bad_inputs_do_not_score(self):
        trials = [
            (dict(Q, open_time="2026-11-24T10:00:00Z"), [F]),
            (dict(Q, resolution="ambiguous"), [F]),
            (Q, [F, F]),
            (Q, [dict(F, forecast_values=[0.1, 0.95])]),
            (Q, [dict(F, forecast_values=[float("nan"), 0.3])]),
            (Q, [dict(F, start_time=mb.timestamp(Q["actual_close_time"]).timestamp())]),
            (Q, [dict(F, end_time=F["start_time"] + 20)]),
            (Q, [dict(F, start_time="2026-11-25T12:01:00")]),
        ]
        for q, fs in trials:
            with self.subTest(q=q, fs=fs):
                s = mb.new_stats()
                self.assertNotEqual(mb.classify(q, fs, T0, NOW, s), "paired")
                self.assertEqual(s["n"], 0)

    def test_account_guard_before_questions(self):
        for bad in [
            dict(ME, is_primary_bot=False),
            dict(ME, id=42),
            dict(ME, username="different"),
            dict(ME, is_bot=False),
        ]:
            api = Api()
            api.me = bad
            with self.assertRaises(RuntimeError):
                mb.update(api, {"rounds": {}}, T0, NOW)
            self.assertEqual(len(api.calls), 1)

    def test_only_own_resolved_rows_and_idempotence(self):
        api = Api()
        ledger = {"rounds": {}}
        mb.update(api, ledger, T0, NOW)
        self.assertEqual(ledger["rounds"]["41"]["stats"]["n"], 1)
        first = deepcopy(ledger)
        mb.update(api, ledger, T0, NOW)
        self.assertEqual(ledger, first)
        self.assertNotIn("PRIVATE", str(ledger))
        self.assertNotIn("NEVER EMIT", str(ledger))
        self.assertNotIn("forecast_values", str(ledger))
        for path, params in api.calls:
            if path == "/posts/":
                self.assertEqual(params["with_cp"], "false")
                self.assertEqual(params["statuses"], "resolved")
        api.own = [dict(F, author_id=999)]
        with self.assertRaises(RuntimeError):
            mb.update(api, ledger, T0, NOW)

    def test_rotating_round_completes_old(self):
        api = Api()
        ledger = {"rounds": {}}
        mb.update(api, ledger, T0, NOW)
        api.current = 42
        mb.update(api, ledger, T0, NOW)
        self.assertTrue(ledger["rounds"]["41"]["complete"])
        self.assertFalse(ledger["rounds"]["42"]["complete"])
        self.assertFalse(mb.belongs({"projects": {"default_project": {"id": 65}}}, 41))
        with self.assertRaises(RuntimeError):
            mb.tournament_id({"id": 41, "slug": "unrelated-tournament"})

    def test_one_fixed_confirmatory_review(self):
        rounds = {}
        for n in range(12):
            s = mb.new_stats()
            s.update(n=10, ln_base_sum=9, ln_c_sum=8)
            rounds[str(n)] = {"start": n, "project": n, "stats": s, "complete": True}
        self.assertIsNone(mb.review(dict(list(rounds.items())[:-1])))
        reviewed = mb.review(rounds)
        self.assertAlmostEqual(reviewed["gain_nats"], 0.1)
        self.assertLess(reviewed["p_two_sided"], 0.05)
        self.assertFalse(reviewed["automatic_promotion"])
        rounds["0"]["complete"] = False
        self.assertIsNone(mb.review(rounds))
