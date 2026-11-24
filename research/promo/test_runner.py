import csv
import os
import subprocess
from decimal import Decimal
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import runner as r

NOW = datetime(2026,11,24,20,tzinfo=timezone.utc)
def sample():
    return {"source":"OpenAI owner billing dashboard", "active_positive_promotions":1,
            "cash_fallback_available":False, "other_org_usage_isolated":True,
            "currency":"USD", "grant_remaining_micro_usd":6_000_000, "target_grant_reference":"test",
            "org":"org-test", "project":"proj_test", "grant_expires_at":"2026-11-26T00:00:00+00:00",
            "checked_at":NOW.isoformat()}


def new_state():
    return {"schema":r.SCHEMA, "panel_hash":r.PANEL_SHA, "arms":{}}


class Memory:
    def __init__(self):
        self.book = new_state()
        self.calls = []

    def commit(self, message):
        self.calls.append(json.dumps(self.book))


class Tests(unittest.TestCase):
    def setUp(self):
        t = mock.patch.object(r,"utc",return_value=NOW)
        t.start()
        self.addCleanup(t.stop)
        self.panel = r.load_panel()
        self.answers = {k:.55 for k in self.panel["cases"]}

    def test_identity_and_bounds(self):
        self.assertEqual(r.sha(r.prompt().encode()), r.PROMPT_SHA)
        self.assertEqual(r.plan(2048),49_920)
        with self.assertRaises(ValueError):
            r.plan(2049)
        self.assertEqual(r.probabilities(json.dumps(self.answers),self.panel),self.answers)
        for bad in ({"wrong":.55},dict.fromkeys(self.answers,True), dict.fromkeys(self.answers,1.01)):
            with self.assertRaises(ValueError):
                r.probabilities(json.dumps(bad),self.panel)
        with self.assertRaises(ValueError):
            r.probabilities('{"aa":0.4,"aa":0.5}',self.panel)

    def test_reference_probability_reproduces_from_frozen_rows(self):
        with Path(__file__).with_name("treasury-asof.csv").open() as f:
            rows = list(csv.DictReader(f))
        dates = [r["observation_date"] for r in rows]
        self.assertEqual(dates, sorted(set(dates)))
        self.assertEqual(dates[-1], self.panel["source_observed_until"])
        for item in self.panel["cases"].values():
            ten = [Decimal(x["ten_year_percent"]) for x in rows]
            two = [Decimal(x["two_year_percent"]) for x in rows]
            vals = two if item["leg"] == "2y" else [x-y for x,y in zip(ten,two)]
            lag=item["lag"]
            wins = [vals[i] - vals[i-lag] > Decimal(item["threshold"]) - vals[-1]
                    for i in range(len(vals)-1, lag-1, -lag)]
            self.assertEqual((sum(wins),len(wins)),(item["over_threshold"],item["analogs"]))
            self.assertEqual((sum(wins)+2)/(len(wins)+4),item["baseline_C"])

    def test_billing_reports_fail_closed(self):
        d = sample()
        for changes in ({"active_positive_promotions":2},{"cash_fallback_available":True},
                     {"other_org_usage_isolated":False},{"checked_at":(NOW-timedelta(hours=1)).isoformat()},
                     {"grant_expires_at":"2026-12-26T00:00:00+00:00"}):
            raw=json.dumps({**d,**changes}).encode()
            with self.assertRaises(ValueError):
                r.audited_grant(raw,r.sha(raw),NOW)
        raw=json.dumps(d).encode()
        with self.assertRaises(ValueError):
            r.audited_grant(raw,r.BILLING_AUDIT_SHA,NOW)
        self.assertEqual(r.audited_grant(raw,r.sha(raw),NOW),d)

    def test_promotional_before_purchased_fallback_requires_real_buffer(self):
        grant = {**sample(),"cash_fallback_available":True,"other_positive_balance_verified_purchased":True,
                 "grant_remaining_micro_usd":30000}
        raw = json.dumps(grant).encode()
        self.assertEqual(r.audited_grant(raw,r.sha(raw),NOW),grant)
        r.lease(Memory(),"openai",grant,{"single":15000})
        with self.assertRaisesRegex(ValueError,"insufficient"):
            r.lease(Memory(),"openai",grant,{"single":15001})
        raw = json.dumps({**grant,"active_positive_promotions":2}).encode()
        with self.assertRaisesRegex(ValueError,"NOT_ISOLATED"):
            r.audited_grant(raw,r.sha(raw),NOW)

    def test_fenced_panel_without_extra_content(self):
        candidate = "`{3}json\n".replace("`{3}","`"*3) + json.dumps(self.answers) + "\n" + "`"*3
        self.assertEqual(r.probabilities(candidate,self.panel), self.answers)
        for bad in ("prefix " + candidate, candidate + "\nexplanation", candidate + "\n" + candidate):
            with self.assertRaises(ValueError):
                r.probabilities(bad,self.panel)

    def test_incomplete_paid_response_keeps_usage_and_stays_spent(self):
        j=Memory()
        failures = [
            {"status":"incomplete", "output":[{"type":"reasoning"}]},
            {"status":"completed", "output":[{"type":"message","content":[]}]},
        ]
        for fragment in failures:
            j=Memory()
            receipt = {**fragment, "model":r.MODEL,"service_tier":"default","id":"resp-private",
                       "usage":{"input_tokens":500,"output_tokens":448,"output_tokens_details":{"reasoning_tokens":440}}}
            with mock.patch.object(r,"prepare_openai",return_value=({},500,r.plan(500))), mock.patch.object(r,"oai",return_value=receipt) as call:
                self.assertEqual(r.run("openai",j,self.panel,sample()),"stopped_no_retry")
                got = j.book["arms"]["openai"]["request"]
                self.assertEqual((got["input_tokens"],got["output_tokens"],got["reasoning_tokens"]),(500,448,440))
                self.assertEqual(got["model"],r.MODEL)
                self.assertEqual(got["billing_reconciliation"],"pending_grant_level_statement")
                self.assertEqual(j.book["arms"]["openai"]["records"],{})
                self.assertEqual(j.book["arms"]["openai"]["max_micro_usd"],r.plan(500))
                self.assertEqual(call.call_count,1)

    def test_actions_git_auth_narrowly_scoped_and_not_in_argv(self):
        result=subprocess.CompletedProcess(args=[],returncode=0,stdout="ok",stderr="")
        with mock.patch.dict(os.environ,{"GITHUB_TOKEN":"EXAMPLE-UNIT-TEST", "GIT_CONFIG_COUNT":"0"}), mock.patch.object(r.subprocess,"run",return_value=result) as call:
            r.git(Path("."),"fetch","origin")
            cmd=call.call_args.args[0]; env=call.call_args.kwargs["env"]
            self.assertNotIn("EXAMPLE-UNIT-TEST",str(cmd))
            self.assertEqual(env["GIT_CONFIG_KEY_0"],"http."+r.REPO+".extraHeader")
            self.assertEqual(env["GIT_CONFIG_VALUE_1"],"false")
            self.assertEqual(env["GIT_TERMINAL_PROMPT"],"0")

    def test_price_drift_or_context_overrun_avoids_paid_call(self):
        with mock.patch.object(r.shadow,"request", return_value=b"unexpected price"), mock.patch.object(r,"oai") as api:
            with self.assertRaisesRegex(ValueError,"pricing_changed"):
                r.prepare_openai(r.prompt(),sample())
            api.assert_not_called()
        with mock.patch.object(r.shadow,"request", return_value=b"| Input | $4 | 1M tokens | | Output | $20 | 1M tokens |"), mock.patch.object(
            r,"oai",return_value={"object":"response.input_tokens","input_tokens":2049}) as api:
            with self.assertRaisesRegex(ValueError,"input_over_cap"):
                r.prepare_openai(r.prompt(),sample())
            self.assertEqual(api.call_args.args[0],"/input_tokens")

    def test_single_request_with_worst_case_reasoning_and_durable_reservation(self):
        journal = Memory()
        d = sample()
        counted = {"model":r.MODEL,"input":r.prompt(),"tools":[]}
        resp = {"status":"completed", "model":r.MODEL, "service_tier":"default", "id":"dummy",
                "output":[{"type":"message","content":[{"type":"output_text","text":json.dumps(self.answers)}]}],
                "usage":{"input_tokens":500,"output_tokens":448,"output_tokens_details":{"reasoning_tokens":395}}}
        def sole_call(path, params, scope):
            self.assertEqual(path,"")
            self.assertEqual(params["tools"],[])
            self.assertEqual(params["tool_choice"],"none")
            self.assertEqual(params["max_output_tokens"],448)
            self.assertEqual(len(journal.calls),1)
            self.assertEqual(journal.book["arms"]["openai"]["status"],"reserved")
            self.assertIs(scope,d)
            return resp
        with mock.patch.object(r,"prepare_openai",return_value=(counted,500,r.plan(500))), mock.patch.object(r,"oai",side_effect=sole_call) as api:
            self.assertEqual(r.run("openai",journal,self.panel,d),"completed")
            self.assertEqual(journal.book["arms"]["openai"]["records"],self.answers)
            self.assertEqual(api.call_count,1)
            with self.assertRaisesRegex(ValueError,"duplicate"):
                r.run("openai",journal,self.panel,d)
            self.assertEqual(api.call_count,1)

    def test_failed_inference_never_retried_and_cash_exhaustion(self):
        j=Memory()
        with mock.patch.object(r,"prepare_openai",return_value=({},500,r.plan(500))), mock.patch.object(r,"oai",side_effect=TimeoutError()) as api:
            with self.assertRaisesRegex(ValueError,"insufficient"):
                r.run("openai",j,self.panel,{**sample(),"grant_remaining_micro_usd":100})
            api.assert_not_called()
            self.assertEqual(r.run("openai",j,self.panel,sample()),"stopped_no_retry")
            with self.assertRaisesRegex(ValueError,"duplicate"):
                r.run("openai",j,self.panel,sample())
            self.assertEqual(api.call_count,1)
            self.assertEqual(j.book["arms"]["openai"]["max_micro_usd"],r.plan(500))

    def test_after_cutoff_closed(self):
        with mock.patch.object(r,"utc",return_value=NOW+timedelta(days=1)), mock.patch.object(r,"prepare_openai") as prep:
            with self.assertRaisesRegex(ValueError,"prospective_window_closed"):
                r.run("openai",Memory(),self.panel,sample())
            prep.assert_not_called()

    def test_two_git_writers_one_reserved_and_visible_after_restart(self):
        with TemporaryDirectory() as wd:
            parent=Path(wd)
            remote=parent/"remote.git"; remote.mkdir()
            origin=parent/"seed"; origin.mkdir()
            r.git(remote,"init","--bare","--quiet")
            r.git(origin,"init","--quiet")
            (origin/"promo-ledger.json").write_text(json.dumps(new_state()))
            r.git(origin,"add","promo-ledger.json")
            r.git(origin,"-c","user.name=Test","-c","user.email=test@example.org","commit","-m","initial")
            r.git(origin,"push",str(remote),"HEAD:refs/heads/"+r.STATE)
            def checkout(name):
                path=parent/name;path.mkdir()
                return r.Journal(path,str(remote))
            a,b=checkout("a"),checkout("b")
            r.lease(a,"openai",sample(),{"single":7500})
            with self.assertRaisesRegex(ValueError,"conflict"):
                r.lease(b,"openai",sample(),{"single":7500})
            recovery=checkout("restarted")
            self.assertEqual(recovery.book["arms"]["openai"]["status"],"reserved")
            with self.assertRaisesRegex(ValueError,"duplicate"):
                r.lease(recovery,"openai",sample(),{"single":7500})
            with self.assertRaisesRegex(ValueError,"prior_run"):
                r.lease(recovery,"gemma",None,{"single":0})


if __name__=="__main__":
    unittest.main()
