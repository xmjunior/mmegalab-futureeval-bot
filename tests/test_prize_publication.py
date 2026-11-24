import asyncio
import io
import logging
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests
from forecasting_tools import BinaryQuestion, BinaryReport, ConditionalQuestion, MetaculusClient

from prize_publication import ProviderAuditHandler, PublicationError, VerifiedPrizeClient, finish_prize_run


class Service:
    """Small in-memory implementation of the public contract. No network fallback."""
    def __init__(self, client):
        self.client = client
        self.posts = []
        self.history = []
        self.comments = []
        self.reject = None
        self.timeout_after_commit = False
        self.hide_comment = False
        self.stored_values = None

    def request(self, method, path, **kwargs):
        if method == "POST":
            assert path == "/questions/bulk-forecast-comment/"
            body = kwargs["json"]
            self.posts.append(body)
            if self.reject:
                return Mock(status_code=self.reject)
            question, = body["forecasts"]
            comment, = body["comments"]
            assert question["question"] == 94 and comment["on_post"] == 12
            assert comment["is_private"] and comment["included_forecast"]
            assert comment["text"] == comment["text"].strip()
            assert not self.history
            if "probability_yes" in question:
                values = [1-question["probability_yes"], question["probability_yes"]]
            elif "continuous_cdf" in question:
                values = question["continuous_cdf"]
            else:
                values = list(question["probability_yes_per_category"].values())
            self.history = [{"question_id":94, "author_id":23, "start_time":1526122000,
                             "forecast_values": self.stored_values or values}]
            self.comments = [{**comment,"id":114,"author":{"id":23}}]
            if self.timeout_after_commit:
                raise requests.Timeout("NO RAW HTTP/PAYLOAD IN LOGS")
            return Mock(status_code=201)
        assert method == "GET"
        if path == "/questions/bulk-forecast-read/":
            assert kwargs["params"] == {"question_ids":94}
            data = {"results":[{"question_id":94,"forecasts":self.history}]}
        else:
            assert path == "/comments/"
            assert kwargs["params"]["post"] == 12 and kwargs["params"]["author"] == 23
            assert kwargs["params"]["is_private"] == "true"
            data = {"results":[] if self.hide_comment else self.comments, "next":None}
        return Mock(status_code=200,json=lambda:data)


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.audit = io.StringIO()
        self.client = VerifiedPrizeClient(self.audit, token="offline-test")
        self.client.owner_id = 23
        self.client.category_order[94] = ["A", "B"]
        self.service = Service(self.client)
        self.client._request = self.service.request
        q = BinaryQuestion.model_construct(id_of_question=94,id_of_post=12)
        self.report = BinaryReport.model_construct(question=q,prediction=0.62,
                                                  explanation="SEALED RATIONALE  \n",errors=[])

    def submit(self):
        asyncio.run(self.report.publish_report_to_metaculus(self.client))

    def test_actual_sdk_binary_report_and_private_round_trip(self):
        self.submit()
        finish_prize_run([self.report],self.client)
        self.assertEqual(len(self.service.posts),1)
        self.assertIn("confirmed=1",self.audit.getvalue())
        self.assertNotIn("SEALED",self.audit.getvalue())
        self.assertNotIn("0.62",self.audit.getvalue())
        self.assertIn('{"probability_yes":0.62}', self.service.comments[0]["text"])
        self.assertIn("forecast_in_comment=true", self.audit.getvalue())

    def test_open_group_cdf_and_categorical_sdk_payload_methods(self):
        for mode in ("numeric","multiple_choice"):
            with self.subTest(mode=mode):
                self.service.history.clear()
                if mode == "numeric":
                    self.client.post_numeric_question_prediction(94,[0.0,0.2,1.0])
                else:
                    self.client.post_multiple_choice_question_prediction(94,{"A":0.3,"B":0.7})
                self.client.post_question_comment(12,"SEALED RATIONALE")
                self.assertIn(94,self.client.verified)

    def test_wrong_categorical_labels_rejected_despite_identical_set_of_masses(self):
        self.service.stored_values=[0.7,0.3]
        self.client.post_multiple_choice_question_prediction(94,{"A":0.3,"B":0.7})
        with self.assertRaisesRegex(PublicationError,"Unexpected persisted"):
            self.client.post_question_comment(12,"SEALED RATIONALE")

    def test_timeout_with_verified_commit_never_resends(self):
        self.service.timeout_after_commit=True
        self.submit()
        self.assertIn("response=interrupted",self.audit.getvalue())
        with self.assertRaisesRegex(PublicationError,"already exists"):
            self.submit()
        self.assertEqual(len(self.service.posts),1)

    def test_rejected_atomic_write_never_succeeds(self):
        self.service.reject=403
        with self.assertRaisesRegex(PublicationError,"403"):
            self.submit()
        self.assertFalse(self.client.verified)
        self.assertFalse(self.service.history)

    def test_timeout_or_5xx_without_commit_never_marks_success(self):
        self.service.reject=503
        with self.assertRaisesRegex(PublicationError,"unconfirmed"):
            self.submit()
        self.assertEqual(len(self.service.posts),1)
        self.assertFalse(self.client.verified)

    def test_201_but_missing_comment_never_reforecasts_or_guesses_rationale(self):
        self.service.hide_comment=True
        with self.assertRaisesRegex(PublicationError,"rationale unconfirmed"):
            self.submit()
        with self.assertRaisesRegex(PublicationError,"already exists"):
            self.submit()
        self.assertEqual(len(self.service.posts),1)
        self.assertFalse(self.client.verified)
        self.assertNotIn("Verified publication",self.audit.getvalue())

    def test_201_but_wrong_forecast_never_marks_success(self):
        self.service.stored_values=[0.1,0.9]
        with self.assertRaisesRegex(PublicationError,"Unexpected persisted"):
            self.submit()
        self.assertFalse(self.client.verified)

    def test_empty_countless_paginated_comment_feed(self):
        self.client._read = Mock(return_value={"results": [], "next": "another-page"})
        self.assertEqual(list(self.client._my_comments(12)), [])
        self.assertEqual(self.client._read.call_count, 1)

    def test_recheck_prior_forecast_without_another_post(self):
        self.submit()
        q=SimpleNamespace(already_forecasted=True,id_of_question=94,id_of_post=12,
                          timestamp_of_my_last_forecast=None)
        with patch.object(MetaculusClient,"get_all_open_questions_from_tournament",return_value=[q]):
            self.client.get_all_open_questions_from_tournament("no-network")
            self.assertFalse(self.client.incomplete_previous)
            self.service.hide_comment=True
            self.client.get_all_open_questions_from_tournament("no-network")
            with self.assertRaisesRegex(PublicationError,"Incomplete"):
                finish_prize_run([],self.client)
        self.assertEqual(len(self.service.posts),1)

    def test_closest_actual_or_scheduled_deadline_first_missing_last(self):
        def q(qid, scheduled=None, actual=None):
            return SimpleNamespace(id_of_question=qid, id_of_post=12, already_forecasted=False,
                api_json={"question":{"scheduled_close_time": scheduled, "actual_close_time": actual}})
        sample = [q(14), q(15,"2026-11-27T01:00:00Z"), q(18,"2026-11-25T02:00:00+01:00"),
                  q(16,"2026-11-27T02:00:00Z","2026-11-25T01:00:00Z"), q(13,"bad date")]
        with patch.object(MetaculusClient, "get_all_open_questions_from_tournament", return_value=sample):
            result = self.client.get_all_open_questions_from_tournament("test")
        self.assertEqual([q.id_of_question for q in result], [16, 18, 15, 13, 14])

    def test_native_attachment_and_group_fallback(self):
        binary = {"probability_yes": 0.62}
        expected = [0.38, 0.62]
        self.assertTrue(self.client.attachment_matches({"included_forecast": binary}, expected, binary))
        self.assertFalse(self.client.attachment_matches(
            {"included_forecast":{"probability_yes":0.2}}, expected, binary))
        self.assertIsNone(self.client.attachment_matches({"included_forecast":None}, expected, binary))
        cdf = {"continuous_cdf":[0.0, 0.5, 1.0]}
        self.assertTrue(self.client.attachment_matches({"included_forecast":cdf}, cdf["continuous_cdf"], cdf))

    def test_conditional_result_requires_both_published_branch_ids(self):
        yes = BinaryQuestion.model_construct(id_of_question=94, id_of_post=12)
        no = BinaryQuestion.model_construct(id_of_question=95, id_of_post=12)
        question = ConditionalQuestion.model_construct(id_of_question=1200, id_of_post=12,
                                                       question_yes=yes, question_no=no)
        report = SimpleNamespace(question=question, errors=[])
        self.client.verified = {94, 95}
        finish_prize_run([report], self.client)
        self.client.verified = {94}
        with self.assertRaises(PublicationError):
            finish_prize_run([report], self.client)

    def test_only_safe_provider_events_pass_public_handler(self):
        log = logging.Logger("test-exclusive")
        log.addHandler(ProviderAuditHandler(self.audit))
        model = "gemini/gemma-4-31b-it"
        unavailable = "Free route %s unavailable (%s), cooldown %ss"
        log.warning(unavailable, model, 503, 12)
        log.warning(unavailable, model, 429, 3600)
        log.info("Model responded: %s", model)
        log.warning(unavailable, model, "SEALED", 1)
        log.warning(unavailable, model, 429, "SEALED")
        log.warning(unavailable + " SEALED", model, 429, 1)
        log.info("Model responded: %s", "SEALED")
        out = self.audit.getvalue()
        self.assertIn("status=503", out)
        self.assertIn("status=429", out)
        self.assertIn("responded model=" + model, out)
        self.assertNotIn("SEALED", out)

    def test_only_actual_routes_are_bound_to_private_question_note(self):
        log = logging.Logger("test-model-note")
        log.addHandler(ProviderAuditHandler(self.audit))
        async def child():
            log.info("Model responded: %s", "gemini/gemma-4-31b-it")
        async def question():
            with self.client.track_question_routes():
                await asyncio.create_task(child())
                await self.report.publish_report_to_metaculus(self.client)
        asyncio.run(question())
        note = self.service.comments[0]["text"]
        self.assertIn("Hosted routes used for this question: gemini/gemma-4-31b-it", note)
        self.assertNotIn("26b", note)
        self.assertNotIn("SEALED", self.audit.getvalue())

    def test_legacy_comment_shapes_and_timestamp_precision(self):
        self.submit()
        when = datetime(2026, 1, 1, 12, 10, 10, 350000, tzinfo=timezone.utc)
        q = SimpleNamespace(already_forecasted=True, id_of_question=94, id_of_post=12,
                            timestamp_of_my_last_forecast=when)
        c = self.service.comments[0]
        c["text"] = "Old private rationale"
        with patch.object(MetaculusClient, "get_all_open_questions_from_tournament", return_value=[q]):
            for attachment in ({"start_time": when.timestamp()}, {"start_time": "2026-01-01T12:10:10Z"}):
                c["included_forecast"] = attachment
                self.client.get_all_open_questions_from_tournament("no-network")
                self.assertFalse(self.client.unverified_legacy)
            for attachment in (True, 52, None):
                c["included_forecast"] = attachment
                self.client.get_all_open_questions_from_tournament("no-network")
                self.assertEqual(self.client.unverified_legacy, {94})
                self.assertFalse(self.client.incomplete_previous)
        self.assertEqual(len(self.service.posts), 1)

    def test_missing_private_comment_with_other_author_same_text(self):
        self.submit()
        self.service.comments[0]["author"]={"id":42}
        self.assertEqual(list(self.client._my_comments(12)),[])

    def test_empty_cycle_vs_exceptions(self):
        finish_prize_run([],self.client)
        with self.assertRaises(PublicationError):
            finish_prize_run([ValueError("SEALED ANSWER")],self.client)
        self.assertNotIn("SEALED",self.audit.getvalue())

    def test_diagnostics_and_sdk_logs_do_not_escape_to_public_output(self):
        code = """
import logging, os, sys
from prize_publication import open_audit_stream
logging.basicConfig(level=logging.INFO)
stream=open_audit_stream()
logging.critical("SEALED_SDK_PAYLOAD")
print("SEALED_STDOUT", flush=True)
os.write(2,b"SEALED_NATIVE_STDERR")
print("audit-metadata-only",file=stream)
raise ValueError("SEALED_EXCEPTION")
"""
        root=Path(__file__).resolve().parents[1]
        result=subprocess.run([sys.executable,"-c",code],cwd=root,text=True,capture_output=True)
        self.assertTrue(result.returncode)
        self.assertIn("audit-metadata-only",result.stdout)
        self.assertIn("ValueError",result.stdout)
        self.assertNotIn("SEALED",result.stdout+result.stderr)
