import asyncio
import io
import json
import logging
import os
import unittest
from unittest.mock import AsyncMock, patch

import requests

import bot_helpers
import groq_free
import main
from prize_publication import ProviderAuditHandler


def response(status, body, headers=None):
    reply = requests.Response()
    reply.status_code = status
    reply._content = json.dumps(body).encode()
    reply.headers.update(headers or {})
    return reply


def denial(message):
    return response(403, {"error": {"message": message}})


def completion(content="41", **changes):
    data = {
        "model": groq_free.MODEL,
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 39, "completion_tokens": 18},
    }
    return response(200, {**data, **changes})


class GroqFreeTests(unittest.TestCase):
    def setUp(self):
        ctx = patch.dict(os.environ, {"GROQ_FREE_ACCOUNT_CONFIRMED": "true",
                                      "GROQ_API_KEY": "synthetic-key", "METACULUS_TOKEN": "test"}, clear=True)
        ctx.start()
        self.addCleanup(ctx.stop)
        self.route = groq_free.GroqFreeRoute()

    def test_ambiguous_denial_upgrade_auth_failure_and_redirect_make_no_inference_call(self):
        for reply in (response(200, {"data": []}), denial("Denied by project permissions"),
                      response(403, {"error": []}), response(401, {}), response(302, {})):
            with self.subTest(status=reply.status_code), patch.object(groq_free, "_request", return_value=reply) as http:
                with self.assertRaises(groq_free.Unavailable):
                    self.route.complete("synthetic request")
                self.assertEqual(http.call_count, 1)
                self.assertEqual(http.call_args.args[0], "GET")

    def test_missing_owner_confirmation_or_key_blocks_before_network(self):
        for field, value in (("GROQ_FREE_ACCOUNT_CONFIRMED", ""), ("GROQ_API_KEY", "REPLACE_ME")):
            with patch.dict(os.environ, {field: value}), patch.object(groq_free, "_request") as http:
                self.assertFalse(groq_free.configured())
                with self.assertRaises(groq_free.Unavailable):
                    self.route.complete("synthetic")
                http.assert_not_called()

    def test_truncation_tool_calls_wrong_model_and_empty_responses_rejected(self):
        invalid = [completion(" "), completion(model="another/model"),
                   completion(choices=[{"message": {"content": "41"}, "finish_reason": "length"}]),
                   completion(choices=[{"message": {"content": "41", "tool_calls": [{}]}, "finish_reason": "stop"}]),
                   completion(usage={})]
        for reply in invalid:
            with self.subTest(reply=reply.json()), patch.object(
                    groq_free, "_request", side_effect=[denial("Not available for your plan"), reply]):
                with self.assertRaises(groq_free.Unavailable):
                    self.route.complete("synthetic")

    def test_single_key_gates_inference_and_only_model_output_is_returned(self):
        with patch.object(groq_free, "_request", side_effect=[
                denial("Batch API is not available for your current plan."), completion()]) as http:
            answer = asyncio.run(self.route.invoke("synthetic", "synthetic system"))
            self.assertEqual(answer, "41")
            self.assertEqual([c.args[:2] for c in http.call_args_list],
                             [("GET", "/batches?limit=1"), ("POST", "/chat/completions")])
            self.assertEqual(http.call_args_list[0].args[2], http.call_args_list[1].args[2])
            self.assertEqual(http.call_args.kwargs["json"]["tool_choice"], "none")

    def test_provider_switch_after_both_gemmas_and_openrouter_fail(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake", "GEMINI_API_KEY": "fake"}):
            llm = main.FreeForecastLlm()
        for route in llm.routes[:-1]:
            route.invoke = AsyncMock(side_effect=groq_free.Unavailable(429, "3600"))
        with patch.object(groq_free, "_request", side_effect=[
                denial("Not available for your plan"), completion()]), self.assertLogs(main.logger) as logs:
            self.assertEqual(asyncio.run(llm.invoke("offline synthetic")), "41")
            self.assertTrue(any(groq_free.ROUTE in line for line in logs.output))
        self.assertEqual([r.invoke.await_count for r in llm.routes[:-1]], [1, 1, 1])
        self.assertIsNone(llm.cpu)

    def test_groq_only_configuration_works_and_collapsible_audit_is_allowlisted(self):
        bot_helpers.check_environment()
        self.assertEqual([r.model for r in main.FreeForecastLlm().routes], [groq_free.ROUTE])
        sink = io.StringIO()
        handler = ProviderAuditHandler(sink)
        handler.handle(logging.LogRecord("x", 20, "", 0, "Model responded: %s",
                                         (groq_free.ROUTE,), None))
        self.assertIn(groq_free.ROUTE, sink.getvalue())

    def test_rate_limit_preserves_retry_hint(self):
        with patch.object(groq_free, "_request", side_effect=[
                denial("Not available for your plan"), response(429, {}, {"retry-after": "35"})]):
            with self.assertRaises(groq_free.Unavailable) as exc:
                self.route.complete("synthetic")
        self.assertEqual(exc.exception.response.headers["retry-after"], "35")

    def test_http_does_not_follow_redirects_or_expose_provider_body(self):
        with patch.object(requests, "request", return_value=response(302, {})) as req:
            self.assertEqual(groq_free._request("GET", "/batches", "test").status_code, 302)
            self.assertFalse(req.call_args.kwargs["allow_redirects"])
