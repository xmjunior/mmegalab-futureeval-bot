"""Recovery probes for a total free-provider outage.

Every route is an AsyncMock and the monotonic clock is synthetic, so these
tests make no network calls. Each test states the behaviour the bot needs in
order to keep forecasting after all free providers stop responding.
"""
import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import requests

import groq_free
import main

ROUTES = ["gemini/gemma-4-31b-it", "gemini/gemma-4-26b-a4b-it", "openrouter/openrouter/free"]


class ProviderError(Exception):
    def __init__(self, status, retry_after="", message="temporary"):
        super().__init__(message)
        self.status_code = status
        self.response = SimpleNamespace(headers={"retry-after": retry_after})


class FakeCpu:
    calls = []

    def invoke(self, prompt, system_prompt=None):
        return "cpu-local"

    def choose_integers(self, prompt, count, low, high):
        type(self).calls.append(count)
        return [low] * count


class TotalOutage(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        FakeCpu.calls = []

    def state(self, cpu_fallback=False):
        state = SimpleNamespace(
            cpu=None,
            cpu_retry_at=0.0,
            lock=asyncio.Lock(),
            routes=[SimpleNamespace(model=name, invoke=AsyncMock()) for name in ROUTES],
            allow_cpu_fallback=cpu_fallback,
            available_at={name: 0.0 for name in ROUTES},
        )
        # choose_integers dispatches through self.invoke, so bind it on the stub.
        state.invoke = main.FreeForecastLlm.invoke.__get__(state, type(state))
        return state

    async def run_invoke(self, state, clock):
        slept = []

        async def sleep(seconds):
            self.assertGreater(seconds, 0)
            slept.append(seconds)
            clock[0] += seconds

        with patch.object(main, "time", SimpleNamespace(monotonic=lambda: clock[0])), \
                patch.object(main.asyncio, "sleep", sleep):
            answer = await main.FreeForecastLlm.invoke(state, "synthetic total outage")
        return answer, slept

    async def test_simultaneous_outage_recovers_once_cooldowns_expire(self):
        state = self.state()
        for route in state.routes:
            route.invoke.side_effect = [ProviderError(503, "45"), ProviderError(503, "45")]
        clock = [100.0]
        with self.assertRaisesRegex(RuntimeError, "refusing degraded"):
            await self.run_invoke(state, clock)
        self.assertIsNone(state.cpu)
        self.assertEqual([r.invoke.await_count for r in state.routes], [2, 2, 2])
        self.assertEqual({state.available_at[name] for name in ROUTES}, {190.0})

        state.routes[0].invoke.side_effect = ["recovered"]
        state.routes[1].invoke.side_effect = ["unexpected"]
        state.routes[2].invoke.side_effect = ["unexpected"]
        clock[0] = 150.0
        answer, slept = await self.run_invoke(state, clock)
        self.assertEqual(answer, "recovered")
        self.assertEqual(slept, [40.0])
        self.assertEqual([r.invoke.await_count for r in state.routes], [3, 2, 2])

    async def test_daily_quota_on_every_route_is_not_retried_as_transient(self):
        state = self.state()
        for route in state.routes:
            route.invoke.side_effect = ProviderError(429, "", "daily quota exceeded")
        clock = [100.0]
        with self.assertRaisesRegex(RuntimeError, "refusing degraded"):
            await self.run_invoke(state, clock)
        self.assertEqual([r.invoke.await_count for r in state.routes], [1, 1, 1])
        self.assertEqual({state.available_at[name] for name in ROUTES}, {3700.0})

    async def test_hosted_routes_are_probed_again_after_cpu_fallback(self):
        state = self.state(cpu_fallback=True)
        for route in state.routes:
            route.invoke.side_effect = ConnectionError("provider down")
        clock = [100.0]
        with patch.object(main, "platform",
                          SimpleNamespace(system=lambda: "Linux", machine=lambda: "x86_64")), \
                patch.object(main, "CpuForecaster", FakeCpu):
            answer, _ = await self.run_invoke(state, clock)
            self.assertEqual(answer, "cpu-local")
            self.assertIsNotNone(state.cpu)
            self.assertEqual([r.invoke.await_count for r in state.routes], [2, 2, 2])
            self.assertEqual(state.cpu_retry_at, 136.0)

            state.routes[0].invoke.side_effect = ["hosted-recovered"]
            state.routes[1].invoke.side_effect = ["unexpected"]
            state.routes[2].invoke.side_effect = ["unexpected"]
            clock[0] += 3600
            answer, slept = await self.run_invoke(state, clock)
        self.assertEqual(answer, "hosted-recovered")
        self.assertEqual(slept, [])
        self.assertEqual(state.routes[0].invoke.await_count, 3)
        self.assertEqual(state.cpu_retry_at, 0.0)

    async def test_transient_429_keeps_short_retry_after_and_second_turn(self):
        state = self.state()
        for route in state.routes:
            route.invoke.side_effect = ProviderError(429, "20", "rate limit exceeded, retry in 20s")
        clock = [100.0]
        with self.assertRaisesRegex(RuntimeError, "refusing degraded"):
            await self.run_invoke(state, clock)
        self.assertEqual([r.invoke.await_count for r in state.routes], [2, 2, 2])
        self.assertEqual({state.available_at[name] for name in ROUTES}, {144.0})

    async def test_daily_quota_respects_longer_retry_after(self):
        for hint, expected in (("7200", 7300.0), ("300", 400.0)):
            state = self.state()
            for route in state.routes:
                route.invoke.side_effect = ProviderError(429, hint, "daily quota exceeded")
            clock = [100.0]
            with self.assertRaisesRegex(RuntimeError, "refusing degraded"):
                await self.run_invoke(state, clock)
            self.assertEqual([r.invoke.await_count for r in state.routes], [1, 1, 1])
            self.assertEqual({state.available_at[name] for name in ROUTES}, {expected})

    async def test_choose_integers_returns_to_hosted_after_cpu_fallback(self):
        state = self.state(cpu_fallback=True)
        for route in state.routes:
            route.invoke.side_effect = ConnectionError("provider down")
        clock = [100.0]

        async def sleep(seconds):
            clock[0] += seconds

        with patch.object(main, "time", SimpleNamespace(monotonic=lambda: clock[0])), \
                patch.object(main.asyncio, "sleep", sleep), \
                patch.object(main, "platform",
                             SimpleNamespace(system=lambda: "Linux", machine=lambda: "x86_64")), \
                patch.object(main, "CpuForecaster", FakeCpu):
            self.assertEqual(await main.FreeForecastLlm.invoke(state, "probe"), "cpu-local")
            clock[0] += 3600
            state.routes[0].invoke.side_effect = [json.dumps([7, 9])]
            state.routes[1].invoke.side_effect = ["unexpected"]
            state.routes[2].invoke.side_effect = ["unexpected"]
            values = await main.FreeForecastLlm.choose_integers(state, "synthetic", 2, 0, 10)
        self.assertEqual(values, [7, 9])
        self.assertEqual(FakeCpu.calls, [])
        self.assertEqual(state.routes[0].invoke.await_count, 3)

    async def test_groq_daily_cap_is_a_boolean_signal_without_provider_text(self):
        def reply(status, body):
            response = requests.Response()
            response.status_code = status
            response._content = json.dumps(body).encode()
            return response

        self.assertTrue(groq_free.daily_quota(reply(429, {"error": {
            "message": "Rate limit reached for requests per day (RPD)"}})))
        self.assertFalse(groq_free.daily_quota(reply(429, {"error": {
            "message": "Rate limit exceeded, retry in 20s"}})))
        self.assertFalse(groq_free.daily_quota(reply(503, {"error": {"message": "daily"}})))
        exc = groq_free.Unavailable(429, "3600", daily=True)
        self.assertNotIn("Rate limit", str(exc))

        state = self.state()
        for route in state.routes:
            route.invoke.side_effect = exc
        clock = [100.0]
        with self.assertRaisesRegex(RuntimeError, "refusing degraded"):
            await self.run_invoke(state, clock)
        self.assertEqual([r.invoke.await_count for r in state.routes], [1, 1, 1])
        self.assertEqual({state.available_at[name] for name in ROUTES}, {3700.0})
