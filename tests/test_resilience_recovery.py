"""Recovery probes for a total free-provider outage.

Every route is an AsyncMock and the monotonic clock is synthetic, so these
tests make no network calls. Each test states the behaviour the bot needs in
order to keep forecasting after all free providers stop responding; tests
marked ``expectedFailure`` document a defect that is currently present and
will report "unexpected success" once the defect is fixed.
"""
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import main

ROUTES = ["gemini/gemma-4-31b-it", "gemini/gemma-4-26b-a4b-it", "openrouter/openrouter/free"]


class ProviderError(Exception):
    def __init__(self, status, retry_after="", message="temporary"):
        super().__init__(message)
        self.status_code = status
        self.response = SimpleNamespace(headers={"retry-after": retry_after})


class TotalOutage(unittest.IsolatedAsyncioTestCase):
    def state(self, cpu_fallback=False):
        return SimpleNamespace(
            cpu=None,
            lock=asyncio.Lock(),
            routes=[SimpleNamespace(model=name, invoke=AsyncMock()) for name in ROUTES],
            allow_cpu_fallback=cpu_fallback,
            available_at={name: 0.0 for name in ROUTES},
        )

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

    @unittest.expectedFailure
    async def test_daily_quota_on_every_route_is_not_retried_as_transient(self):
        state = self.state()
        for route in state.routes:
            route.invoke.side_effect = ProviderError(429, "", "daily quota exceeded")
        clock = [100.0]
        with self.assertRaisesRegex(RuntimeError, "refusing degraded"):
            await self.run_invoke(state, clock)
        self.assertEqual([r.invoke.await_count for r in state.routes], [1, 1, 1])
        self.assertTrue(all(state.available_at[name] - 100.0 >= 3600 for name in ROUTES),
                        state.available_at)

    @unittest.expectedFailure
    async def test_hosted_routes_are_probed_again_after_cpu_fallback(self):
        class FakeCpu:
            def invoke(self, prompt, system_prompt=None):
                return "cpu-local"

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

            state.routes[0].invoke.side_effect = ["hosted-recovered"]
            state.routes[1].invoke.side_effect = ["unexpected"]
            state.routes[2].invoke.side_effect = ["unexpected"]
            clock[0] += 3600
            answer, _ = await self.run_invoke(state, clock)
        self.assertEqual(answer, "hosted-recovered")
        self.assertEqual(state.routes[0].invoke.await_count, 3)
