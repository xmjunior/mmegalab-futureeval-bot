import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import main


class ProviderError(Exception):
    def __init__(self, status, retry_after, message="temporary"):
        super().__init__(message)
        self.status_code = status
        self.response = SimpleNamespace(headers={"retry-after":retry_after})


class RetryWindows(unittest.IsolatedAsyncioTestCase):
    async def call(self, specs):
        clock = [100.0]
        routes = [SimpleNamespace(model=m, invoke=AsyncMock(side_effect=outcomes)) for m,outcomes in specs]
        state = SimpleNamespace(cpu=None, lock=asyncio.Lock(), routes=routes, allow_cpu_fallback=False,
                                available_at={r.model:0 for r in routes})
        async def sleep(seconds):
            self.assertGreater(seconds, 0)
            clock[0] += seconds
        self.routes = routes
        with patch.object(main, "time", SimpleNamespace(monotonic=lambda:clock[0])), patch.object(main.asyncio,"sleep",sleep):
            return await main.FreeForecastLlm.invoke(state,"synthetic transport test")

    async def test_later_route_still_gets_its_second_turn(self):
        a = ProviderError(503, "45")
        b = ProviderError(503, "75")
        answer = await self.call([("gemini/a",[a,a]),("gemini/b",[b,"ok"]),
                                  ("openrouter/free",[ProviderError(429,"","daily limit")])])
        self.assertEqual(answer, "ok")
        self.assertEqual([r.invoke.await_count for r in self.routes], [2,2,1])

    async def test_daily_quota_or_long_retry_does_not_sleep_past_deadline(self):
        with self.assertRaisesRegex(RuntimeError, "refusing degraded"):
            await self.call([("gemini/a",[ProviderError(503,"240")]),
                             ("openrouter/free",[ProviderError(429,"","daily quota")])])
        self.assertEqual([r.invoke.await_count for r in self.routes], [1,1])

    async def test_empty_response_can_use_another_hosted_route(self):
        self.assertEqual(await self.call([("gemini/a",[""]),("gemini/b",["ok"])]), "ok")
