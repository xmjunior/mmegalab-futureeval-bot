"""Groq fallback for a verified Free organization. Never follow redirects or upgrade."""
import asyncio
import json
import os
import re

import requests

BASE = "https://api.groq.com/openai/v1"
MODEL = "openai/gpt-oss-120b"
ROUTE = "groq/" + MODEL


class Unavailable(RuntimeError):
    """Carry a status and retry hint, without credentials, prompts or provider text."""

    def __init__(self, status, retry_after=None, daily=False):
        super().__init__(f"Groq unavailable ({status})")
        self.status_code = status
        self.daily = daily
        self.response = requests.Response()
        if retry_after:
            self.response.headers["retry-after"] = retry_after


DAILY_QUOTA = re.compile(r"\b(?:daily|per[ -]?day|day(?:ly)?\s+limit|requests?\s+per\s+day|rpd)\b",
                         re.IGNORECASE)


def daily_quota(response):
    """Boolean daily-cap signal; the provider body itself is never propagated."""
    if response.status_code != 429:
        return False
    try:
        error = response.json().get("error", {})
    except (ValueError, TypeError, AttributeError):
        return False
    if isinstance(error, dict):
        error = error.get("message", "")
    return bool(DAILY_QUOTA.search(str(error)))


def configured():
    return (os.environ.get("GROQ_FREE_ACCOUNT_CONFIRMED") == "true"
            and os.environ.get("GROQ_API_KEY", "").strip() not in
            {"", "REPLACE_ME", "1234567890", "your-api-key-here", "your-token-here"})


def plan_denial(response):
    # Developer and Enterprise support Batch. Reject a generic permissions 403
    # as it can also occur on a paid project with a restricted key.
    if response.status_code != 403:
        return False
    try:
        error = response.json()["error"]
        if not isinstance(error, dict):
            return False
    except (ValueError, KeyError, TypeError):
        return False
    if error.get("code") in ("free_tier_not_supported", "free_tier_not_allowed"):
        return True
    message = str(error.get("message", "")).lower()
    return bool(re.search(
        r"\b(?:not available|not supported|unavailable) (?:for|on) "
        r"(?:your|the) (?:(?:current|free) )?(?:plan|tier)\b", message))


def _request(method, path, key, **kwargs):
    try:
        return requests.request(method, BASE + path, headers={"Authorization": "Bearer " + key},
                                allow_redirects=False, timeout=(5, 40), **kwargs)
    except requests.Timeout:
        raise TimeoutError("Groq request timed out") from None
    except requests.RequestException:
        raise ConnectionError("Groq connection failed") from None


class GroqFreeRoute:
    model = ROUTE

    def complete(self, prompt, system_prompt=None, max_tokens=2048):
        if not configured():
            raise Unavailable(403)
        key = os.environ["GROQ_API_KEY"]
        # The owner's Free declaration and this same-key current plan check must
        # both pass. An upgrade, an ambiguous denial or any outage stops inference.
        plan = _request("GET", "/batches?limit=1", key)
        if not plan_denial(plan):
            status = plan.status_code if plan.status_code in (408, 429, 500, 502, 503, 504) else 403
            raise Unavailable(status, plan.headers.get("retry-after"), daily_quota(plan))
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = _request("POST", "/chat/completions", key, json={
            "model": MODEL, "messages": messages, "service_tier": "on_demand",
            "reasoning_effort": "low", "max_completion_tokens": max_tokens,
            "tools": [], "tool_choice": "none", "stream": False,
        })
        if response.status_code != 200:
            # Isolate unexpected statuses to this provider. Preserve transport,
            # quota, credential and request-size signals the caller handles.
            status = response.status_code
            if status not in (400, 401, 402, 403, 404, 408, 413, 422, 429, 500, 502, 503, 504):
                status = 502
            raise Unavailable(status, response.headers.get("retry-after"), daily_quota(response))
        try:
            result = response.json()
            choice = result["choices"][0]
            answer = choice["message"]["content"]
            usage = result["usage"]
            tokens = {k: usage[k] for k in ("prompt_tokens", "completion_tokens")}
            valid = (result["model"] == MODEL and choice["finish_reason"] == "stop"
                     and not choice["message"].get("tool_calls")
                     and isinstance(answer, str) and answer.strip()
                     and all(type(n) is int and n >= 0 for n in tokens.values()))
        except (KeyError, IndexError, ValueError, TypeError, AttributeError):
            valid = False
        if not valid:
            raise Unavailable(502)
        return answer, tokens

    async def invoke(self, prompt, system_prompt=None):
        answer, _ = await asyncio.to_thread(self.complete, prompt, system_prompt)
        return answer


def smoke():
    try:
        answer, usage = GroqFreeRoute().complete(
            'Reply with the single integer that equals 13 plus 28. No other text.', max_tokens=256)
        if answer.strip() != "41":
            raise ValueError("synthetic_answer_mismatch")
    except (Unavailable, TimeoutError, ConnectionError, ValueError) as exc:
        print(json.dumps({"synthetic_probe": "failed", "status": getattr(exc, "status_code", type(exc).__name__)}))
        return 1
    print(json.dumps({"synthetic_probe": "passed", "model": MODEL, "usage": usage,
                      "billing_evidence": "owner_free_and_current_batch_plan_denial",
                      "competition_submissions": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(smoke())
