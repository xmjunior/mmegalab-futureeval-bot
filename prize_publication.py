"""Verify prize submissions without writing sealed forecasts to Actions logs."""
import contextvars
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
import sys
import time

import requests
from forecasting_tools import MetaculusClient


def open_audit_stream():
    """SDKs can print raw prompts to pre-bound handlers or native file descriptors."""
    sys.stdout.flush()
    sys.stderr.flush()
    stream = os.fdopen(os.dup(1), "w", buffering=1)
    with open(os.devnull, "w") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)

    def safe_exception(kind, value, traceback):
        print(f"Production run failed: {kind.__name__}", file=stream)

    sys.excepthook = safe_exception
    return stream


class ProviderAuditHandler(logging.Handler):
    """Emit only known provider events, with validated scalar arguments."""
    MODELS = frozenset(("gemini/gemma-4-31b-it", "gemini/gemma-4-26b-a4b-it", "openrouter/openrouter/free"))
    CONNECTION_ERRORS = frozenset(("Timeout", "TimeoutError", "ConnectionError", "APIConnectionError"))

    def __init__(self, output):
        super().__init__()
        self.output = output

    def emit(self, record):
        args = record.args
        if not isinstance(args, tuple) or not args or type(args[0]) is not str or args[0] not in self.MODELS:
            return
        if record.msg == "Model responded: %s" and len(args) == 1:
            print(f"Hosted model responded model={args[0]}", file=self.output)
        elif record.msg == "Free route %s unavailable (%s), cooldown %ss" and len(args) == 3:
            model, status, cooldown = args
            valid_status = (type(status) is int and 100 <= status <= 599
                            or type(status) is str and status in self.CONNECTION_ERRORS)
            if valid_status and type(cooldown) is int and cooldown >= 0:
                print(f"Hosted model unavailable model={model} status={status} cooldown_s={cooldown}", file=self.output)


class PublicationError(RuntimeError):
    pass


class VerifiedPrizeClient(MetaculusClient):
    """Use the SDK's validated payloads, submitting each pair exactly once."""

    def __init__(self, audit_stream, **kwargs):
        super().__init__(**kwargs)
        self.audit = audit_stream
        self.pending = contextvars.ContextVar("pending_forecast", default=None)
        self.category_order = {}
        self.verified = set()
        self.incomplete_previous = set()
        self.unverified_legacy = set()
        self.owner_id = None

    def get_current_user_id(self):
        if self.owner_id is None:
            self.owner_id = super().get_current_user_id()
        return self.owner_id

    def _post_question_prediction(self, question_id, forecast_payload):
        # The three SDK report classes call this followed by post_question_comment
        # in the same asyncio task. Delay both writes until we have the rationale.
        if self.pending.get() is not None:
            raise PublicationError("Unpaired SDK forecast")
        self.pending.set((question_id, forecast_payload))

    def _request(self, method, path, **kwargs):
        self._sleep_between_requests()
        return requests.request(method, self.base_url + path, timeout=self.timeout,
                                allow_redirects=False, **self._get_auth_headers(), **kwargs)

    def _read(self, path, params):
        for attempt in range(3):
            try:
                response = self._request("GET", path, params=params)
            except requests.RequestException:
                response = None
            if response is not None:
                if response.status_code == 200:
                    return response.json()
                if response.status_code not in (408, 429, 500, 502, 503, 504):
                    raise PublicationError(f"Read-back HTTP {response.status_code}")
            if attempt < 2:
                time.sleep(2 ** attempt)
        status = response.status_code if response is not None else "interrupted"
        raise PublicationError(f"Read-back unavailable: {status}")

    def _my_forecasts(self, question_id):
        result = self._read("/questions/bulk-forecast-read/", {"question_ids": question_id})
        rows = result["results"]
        if len(rows) != 1 or rows[0]["question_id"] != question_id:
            raise PublicationError("Question missing from authorized read-back")
        forecasts = rows[0]["forecasts"]
        if not isinstance(forecasts, list) or any(
            f.get("author_id") != self.owner_id or f.get("question_id") != question_id for f in forecasts
        ):
            raise PublicationError("Unexpected forecast owner or question")
        return forecasts

    def _my_comments(self, post_id):
        # Filter by the token owner's author id, privacy and POST id. Only ever
        # send credentials to our configured API, even when a page advertises next.
        offset = 0
        while True:
            page = self._read("/comments/", {"post": post_id, "author": self.owner_id,
                                              "is_private": "true", "limit": 100, "offset": offset})
            comments = page["results"]
            for c in comments:
                if (c.get("on_post") == post_id and c.get("author", {}).get("id") == self.owner_id
                        and c.get("is_private") is True and not c.get("is_soft_deleted")):
                    yield c
            # Metaculus CountlessLimitOffsetPagination uses infinite count, so
            # even the final empty page can advertise a next link.
            if not comments or not page.get("next"):
                return
            offset += len(comments)

    def get_all_open_questions_from_tournament(self, *args, **kwargs):
        questions = super().get_all_open_questions_from_tournament(*args, **kwargs)
        by_post = {}
        for q in questions:
            if getattr(q, "options", None):
                raw = (q.api_json or {}).get("question", {})
                self.category_order[q.id_of_question] = raw.get("all_options_ever") or q.options
            if not q.already_forecasted:
                print(f"Unanswered question={q.id_of_question} post={q.id_of_post}", file=self.audit)
                continue
            post_id = q.id_of_post
            if post_id not in by_post:
                by_post[post_id] = list(self._my_comments(post_id))
            marker = f"Forecast receipt {q.id_of_question}/"
            last = q.timestamp_of_my_last_forecast
            legacy = False
            for c in by_post[post_id]:
                text = c.get("text") or ""
                if text.startswith(marker):
                    break
                if not text.strip() or text.startswith("Forecast receipt "):
                    continue
                # Historical notes from the two-step SDK carry no question marker.
                # Some deployments omit the attached object or date precision.
                legacy = True
                if getattr(q, "question_ids_of_group", None):
                    continue
                attached = c.get("included_forecast")
                if isinstance(attached, dict) and self._same_forecast_time(attached.get("start_time"), last):
                    break
            else:
                if legacy:
                    self.unverified_legacy.add(q.id_of_question)
                    print(f"Legacy private note present; linkage unverified: "
                          f"question={q.id_of_question} post={post_id}", file=self.audit)
                else:
                    self.incomplete_previous.add(q.id_of_question)
                    print(f"Existing forecast has no verified private rationale: "
                          f"question={q.id_of_question} post={post_id}", file=self.audit)
        return questions

    @staticmethod
    def _same_forecast_time(start, last):
        if start is None or last is None or isinstance(start, bool):
            return False
        try:
            if isinstance(start, str):
                try:
                    start = float(start)
                except ValueError:
                    start = datetime.fromisoformat(start)
            if isinstance(start, datetime):
                if start.tzinfo is None:
                    start = start.replace(tzinfo=timezone.utc)
                start = start.timestamp()
            # Older serializers truncated ISO strings to whole seconds.
            return abs(float(start) - last.timestamp()) < 1
        except (TypeError, ValueError, OverflowError):
            return False

    def post_question_comment(self, post_id, comment_text, is_private=True, included_forecast=True):
        pending = self.pending.get()
        self.pending.set(None)
        if pending is None or not is_private or not included_forecast or not comment_text.strip():
            raise PublicationError("Incomplete private forecast/comment pair")
        question_id, payload = pending
        # The API serializes comment text with Django CharField trim_whitespace.
        comment_text = comment_text.strip()
        try:
            self._commit_pair(question_id, post_id, payload, comment_text)
        except PublicationError as exc:
            # Only fixed messages produced in this module; never server text.
            print(f"Publication unconfirmed question={question_id} post={post_id}: {exc}", file=self.audit)
            raise

    def _commit_pair(self, question_id, post_id, payload, comment_text):
        self.get_current_user_id()
        if self._my_forecasts(question_id):
            # The ordinary SDK skip usually handles this before model inference.
            # A duplicate detected at the last moment must never become an update.
            raise PublicationError("Forecast already exists; no write sent")
        signature = hashlib.sha256(
            json.dumps([question_id, payload, comment_text], sort_keys=True,
                       ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()[:24]
        # Marker binds group subquestions to the exact payload and original reasoning.
        text = f"Forecast receipt {question_id}/{signature}\n\n{comment_text}"
        body = {"forecasts": [{"question": question_id, **payload}],
                "comments": [{"on_post": post_id, "text": text, "is_private": True,
                              "included_forecast": True}]}
        try:
            response = self._request("POST", "/questions/bulk-forecast-comment/", json=body)
        except requests.RequestException:
            status = None  # An interrupted connection can still have committed.
        else:
            status = response.status_code
        print(f"Atomic response question={question_id} http={status or 'interrupted'}", file=self.audit)
        if status is not None and 400 <= status < 500:
            raise PublicationError(f"Atomic submission rejected: HTTP {status}")
        if status not in (None, 201) and status < 500:
            raise PublicationError(f"Unexpected submission HTTP {status}")
        # Never retry a write on an uncertain response. The server commits both
        # rows atomically. Confirm ownership and actual stored records using reads.
        forecasts = self._my_forecasts(question_id)
        if len(forecasts) != 1 or not forecasts[0].get("start_time"):
            raise PublicationError("Single persisted forecast unconfirmed")
        values = forecasts[0].get("forecast_values") or []
        if not values or not any(v is not None for v in values):
            raise PublicationError("Persisted forecast has no values")
        if not all(v is None or (math.isfinite(v) and 0 <= v <= 1) for v in values):
            raise PublicationError("Persisted forecast values invalid")
        if "probability_yes" in payload:
            expected = [1 - payload["probability_yes"], payload["probability_yes"]]
        elif "continuous_cdf" in payload:
            expected = payload["continuous_cdf"]
        else:
            # API vectors follow option history, including withdrawn options.
            probabilities = payload["probability_yes_per_category"]
            order = self.category_order.get(question_id)
            if not order or not set(probabilities).issubset(order):
                raise PublicationError("Category order could not be verified")
            expected = [probabilities.get(label) for label in order]
        if len(values) != len(expected) or any(
                (v is None) != (e is None)
                or (v is not None and not math.isclose(v, e, rel_tol=1e-7, abs_tol=1e-9))
                for v, e in zip(values, expected)):
            raise PublicationError("Unexpected persisted forecast")
        comment_id = next((int(c["id"]) for c in self._my_comments(post_id)
                           if c.get("text") == text), None)
        if not comment_id:
            raise PublicationError("Matching private rationale unconfirmed")
        self.verified.add(question_id)
        print(f"Verified publication question={question_id} post={post_id} comment={comment_id} "
              f"response={status or 'interrupted'} stored_values={len(values)} private=true", file=self.audit)


def finish_prize_run(reports, client):
    failed = sum(isinstance(r, BaseException) for r in reports)
    returned = [r for r in reports if not isinstance(r, BaseException)]
    warnings = sum(len(r.errors or []) for r in returned)
    fixed_codes = {PublicationError: "publication", ValueError: "validation", RuntimeError: "runtime",
                   TimeoutError: "timeout", ConnectionError: "connection"}
    for r in reports:
        if isinstance(r, BaseException):
            print(f"SDK error code={fixed_codes.get(type(r), 'forecast_failure')}", file=client.audit)
        elif r.errors:
            print(f"SDK error question={r.question.id_of_question} post={r.question.id_of_post} "
                  f"code=incomplete_report count={len(r.errors)}", file=client.audit)
    expected = {r.question.id_of_question for r in returned}
    missing = expected != client.verified or len(expected) != len(returned)
    print(f"Completed verification: confirmed={len(client.verified)} failed={failed} "
          f"report_errors={warnings} inconsistent={int(missing)} "
          f"previous_unconfirmed={len(client.incomplete_previous)} legacy_unverified={len(client.unverified_legacy)}", file=client.audit)
    if failed or warnings or missing or client.incomplete_previous:
        raise PublicationError("Incomplete prize cycle")
