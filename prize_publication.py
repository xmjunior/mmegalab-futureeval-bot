"""Verify prize submissions without writing sealed forecasts to Actions logs."""
import contextvars
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
import sys
import time

import requests
from groq_free import ROUTE as GROQ_ROUTE
from forecasting_tools import ConditionalQuestion, MetaculusClient


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


_question_routes = contextvars.ContextVar("question_routes", default=None)


class ProviderAuditHandler(logging.Handler):
    """Emit only known provider events, with validated scalar arguments."""
    MODELS = frozenset(("gemini/gemma-4-31b-it", "gemini/gemma-4-26b-a4b-it", "openrouter/openrouter/free", GROQ_ROUTE))
    CONNECTION_ERRORS = frozenset(("Timeout", "TimeoutError", "ConnectionError", "APIConnectionError"))

    def __init__(self, output):
        super().__init__()
        self.output = output

    def emit(self, record):
        args = record.args
        if not isinstance(args, tuple) or not args or type(args[0]) is not str or args[0] not in self.MODELS:
            return
        if record.msg == "Model responded: %s" and len(args) == 1:
            routes = _question_routes.get()
            if routes is not None:
                routes.add(args[0])
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
        self.deadlines = {}
        self.verified = set()
        self.incomplete_previous = set()
        self.unverified_legacy = set()
        self.owner_id = None

    @contextmanager
    def track_question_routes(self):
        # Async research and forecast subtasks inherit this question's set.
        token = _question_routes.set(set())
        try:
            yield
        finally:
            _question_routes.reset(token)

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

    @staticmethod
    def forecast_targets(question):
        if isinstance(question, ConditionalQuestion):
            return (question.question_yes, question.question_no)
        return (question,)

    @staticmethod
    def deadline(question):
        raw = (getattr(question, "api_json", None) or {}).get("question") or {}
        candidates = (getattr(question, "close_time", None), raw.get("actual_close_time"),
                      raw.get("scheduled_close_time"))
        ends = []
        for value in candidates:
            try:
                dt = datetime.fromisoformat(value) if isinstance(value, str) else value
                if isinstance(dt, datetime):
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    ends.append(dt.timestamp())
            except (ValueError, OverflowError):
                continue
        return min(ends, default=math.inf)

    def get_all_open_questions_from_tournament(self, *args, **kwargs):
        questions = super().get_all_open_questions_from_tournament(*args, **kwargs)
        by_post = {}
        targets = []
        for question in questions:
            parts = self.forecast_targets(question)
            targets.extend(parts)
            if len(parts) == 2 and any(p.already_forecasted for p in parts):
                for part in parts:
                    if not part.already_forecasted:
                        self.incomplete_previous.add(part.id_of_question)
                        print(f"Conditional branch lacks prior forecast question={part.id_of_question} "
                              f"post={part.id_of_post}", file=self.audit)
        for q in targets:
            self.deadlines[q.id_of_question] = self.deadline(q)
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
        return sorted(questions, key=lambda q: (self.deadline(q), q.id_of_question or 0))

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
        routes = _question_routes.get()
        if routes:
            comment_text += "\n\nHosted routes used for this question: " + ", ".join(sorted(routes))
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
        # A conditional or grouped post has no single native forecast to attach.
        # Preserve the exact question-specific forecast in its private note too.
        text = f"Forecast receipt {question_id}/{signature}\n\n{comment_text}"
        text += self.forecast_note_suffix(question_id, payload)
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
        if not self._same_values(values, expected):
            raise PublicationError("Unexpected persisted forecast")
        comment = next((c for c in self._my_comments(post_id) if c.get("text") == text), None)
        if not comment or not comment.get("id"):
            raise PublicationError("Matching private rationale unconfirmed")
        attached = self.attachment_matches(comment, values, payload)
        if attached is False:
            raise PublicationError("Private comment attaches a different forecast")
        self.verified.add(question_id)
        deadline = self.deadlines.get(question_id, math.inf)
        margin = int(deadline - time.time()) if math.isfinite(deadline) else "unknown"
        print(f"Verified publication question={question_id} post={post_id} comment={int(comment['id'])} "
              f"response={status or 'interrupted'} stored_values={len(values)} private=true "
              f"native_attachment={'verified' if attached else 'unavailable'} "
              f"forecast_in_comment=true seconds_to_close={margin}", file=self.audit)

    @staticmethod
    def forecast_note_suffix(question_id, payload):
        stated = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        return f"\n\nSubmitted forecast for question {question_id}:\n```json\n{stated}\n```"

    @staticmethod
    def _same_values(values, expected):
        return (isinstance(values, list) and len(values) == len(expected) and all(
            (v is None and e is None) or (type(v) in (int, float) and type(e) in (int, float)
                                         and math.isclose(v, e, rel_tol=1e-7, abs_tol=1e-9))
            for v, e in zip(values, expected)))

    @classmethod
    def attachment_matches(cls, comment, own_values, payload):
        """Return None when the server omits the native attachment or returns a flag."""
        obj = comment.get("included_forecast")
        if not isinstance(obj, dict):
            return None
        if "probability_yes" in payload:
            p = obj.get("probability_yes")
            attached = [1 - p, p] if type(p) in (int, float) else []
        elif "continuous_cdf" in payload:
            attached = obj.get("continuous_cdf")
        else:
            attached = obj.get("probability_yes_per_category")
        return cls._same_values(attached, own_values)



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
    targets = [q.id_of_question for r in returned for q in client.forecast_targets(r.question)]
    missing = set(targets) != client.verified or len(set(targets)) != len(targets)
    print(f"Completed verification: confirmed={len(client.verified)} failed={failed} "
          f"report_errors={warnings} inconsistent={int(missing)} "
          f"previous_unconfirmed={len(client.incomplete_previous)} legacy_unverified={len(client.unverified_legacy)}", file=client.audit)
    if failed or warnings or missing or client.incomplete_previous:
        raise PublicationError("Incomplete prize cycle")
