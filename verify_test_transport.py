"""Verify the existing unscored receipt. One historical run may amend its private note."""
import hashlib
import json
import os

import requests
import sys

from forecasting_tools import NumericQuestion
from prize_publication import PublicationError, VerifiedPrizeClient, open_audit_stream


TEST_AREA = "bot-testing-area"
PROJECT_ID, POST_ID, QUESTION_ID, COMMENT_ID = 32977, 43322, 43324, 1358826


def main(audit):
    client = VerifiedPrizeClient(audit, base_url="https://www.metaculus.com/api")
    identity = client._read("/users/me", {})
    if identity.get("username") != "megalab-mario-bot" or client.get_current_user_id() != identity.get("id"):
        raise PublicationError("Expected test bot owner was not authenticated")
    post = client.get_question_by_post_id(POST_ID, "unpack_subquestions")
    questions = post if isinstance(post, list) else [post]
    q = next((q for q in questions if type(q) is NumericQuestion and
              (q.default_project_id, q.id_of_post, q.id_of_question) == (PROJECT_ID, POST_ID, QUESTION_ID)), None)
    if q is None:
        raise PublicationError("Pinned unscored test question unavailable")
    projects = q.api_json["projects"]
    if (projects["default_project"].get("slug") != TEST_AREA or
            any(t.get("slug") != TEST_AREA for t in projects.get("tournament", []))):
        raise PublicationError("Test area exclusivity failed")
    forecasts = client._my_forecasts(QUESTION_ID)
    comments = list(client._my_comments(POST_ID))
    if len(forecasts) != 1 or len(comments) != 1 or comments[0].get("id") != COMMENT_ID:
        raise PublicationError("Expected exactly one owned forecast and one specific private note")
    values = forecasts[0]["forecast_values"]
    if len(values) != 201 or not forecasts[0].get("start_time"):
        raise PublicationError("Unexpected persisted numeric vector")
    note = comments[0]
    payload = {"continuous_cdf": values}
    suffix = client.forecast_note_suffix(QUESTION_ID, payload)
    old_text = note["text"].removesuffix(suffix)
    header, sep, rationale = old_text.partition("\n\n")
    digest = hashlib.sha256(json.dumps([QUESTION_ID, payload, rationale], sort_keys=True,
        ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()[:24]
    if not sep or header != f"Forecast receipt {QUESTION_ID}/{digest}":
        raise PublicationError("Private note digest and stored forecast do not agree")
    attached = client.attachment_matches(note, values, payload)
    if attached is False:
        raise PublicationError("Incorrect native attachment")
    expected_text = old_text + suffix
    edits = 0
    if note["text"] != expected_text:
        if not (q.question_ids_of_group and attached is None and os.getenv("ALLOW_TEST_NOTE_EDIT") == "true"):
            raise PublicationError("Explicit original distribution unavailable in private note")
        # The verified old_text is the complete existing note. This path is bound
        # to an exact existing comment and one workflow run/attempt by the workflow.
        edits = 1
        try:
            response = client._request("POST", f"/comments/{COMMENT_ID}/edit/", json={"text":expected_text})
            status = response.status_code
        except requests.RequestException:
            status = None
        print(f"Private note edit comment={COMMENT_ID} http={status or 'interrupted'}", file=audit)
        if status is not None and status < 500 and status != 200:
            raise PublicationError("Private note edit rejected")
        # Never resend on an ambiguous result. Verify both resources instead.
        if client._my_forecasts(QUESTION_ID) != forecasts:
            raise PublicationError("Original forecast changed")
        checks = list(client._my_comments(POST_ID))
        if len(checks) != 1 or checks[0].get("id") != COMMENT_ID:
            raise PublicationError("Original private note missing")
        note = checks[0]
    if note.get("text") != expected_text:
        raise PublicationError("Private note distribution unconfirmed")
    if client.attachment_matches(note, values, payload) != attached:
        raise PublicationError("Native attachment changed")
    print(f"Verified receipt project={PROJECT_ID} question={QUESTION_ID} post={POST_ID} comment={COMMENT_ID} "
          f"own_forecasts=1 private_comments=1 stored_values={len(values)} digest_verified=true "
          f"group={bool(q.question_ids_of_group)} native_attachment={'verified' if attached else 'unavailable'} "
          f"forecast_in_comment=true provider_calls=0 forecast_writes=0 comment_edits={edits}", file=audit)


if __name__ == "__main__":
    audit = open_audit_stream()
    try:
        main(audit)
    except PublicationError as exc:
        print(f"Test transport stopped: {exc}", file=audit)
        sys.exit(1)
