"""Read-only recheck of the one unscored receipt created by run 45465866165."""
import hashlib
import json
import sys

from forecasting_tools import NumericQuestion
from prize_publication import PublicationError, VerifiedPrizeClient, open_audit_stream


TEST_AREA = "bot-testing-area"
PROJECT_ID, POST_ID, QUESTION_ID, COMMENT_ID = 32977, 43322, 43324, 1358826


def main(audit):
    client = VerifiedPrizeClient(audit, base_url="https://www.metaculus.com/api")
    client.get_current_user_id()
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
    header, sep, rationale = note["text"].partition("\n\n")
    payload = {"continuous_cdf": values}
    digest = hashlib.sha256(json.dumps([QUESTION_ID, payload, rationale], sort_keys=True,
        ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()[:24]
    if not sep or header != f"Forecast receipt {QUESTION_ID}/{digest}":
        raise PublicationError("Private note digest and stored forecast do not agree")
    attached = client.attachment_matches(note, values, payload)
    print(f"Read-only receipt project={PROJECT_ID} question={QUESTION_ID} post={POST_ID} comment={COMMENT_ID} "
          f"own_forecasts=1 private_comments=1 stored_values={len(values)} digest_verified=true "
          f"group={bool(q.question_ids_of_group)} native_attachment="
          f"{'verified' if attached else 'unavailable' if attached is None else 'mismatch'} "
          "provider_calls=0 writes=0", file=audit)
    if attached is not True:
        raise PublicationError("Original private note native attachment not verified")


if __name__ == "__main__":
    audit = open_audit_stream()
    try:
        main(audit)
    except PublicationError as exc:
        print(f"Test transport stopped: {exc}", file=audit)
        sys.exit(1)
