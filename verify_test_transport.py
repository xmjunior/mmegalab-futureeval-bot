"""One private transport test in the unscored Metaculus Bot Testing Area."""
import asyncio
import json
import os
import sys

from forecasting_tools import BinaryQuestion, BinaryReport, GeneralLlm, MetaculusClient
from prize_publication import PublicationError, VerifiedPrizeClient, finish_prize_run, open_audit_stream


TEST_AREA = "bot-testing-area"
# Discovery is read-only. A later reviewed commit can pin one unanswered pair.
PINNED = None


def in_test_area(question):
    projects = (question.api_json or {}).get("projects") or {}
    default = projects.get("default_project") or {}
    others = [t for t in projects.get("tournament", []) if t.get("slug") != TEST_AREA]
    return default.get("slug") == TEST_AREA and not others


async def test_report(question, audit):
    prompt = (f"Answer this unscored testing-area forecasting question: {question.question_text}\n"
              f"Resolution: {question.resolution_criteria}\n"
              'Output only a JSON object with probability (0 to 1) and rationale (a short sentence).')
    if os.environ.get("GEMINI_API_KEY"):
        for name in ("gemini/gemma-4-31b-it", "gemini/gemma-4-26b-a4b-it"):
            try:
                model = GeneralLlm(name, temperature=0.2, timeout=45, allowed_tries=1, num_retries=0)
                raw = await asyncio.wait_for(model.invoke(prompt), timeout=60)
                record = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
                p, note = record["probability"], record["rationale"]
                if type(p) not in (int, float) or not 0.001 <= p <= 0.999 or not isinstance(note, str) or not note.strip():
                    raise ValueError("Malformed model output")
            except Exception as exc:
                http = getattr(exc, "status_code", None)
                print(f"Hosted test model={name} outcome={type(exc).__name__} "
                      f"http={http if isinstance(http, int) else 'none'}", file=audit)
                continue
            print(f"Test forecast source=hosted_model model={name}", file=audit)
            return BinaryReport.model_construct(question=question, prediction=p, errors=[],
                explanation="Unscored testing-area check.\n\n" + note)
    print("Test forecast source=transport_fixture model=none", file=audit)
    return BinaryReport.model_construct(question=question, prediction=0.5, errors=[],
        explanation="UNSCORED TRANSPORT FIXTURE. This fixed test value verifies private API storage.")


def main(audit):
    client = VerifiedPrizeClient(audit, base_url="https://www.metaculus.com/api")
    if client.get_current_user_id() <= 0:
        raise PublicationError("Authentication failed")
    # Call the ordinary discovery method rather than auditing older testing-area comments.
    questions = MetaculusClient.get_all_open_questions_from_tournament(client, TEST_AREA, "unpack_subquestions")
    binary = sorted((q for q in questions if type(q) is BinaryQuestion), key=lambda q: q.id_of_question)
    eligible = [q for q in binary if in_test_area(q)]
    print(f"Auth verified project_slug={TEST_AREA} binaries={len(binary)} "
          f"exclusive_project_binaries={len(eligible)}", file=audit)
    for q in eligible:
        project_id = q.api_json["projects"]["default_project"]["id"]
        ids = project_id, q.id_of_post, q.id_of_question
        if PINNED is not None and ids != PINNED:
            continue
        forecasts = client._my_forecasts(q.id_of_question)
        comments = list(client._my_comments(q.id_of_post))
        if q.already_forecasted or forecasts or comments:
            print(f"Excluded previous records project={ids[0]} post={ids[1]} question={ids[2]} "
                  f" SDK_prior={int(bool(q.already_forecasted))} own_forecasts={len(forecasts)} "
                  f"private_comments={len(comments)}", file=audit)
            if PINNED is not None:
                raise PublicationError("Pinned question has pre-existing records; no write sent")
            continue
        print(f"Unanswered test question project={ids[0]} post={ids[1]} question={ids[2]} "
              "own_forecasts=0 private_comments=0", file=audit)
        if PINNED is None:
            print("Read-only discovery complete; provider_calls=0 writes=0", file=audit)
            return
        report = asyncio.run(test_report(q, audit))
        asyncio.run(report.publish_report_to_metaculus(client))
        finish_prize_run([report], client)
        return
    raise PublicationError("No eligible unanswered binary in the exclusive official test project")


if __name__ == "__main__":
    audit = open_audit_stream()
    try:
        main(audit)
    except PublicationError as exc:
        print(f"Test transport stopped: {exc}", file=audit)
        sys.exit(1)
