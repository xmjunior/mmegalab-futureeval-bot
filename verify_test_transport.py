"""One private transport test in the unscored Metaculus Bot Testing Area."""
import asyncio
import json
import os
import sys

from forecasting_tools import (
    BinaryQuestion, BinaryReport, GeneralLlm, MetaculusClient, MultipleChoiceQuestion,
    MultipleChoiceReport, NumericDistribution, NumericQuestion, NumericReport, Percentile,
    PredictedOption, PredictedOptionList,
)
from prize_publication import PublicationError, VerifiedPrizeClient, finish_prize_run, open_audit_stream


TEST_AREA = "bot-testing-area"
# Read-only discovery run 45465427673 verified empty forecast and private comment history.
# Only this question in the exclusive, unscored project is eligible for a write.
PINNED = (32977, 43322, 43324)


def in_test_area(question):
    projects = (question.api_json or {}).get("projects") or {}
    default = projects.get("default_project") or {}
    others = [t for t in projects.get("tournament", []) if t.get("slug") != TEST_AREA]
    return default.get("slug") == TEST_AREA and not others


def question_report(q, values, note):
    if type(q) is BinaryQuestion:
        if type(values) not in (int, float) or not 0.001 <= values <= 0.999:
            raise ValueError("Invalid probability")
        model, prediction = BinaryReport, values
    elif type(q) is MultipleChoiceQuestion:
        if not isinstance(values, list) or len(values) != len(q.options) or any(
                type(v) not in (int, float) or not 0 < v < 1 for v in values):
            raise ValueError("Invalid category masses")
        model = MultipleChoiceReport
        prediction = PredictedOptionList(predicted_options=[
            PredictedOption(option_name=label, probability=v) for label, v in zip(q.options, values)
        ])
    else:
        if not isinstance(values, list) or len(values) != 3 or any(type(v) not in (int, float) for v in values):
            raise ValueError("Invalid test quantiles")
        points = [Percentile(percentile=level, value=v) for level, v in zip((0.1, 0.5, 0.9), values)]
        model, prediction = NumericReport, NumericDistribution.from_question(points, q)
        prediction.get_cdf()
    return model.model_construct(question=q, prediction=prediction, errors=[], explanation=note)


async def test_report(question, audit):
    if type(question) is BinaryQuestion:
        spec, fixture = "a single probability", 0.5
    elif type(question) is MultipleChoiceQuestion:
        spec = f"a list of probabilities totaling 1 in this order: {question.options}"
        fixture = [1 / len(question.options)] * len(question.options)
    else:
        low, high = question.lower_bound, question.upper_bound
        spec = f"a list of the 10th, 50th and 90th percentiles in native units (range {low} to {high})"
        fixture = [low + (high - low) * p for p in (0.1, 0.5, 0.9)]
    prompt = (f"Answer this unscored testing-area forecasting question: {question.question_text}\n"
              f"Resolution: {question.resolution_criteria}\n"
              f"Return JSON with forecast ({spec}) and rationale (a short sentence).")
    if os.environ.get("GEMINI_API_KEY"):
        for name in ("gemini/gemma-4-31b-it", "gemini/gemma-4-26b-a4b-it"):
            try:
                model = GeneralLlm(name, temperature=0.2, timeout=45, allowed_tries=1, num_retries=0)
                raw = await asyncio.wait_for(model.invoke(prompt), timeout=60)
                record = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
                note = record["rationale"]
                if not isinstance(note, str) or not note.strip():
                    raise ValueError("Missing model rationale")
                report = question_report(question, record["forecast"], f"Unscored check using {name}.\n\n" + note)
            except Exception as exc:
                http = getattr(exc, "status_code", None)
                print(f"Hosted test model={name} outcome={type(exc).__name__} "
                      f"http={http if isinstance(http, int) else 'none'}", file=audit)
                continue
            print(f"Test forecast source=hosted_model model={name}", file=audit)
            return report
    print("Test forecast source=transport_fixture model=none", file=audit)
    return question_report(question, fixture,
        "UNSCORED TRANSPORT FIXTURE. This fixed test distribution verifies private API storage.")


def main(audit):
    client = VerifiedPrizeClient(audit, base_url="https://www.metaculus.com/api")
    if client.get_current_user_id() <= 0:
        raise PublicationError("Authentication failed")
    # Call the ordinary discovery method rather than auditing older testing-area comments.
    questions = MetaculusClient.get_all_open_questions_from_tournament(client, TEST_AREA, "unpack_subquestions")
    priorities = (BinaryQuestion, MultipleChoiceQuestion, NumericQuestion)
    eligible = sorted((q for q in questions if type(q) in priorities and in_test_area(q)),
                      key=lambda q: (priorities.index(type(q)), q.id_of_question))
    print(f"Auth verified project_slug={TEST_AREA} exclusive_project_candidates={len(eligible)}", file=audit)
    for q in eligible:
        project_id = q.api_json["projects"]["default_project"]["id"]
        ids = project_id, q.id_of_post, q.id_of_question
        if PINNED is not None and ids != PINNED:
            continue
        if type(q) is MultipleChoiceQuestion:
            client.category_order[q.id_of_question] = q.api_json["question"].get("all_options_ever") or q.options
        forecasts = client._my_forecasts(q.id_of_question)
        comments = list(client._my_comments(q.id_of_post))
        if q.already_forecasted or forecasts or comments:
            print(f"Excluded previous records project={ids[0]} post={ids[1]} question={ids[2]} "
                  f" SDK_prior={int(bool(q.already_forecasted))} own_forecasts={len(forecasts)} "
                  f"private_comments={len(comments)}", file=audit)
            if PINNED is not None:
                raise PublicationError("Pinned question has pre-existing records; no write sent")
            continue
        print(f"Unanswered test question project={ids[0]} post={ids[1]} question={ids[2]} type={type(q).__name__} "
              "own_forecasts=0 private_comments=0", file=audit)
        if PINNED is None:
            print("Read-only discovery complete; provider_calls=0 writes=0", file=audit)
            return
        report = asyncio.run(test_report(q, audit))
        asyncio.run(report.publish_report_to_metaculus(client))
        finish_prize_run([report], client)
        return
    raise PublicationError("No eligible unanswered question in the exclusive official test project")


if __name__ == "__main__":
    audit = open_audit_stream()
    try:
        main(audit)
    except PublicationError as exc:
        print(f"Test transport stopped: {exc}", file=audit)
        sys.exit(1)
