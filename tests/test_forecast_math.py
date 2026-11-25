"""Affected forecast paths. Every provider and transport uses a local fixture."""
import unittest
from datetime import datetime, timedelta, timezone
from statistics import median
from unittest.mock import AsyncMock, Mock, patch

from forecasting_tools import BinaryQuestion, ConditionalQuestion, DateQuestion, MetaculusClient, NumericQuestion, PredictionAffirmed
from forecasting_tools.data_models.data_organizer import DataOrganizer
from forecasting_tools.data_models.questions import DiscreteQuestion
from pydantic import ValidationError

import main
import main_with_no_framework as standalone
from continuous_math import GridDistribution, student_t_distribution


LEVELS = (10, 20, 40, 60, 80, 90)


def response(values):
    return "\n".join(f"Percentile {p}: {v}" for p, v in zip(LEVELS, values, strict=True))


def fixture(qtype=NumericQuestion, **kwargs):
    defaults = dict(question_text="Offline synthetic fixture", id_of_question=915, id_of_post=716,
                    lower_bound=0, upper_bound=100, open_lower_bound=True, open_upper_bound=True)
    return qtype(**(defaults | kwargs))


def heights(distribution):
    return [p.percentile for p in distribution.get_cdf()]


class CapturingClient(MetaculusClient):
    """Exercise SDK payload construction and stop immediately before HTTP."""
    def __init__(self):
        super().__init__(token="offline-test")
        self.posts = []

    def _post_question_prediction(self, question_id, payload):
        self.posts.append((question_id, payload))

    def post_question_comment(self, *args, **kwargs):
        pass


class ForecastMathTests(unittest.IsolatedAsyncioTestCase):
    async def test_framework_median_sent_to_sdk_and_archive_is_exact(self):
        votes = ([42, 45, 47, 50, 53, 55], [46, 47, 51, 58, 61, 66])
        q = fixture()
        expected = [median(column) for column in zip(*[
            heights(student_t_distribution(q, [(p/100, v) for p, v in zip(LEVELS, values)]))
            for values in votes])]
        client = CapturingClient()
        bot = main.FallTemplateBot2026(predictions_per_research_report=2, metaculus_client=client,
                                     enable_summarize_research=False, publish_reports_to_metaculus=True)
        model = Mock(invoke=AsyncMock(side_effect=[response(v) for v in votes]))
        with patch.object(bot, "run_research", AsyncMock(return_value="Fixture research")), \
             patch.object(bot, "get_llm", return_value=model):
            report = await bot.forecast_question(q)
        self.assertEqual(client.posts, [(915, {"continuous_cdf": expected})])
        restored, = DataOrganizer._load_objects_from_json([report.to_json()], DataOrganizer.get_all_report_types())
        self.assertIsInstance(restored.prediction, GridDistribution)
        await restored.publish_report_to_metaculus(client)
        self.assertEqual(client.posts[0], client.posts[1])

    async def test_discrete_narrow_ties_survive_reelicitation(self):
        q = fixture(DiscreteQuestion, upper_bound=0.000001, cdf_size=24)
        indices = [10, 20, 20, 20, 70, 85]
        bot = main.FallTemplateBot2026(metaculus_client=CapturingClient())
        model = Mock(invoke=AsyncMock(return_value="Unstructured fixture"),
                     choose_integers=AsyncMock(return_value=indices))
        with patch.object(bot, "get_llm", return_value=model):
            result = await bot._numeric_prompt_to_forecast(q, "fixture")
        cls = DataOrganizer.get_report_type_for_question_type(type(q))
        agg = await cls.aggregate_predictions([result.prediction_value], q)
        restored = cls.from_json(cls(question=q, prediction=agg, explanation="# fixture").to_json())
        self.assertEqual(len(restored.prediction.get_cdf()), 24)
        self.assertEqual(heights(restored.prediction), heights(result.prediction_value))
        model.choose_integers.assert_awaited_once()

    async def test_date_intraday_bounds_and_invalid_midnight_reelicited(self):
        tz = timezone(timedelta(hours=-3))
        lower = datetime(2026, 11, 20, 9, 5, tzinfo=tz)
        upper = lower + timedelta(hours=16)
        q = fixture(DateQuestion, lower_bound=lower, upper_bound=upper,
                    open_lower_bound=False, open_upper_bound=False)
        bot = main.FallTemplateBot2026(metaculus_client=CapturingClient())
        upper_message, lower_message = bot._create_upper_and_lower_bound_messages(q)
        self.assertIn("2026-11-20T12:05:00Z", lower_message)
        self.assertIn("2026-11-21T04:05:00Z", upper_message)
        model = Mock(invoke=AsyncMock(return_value=response(["2026-11-20"]*6)),
                     choose_integers=AsyncMock(return_value=[0,20,40,60,80,100]))
        with patch.object(bot, "get_llm", return_value=model):
            result = await bot._numeric_prompt_to_forecast(q, "fixture")
        d = result.prediction_value
        self.assertEqual((d.lower_bound, d.upper_bound), (lower.timestamp(), upper.timestamp()))
        self.assertEqual((heights(d)[0], heights(d)[-1]), (0, 1))
        model.choose_integers.assert_awaited_once()
        report_cls = DataOrganizer.get_report_type_for_question_type(type(q))
        r = report_cls(question=q, prediction=d, explanation="# fixture")
        self.assertEqual(heights(report_cls.from_json(r.to_json()).prediction), heights(d))

    async def test_binary_extremes_direct_fallback_and_standalone(self):
        question = BinaryQuestion(question_text="Offline binary fixture")
        bot = main.FallTemplateBot2026(metaculus_client=CapturingClient())
        for text, expected in [("0", .001), (".2", .002), ("50", .5), ("99.9", .999), ("100", .999)]:
            with self.subTest(percent=text):
                model = Mock(invoke=AsyncMock(return_value=f"Probability: {text}%"))
                with patch.object(bot, "get_llm", return_value=model):
                    prediction = await bot._binary_prompt_to_forecast(question, "fixture")
                self.assertEqual(prediction.prediction_value, expected)
                self.assertAlmostEqual(standalone.extract_probability_from_response_as_percentage_not_decimal(
                    f"Earlier 50%, final {text}%"), 100*expected)
        for value, expected in [(0, .001), (100, .999)]:
            model = Mock(invoke=AsyncMock(return_value="Unstructured fixture"),
                         choose_integers=AsyncMock(return_value=[value]))
            with patch.object(bot, "get_llm", return_value=model):
                prediction = await bot._binary_prompt_to_forecast(question, "fixture")
            self.assertEqual(prediction.prediction_value, expected)

    async def test_standalone_binary_payload_at_upper_limit(self):
        q = dict(title="Fixture", resolution_criteria="", description="", fine_print="", type="binary")
        with patch.object(standalone, "run_research", return_value="Offline"), \
             patch.object(standalone, "call_llm", AsyncMock(return_value="Probability: 99.9%")):
            probability, _ = await standalone.get_binary_gpt_prediction(q, 1)
        self.assertEqual(standalone.create_forecast_payload(probability, "binary")["probability_yes"], .999)

    async def test_log_standalone_and_sdk_payloads_match(self):
        for lo, hi, zero, values in [(1, 1000, 0, [1, 2, 3, 4, 7, 10]),
                                      (-1000, -1, 0, [-800, -500, -200, -150, -100, -3])]:
            with self.subTest(lower=lo):
                q = fixture(lower_bound=lo, upper_bound=hi, zero_point=zero,
                            open_lower_bound=False, open_upper_bound=False)
                d = student_t_distribution(q, [(p/100, v) for p,v in zip(LEVELS, values)])
                cdf = standalone.generate_continuous_cdf(dict(zip(LEVELS, values)), "numeric",
                                                        False, False, hi, lo, zero, 201)
                report_cls = DataOrganizer.get_report_type_for_question_type(type(q))
                report = report_cls(question=q, prediction=d, explanation="# fixture")
                client = CapturingClient()
                await report_cls.from_json(report.to_json()).publish_report_to_metaculus(client)
                self.assertEqual(client.posts[0][1]["continuous_cdf"],
                                 standalone.create_forecast_payload(cdf, "numeric")["continuous_cdf"])

    async def test_conditional_affirmations_and_continuous_branches_round_trip(self):
        q = fixture(lower_bound=-100, upper_bound=-1, zero_point=0)
        parent = BinaryQuestion(question_text="Offline parent")
        conditional = ConditionalQuestion(question_text="Offline conditional", parent=parent, child=q,
                                         question_yes=q, question_no=q.model_copy(update={"id_of_question":916}))
        d = student_t_distribution(q, [(.10,-80),(.90,-30)])
        vote = main._GridConditionalPrediction(parent=PredictionAffirmed(), child=PredictionAffirmed(),
                                              prediction_yes=d, prediction_no=d)
        cls = DataOrganizer.get_report_type_for_question_type(type(conditional))
        aggregate = await cls.aggregate_predictions([vote], conditional)
        report = cls.from_json(cls(question=conditional, prediction=aggregate, explanation="# fixture").to_json())
        client = CapturingClient()
        await report.publish_report_to_metaculus(client)
        self.assertIsInstance(report.prediction.parent, PredictionAffirmed)
        self.assertEqual(client.posts, [(i, {"continuous_cdf": heights(d)}) for i in (915, 916)])

    async def test_historical_sparse_report_and_invalid_tagged_grid(self):
        q = fixture()
        cls = DataOrganizer.get_report_type_for_question_type(type(q))
        legacy = main.NumericDistribution.from_question(
            [main.Percentile(percentile=.1, value=30), main.Percentile(percentile=.9, value=60)], q)
        saved = cls(question=q, prediction=legacy, explanation="# fixture").to_json()
        self.assertIs(type(cls.from_json(saved).prediction), main.NumericDistribution)
        grid = student_t_distribution(q, [(.1,30),(.9,60)])
        saved = cls(question=q, prediction=grid, explanation="# fixture").to_json()
        saved["prediction"]["declared_percentiles"][0]["percentile"] = .2
        with self.assertRaises(ValidationError):
            cls.from_json(saved)
