import argparse
import asyncio
import json
import logging
import os
import platform
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Literal

import dotenv

from local_forecast import CpuForecaster, MODEL_NAME

# Runtime helpers (env validation, banners, dependency-warning suppression).
from bot_helpers import (
    check_environment,
    print_run_summary_banner,
    print_startup_banner,
    silence_noisy_dependencies,
)

silence_noisy_dependencies()

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    BinaryReport,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    DateQuestion,
    Percentile,
    ConditionalQuestion,
    ConditionalPrediction,
    PredictionTypes,
    PredictionAffirmed,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
)

dotenv.load_dotenv()
logger = logging.getLogger(__name__)


class FreeForecastLlm(GeneralLlm):
    """Use only free endpoints. Share backoff across research, forecasts and parsing."""

    def __init__(self):
        names = []
        if os.getenv("GEMINI_API_KEY"):
            names.extend(["gemini/gemma-4-31b-it", "gemini/gemma-4-26b-a4b-it"])
        if os.getenv("OPENROUTER_API_KEY"):
            names.append("openrouter/openrouter/free")
        if not names:
            raise ValueError("A Gemini or OpenRouter API key is required")

        super().__init__(names[0], allowed_tries=1)
        self.routes = [
            GeneralLlm(name, temperature=0.2, timeout=55, allowed_tries=1, num_retries=0)
            for name in names
        ]
        self.available_at = dict.fromkeys(names, 0.0)
        self.lock = asyncio.Lock()
        self.cpu = None
        # CPU fallback is useful for dry-run diagnostics, but it is intentionally
        # disabled by default for prize-bearing runs because a tiny local model can
        # produce badly calibrated forecasts. Enable only with an explicit env flag.
        self.allow_cpu_fallback = os.getenv("ALLOW_CPU_FALLBACK", "false").lower() in {"1", "true", "yes"}

    def to_dict(self):
        return {"free_routes_in_order": [route.model for route in self.routes],
                "cpu_fallback": MODEL_NAME,
                "cpu_fallback_enabled": self.allow_cpu_fallback}

    async def choose_integers(self, prompt: str, count: int, low: int, high: int) -> list[int]:
        prompt += (f"\nReturn only a JSON array of {count} integers between {low} and {high}, "
                   "inclusive. Each number must be your own selection.")
        if self.cpu is None:
            answer = await self.invoke(prompt)
            try:
                values = json.loads(answer.strip().removeprefix("```json").removesuffix("```").strip())
            except ValueError:
                values = None
            if (isinstance(values, list) and len(values) == count
                    and all(type(n) is int and low <= n <= high for n in values)):
                return values
        if not self.allow_cpu_fallback:
            raise RuntimeError("Hosted model returned an invalid structured answer; CPU fallback is disabled")
        async with self.lock:
            if self.cpu is None:
                self.cpu = await asyncio.to_thread(CpuForecaster)
            return await asyncio.to_thread(self.cpu.choose_integers, prompt, count, low, high)

    async def invoke(self, prompt, system_prompt=None):
        async with self.lock:
            if self.cpu is not None:
                if not self.allow_cpu_fallback:
                    raise RuntimeError("CPU fallback is disabled for this run")
                return await asyncio.to_thread(self.cpu.invoke, prompt, system_prompt)
            # A 503 on one route can have a later retry time than a 503 on another.
            # Keep each route eligible for its second attempt even after its peer retries.
            deadline = time.monotonic() + 180
            attempts = dict.fromkeys(self.available_at, 0)
            while time.monotonic() < deadline:
                pending = [r for r in self.routes if attempts[r.model] < 2
                           and self.available_at[r.model] < deadline]
                if not pending:
                    break
                now = time.monotonic()
                ready = [r for r in pending if self.available_at[r.model] <= now]
                if not ready:
                    await asyncio.sleep(min(self.available_at[r.model] for r in pending) - now)
                    continue
                route = min(ready, key=lambda r: attempts[r.model])
                attempts[route.model] += 1
                try:
                    answer = await route.invoke(prompt, system_prompt)
                except Exception as exc:
                    status = getattr(exc, "status_code", None)
                    retryable = status in (408, 429, 500, 502, 503, 504) or isinstance(
                        exc, (TimeoutError, ConnectionError)
                    ) or type(exc).__name__ in ("APIConnectionError", "Timeout")
                    if not retryable and status not in (401, 402, 403, 404):
                        raise
                    headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
                    hint = str(headers.get("retry-after", "")).strip()
                    delay = 45 if status == 503 else 12 * attempts[route.model]
                    if hint.isdigit():
                        delay = max(delay, int(hint))
                    elif hint:
                        try:
                            until = parsedate_to_datetime(hint)
                            delay = max(delay, int((until - datetime.now(timezone.utc)).total_seconds()) + 1)
                        except (ValueError, TypeError, OverflowError):
                            pass
                    if not retryable or (route.model.startswith("openrouter/")
                                         and status == 429 and re.search(r"day|daily", str(exc), re.I)):
                        delay = max(delay, 3600)
                    self.available_at[route.model] = time.monotonic() + delay
                    logger.warning("Free route %s unavailable (%s), cooldown %ss",
                                   route.model, status or type(exc).__name__, delay)
                    continue
                if not isinstance(answer, str) or not answer.strip():
                    self.available_at[route.model] = time.monotonic() + 12
                    continue
                logger.info("Model responded: %s", route.model)
                return answer
            if self.allow_cpu_fallback and platform.system() == "Linux" and platform.machine() == "x86_64":
                logger.info("Hosted routes unavailable; starting CPU inference for this run")
                self.cpu = await asyncio.to_thread(CpuForecaster)
                return await asyncio.to_thread(self.cpu.invoke, prompt, system_prompt)
            if not self.allow_cpu_fallback:
                raise RuntimeError("All hosted free routes unavailable; refusing degraded CPU forecast")
            raise RuntimeError("Free routes unavailable; CPU fallback requires Linux x86_64")


class FallTemplateBot2026(ForecastBot):
    """
    This is the template bot for the Fall 2026 FutureEval Bot Tournament.
    This is a copy of what is used by Metaculus to run the Metac Bots in our benchmark, provided as a template for new bot makers.
    This template is given as-is, and is use-at-your-own-risk.
    We have covered most test cases in forecasting-tools it may be worth double checking key components locally.
    So far our track record has been 1 mentionable bug per season (affecting forecasts for 1-2% of total questions)

    This bot is identical to SummerTemplateBot2026.

    The main entry point of this bot is `bot.forecast_on_tournament(tournament_id)` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Alternatively, you can use the MetaculusClient to make a custom filter of questions to forecast on
    and forecast them with `bot.forecast_questions(questions)`

    Only the research and forecast functions need to be implemented in ForecastBot subclasses,
    though you may want to override other ForecastBot functions.
    The forecast methods validate explicit model outputs and re-elicit ambiguous formats before submission.

    By default (i.e. 'tournament' mode), when you run this script, it will forecast on any open questions in the
    primary bot tournament and MiniBench. If you want to forecast on only one or the other, you can remove one
    of them from the 'tournament' mode code at the bottom of the file.

    You can experiment with what models work best with your bot by using the `llms` parameter when initializing the bot.
    You can initialize the bot with any number of models. For example,
    ```python
    my_bot = MyBot(
        ...
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
                model="openrouter/openai/gpt-4o", # "anthropic/claude-sonnet-4-20250514", etc (see docs for litellm)
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openai/gpt-4o-mini",
            "researcher": "asknews/news-summaries",
            "parser": "openai/gpt-4o-mini",
        },
    )
    ```

    Then you can access the model in custom functions like this:
    ```python
    research_strategy = self.get_llm("researcher", "model_name")
    if research_strategy == "asknews/news-summaries":
        ...
    # OR
    summarizer = await self.get_llm("summarizer", "llm").invoke(prompt)
    # OR
    reasoning = await self.get_llm("default", "llm").invoke(prompt)
    ```

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```python
    from forecasting_tools import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Additionally OpenRouter has large rate limits immediately on account creation
    """

    _max_concurrent_questions = (
        1  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    async def _run_individual_question(self, question):
        tracker = getattr(self.metaculus_client, "track_question_routes", None)
        if tracker is None:
            return await super()._run_individual_question(question)
        with tracker():
            return await super()._run_individual_question(question)

    ##################################### RESEARCH #####################################

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            research = ""
            researcher = self.get_llm("researcher")

            prompt = self._get_research_prompt(question, researcher)

            if isinstance(researcher, GeneralLlm):
                research = await researcher.invoke(prompt)
            elif (
                researcher == "asknews/news-summaries"
                or researcher == "asknews/deep-research/low-depth"
                or researcher == "asknews/deep-research/medium-depth"
                or researcher == "asknews/deep-research/high-depth"
            ):
                research = await AskNewsSearcher().call_preconfigured_version(
                    researcher, prompt
                )
            elif researcher.startswith("smart-searcher"):
                model_name = researcher.removeprefix("smart-searcher/")
                searcher = SmartSearcher(
                    model=model_name,
                    temperature=0,
                    num_searches_to_run=2,
                    num_sites_per_search=10,
                    use_advanced_filters=False,
                )
                research = await searcher.invoke(prompt)
            elif not researcher or researcher == "None" or researcher == "no_research":
                research = ""
            else:
                research = await self.get_llm("researcher", "llm").invoke(prompt)
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            return research

    @staticmethod
    def _get_research_prompt(
        question: MetaculusQuestion, researcher: str | GeneralLlm
    ) -> str:
        if GeneralLlm.to_model_name(researcher) == "asknews/news-summaries":
            return question.question_text

        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            You do not produce forecasts yourself.

            Question:
            {question.question_text}

            This question's outcome will be determined by the specific criteria below:
            {question.resolution_criteria}

            {question.fine_print}
            """
        )
        return prompt

    ##################################### BINARY QUESTIONS #####################################

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.
            {self._get_conditional_disclaimer_if_necessary(question)}

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )

        return await self._binary_prompt_to_forecast(question, prompt)

    async def _binary_prompt_to_forecast(
        self,
        question: BinaryQuestion,
        prompt: str,
    ) -> ReasonedPrediction[float]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        # The forecast prompt already specifies a machine-readable final line.
        last_line = reasoning.strip().splitlines()[-1].strip().replace("**", "")
        match = re.fullmatch(
            r"(?:Final answer:\s*)?Probability:\s*(\d+(?:\.\d+)?)%(?:,\s*0-100)?[.]?",
            last_line, re.I,
        )
        if match:
            parsed = BinaryPrediction(prediction_in_decimal=float(match[1]) / 100)
            return ReasonedPrediction(
                prediction_value=max(0.01, min(0.99, parsed.prediction_in_decimal)),
                reasoning=reasoning,
            )
        request = (f"Question: {question.question_text}\n{question.resolution_criteria}\n"
                   f"Analysis: {reasoning}\nSelect your final probability of YES as a whole percent.")
        (percent,) = await self.get_llm("default", "llm").choose_integers(request, 1, 0, 100)
        decimal_pred = max(0.01, min(0.99, percent / 100))
        reasoning += f"\nFinal model probability: {percent}%"
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {decimal_pred}."
        )
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    ##################################### MULTIPLE CHOICE QUESTIONS #####################################

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of a scenario that results in an unexpected outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            End with one line per option in this order: {question.options}.
            Use the exact option name, a colon, then its probability as a percentage.
            The percentages must sum to 100%. Give no further text after these lines.
            """
        )
        return await self._multiple_choice_prompt_to_forecast(question, prompt)

    async def _multiple_choice_prompt_to_forecast(
        self,
        question: MultipleChoiceQuestion,
        prompt: str,
    ) -> ReasonedPrediction[PredictedOptionList]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        lines = [line.strip().replace("**", "") for line in reasoning.splitlines() if line.strip()]
        final = lines[-len(question.options):]
        options = []
        for option, line in zip(question.options, final):
            match = re.fullmatch(rf"(?:[-*]\s*)?{re.escape(option)}\s*:\s*(\d+(?:\.\d+)?)%", line, re.I)
            if not match:
                break
            options.append({"option_name": option, "probability": float(match[1]) / 100})
        if len(options) == len(question.options):
            try:
                parsed = PredictedOptionList.model_validate({"predicted_options": options})
                return ReasonedPrediction(prediction_value=parsed, reasoning=reasoning)
            except ValueError:
                pass  # Request explicit relative weights if the percentages didn't sum to 100.
        request = (f"Question: {question.question_text}\n{question.resolution_criteria}\n"
                   f"Analysis: {reasoning}\nOptions in order: {question.options}\n"
                   "Assign one relative likelihood weight to each option in that order. "
                   "Weights need not sum to 100; they will be divided by their sum.")
        weights = await self.get_llm("default", "llm").choose_integers(request, len(question.options), 1, 100)
        total = sum(weights)
        predicted_option_list = PredictedOptionList.model_validate({"predicted_options": [
            {"option_name": name, "probability": weight / total}
            for name, weight in zip(question.options, weights)
        ]})
        reasoning += f"\nFinal model weights in option order: {weights}"
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}."
        )
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=reasoning
        )

    ##################################### NUMERIC QUESTIONS #####################################

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested and give your answer in these units (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there. The value for percentile 10 should always be less than the value for percentile 20, and so on.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX (lowest number value)
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX (highest number value)
            "
            """
        )
        return await self._numeric_prompt_to_forecast(question, prompt)

    @staticmethod
    def _parse_percentile_lines(reasoning: str, question) -> list[Percentile] | None:
        # Read only a complete final block in the format requested from the forecaster.
        # If it used a different layout or units, let the framework parser handle it.
        lines = [line.strip().replace("**", "") for line in reasoning.splitlines()
                 if line.strip() and line.strip() not in {'"', '```'}]
        levels = (10, 20, 40, 60, 80, 90)
        if len(lines) < len(levels):
            return None
        result = []
        for level, line in zip(levels, lines[-len(levels):]):
            match = re.fullmatch(rf"(?:[-*]\s*)?Percentile\s+{level}%?\s*:\s*(.+)", line, re.I)
            if not match:
                return None
            raw = match[1].strip()
            if isinstance(question, DateQuestion):
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}Z)?", raw):
                    return None
                try:
                    value = datetime.fromisoformat(raw).replace(tzinfo=timezone.utc).timestamp()
                except ValueError:
                    return None
            else:
                if not re.fullmatch(r"[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?", raw):
                    return None
                value = float(raw.replace(",", ""))
            result.append(Percentile(percentile=level / 100, value=value))
        return result

    async def _numeric_prompt_to_forecast(self, question, prompt):
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info("Reasoning for URL %s: %s", question.page_url, reasoning)
        direct = self._parse_percentile_lines(reasoning, question)
        if direct is not None:
            try:
                prediction = NumericDistribution.from_question(direct, question)
                return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)
            except ValueError as exc:
                logger.info("Model distribution needs revision for %s: %s", question.page_url, exc)

        # Re-elicit actual outcomes anchored to the question's scale when units,
        # syntax or bounds are ambiguous. Never silently clamp or invent outcomes.
        is_date = isinstance(question, DateQuestion)
        lower = question.lower_bound.timestamp() if is_date else question.lower_bound
        upper = question.upper_bound.timestamp() if is_date else question.upper_bound
        zero = question.zero_point
        grid = []
        labels = []
        for index in range(101):
            fraction = index / 100
            if zero is None:
                value = lower + (upper - lower) * fraction
            else:
                value = zero + (lower - zero) * ((upper - zero) / (lower - zero)) ** fraction
            grid.append(value)
            label = (datetime.fromtimestamp(value, timezone.utc).isoformat() if is_date
                     else format(value, ".10g"))
            labels.append(f"{index}={label}")
        levels = [10, 20, 40, 60, 80, 90]
        request = (f"Question: {question.question_text}\n{question.resolution_criteria}\n"
                   f"Prior analysis (check its units and revise if needed): {reasoning}\n"
                   f"Outcome grid in the question's units: {', '.join(labels)}\n"
                   "Choose SIX outcome indices for your distribution, from lowest to highest "
                   "in numeric or date order. Your six selections will be sorted by outcome and "
                   "assigned to percentiles 10, 20, 40, 60, 80 and 90, respectively. "
                   "These integers identify outcome values, not percentile levels. "
                   "Consider the evidence and the uncertainty before choosing.")
        indices = await self.get_llm("default", "llm").choose_integers(request, 6, 0, 100)
        points = [Percentile(percentile=level / 100, value=grid[index])
                  for level, index in zip(levels, sorted(indices))]
        prediction = NumericDistribution.from_question(points, question)
        outcomes = [labels[i].split("=", 1)[1] for i in sorted(indices)]
        reasoning += f"\nFinal model-selected outcomes for percentiles {levels}: {outcomes}"
        logger.info("Validated model-selected percentiles for %s: %s", question.page_url, outcomes)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    ##################################### DATE QUESTIONS #####################################

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - This is a date question, and as such, the answer must be expressed in terms of dates.
            - The dates must be written in the format of YYYY-MM-DD. If hours matter, please append the date with the hour in UTC and military time: YYYY-MM-DDTHH:MM:SSZ. No other formatting is allowed.
            - Always start with a lower date chronologically and then increase from there.
            - Do NOT forget this. The dates must be written in chronological order starting at the earliest time at percentile 10 and increasing from there.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: YYYY-MM-DD (oldest date)
            Percentile 20: YYYY-MM-DD
            Percentile 40: YYYY-MM-DD
            Percentile 60: YYYY-MM-DD
            Percentile 80: YYYY-MM-DD
            Percentile 90: YYYY-MM-DD (newest date)
            "
            """
        )
        return await self._numeric_prompt_to_forecast(question, prompt)

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion | DateQuestion
    ) -> tuple[str, str]:
        if isinstance(question, NumericQuestion):
            if question.nominal_upper_bound is not None:
                upper_bound_number = question.nominal_upper_bound
            else:
                upper_bound_number = question.upper_bound
            if question.nominal_lower_bound is not None:
                lower_bound_number = question.nominal_lower_bound
            else:
                lower_bound_number = question.lower_bound
            unit_of_measure = question.unit_of_measure
        elif isinstance(question, DateQuestion):
            upper_bound_number = question.upper_bound.date().isoformat()
            lower_bound_number = question.lower_bound.date().isoformat()
            unit_of_measure = ""
        else:
            raise ValueError()

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number} {unit_of_measure}."
        else:
            upper_bound_message = f"The outcome can not be higher than {upper_bound_number} {unit_of_measure}."

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number} {unit_of_measure}."
        else:
            lower_bound_message = f"The outcome can not be lower than {lower_bound_number} {unit_of_measure}."
        return upper_bound_message, lower_bound_message

    ##################################### CONDITIONAL QUESTIONS #####################################

    async def _run_forecast_on_conditional(
        self, question: ConditionalQuestion, research: str
    ) -> ReasonedPrediction[ConditionalPrediction]:
        parent_info, full_research = await self._get_question_prediction_info(
            question.parent, research, "parent"
        )
        child_info, full_research = await self._get_question_prediction_info(
            question.child, research, "child"
        )
        yes_info, full_research = await self._get_question_prediction_info(
            question.question_yes, full_research, "yes"
        )
        no_info, full_research = await self._get_question_prediction_info(
            question.question_no, full_research, "no"
        )
        full_reasoning = clean_indents(
            f"""
            ## Parent Question Reasoning
            {parent_info.reasoning}
            ## Child Question Reasoning
            {child_info.reasoning}
            ## Yes Question Reasoning
            {yes_info.reasoning}
            ## No Question Reasoning
            {no_info.reasoning}
        """
        )
        full_prediction = ConditionalPrediction(
            parent=parent_info.prediction_value,  # type: ignore
            child=child_info.prediction_value,  # type: ignore
            prediction_yes=yes_info.prediction_value,  # type: ignore
            prediction_no=no_info.prediction_value,  # type: ignore
        )
        return ReasonedPrediction(
            reasoning=full_reasoning, prediction_value=full_prediction
        )

    async def _get_question_prediction_info(
        self, question: MetaculusQuestion, research: str, question_type: str
    ) -> tuple[ReasonedPrediction[PredictionTypes | PredictionAffirmed], str]:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        previous_forecasts = question.previous_forecasts
        if (
            question_type in ["parent", "child"]
            and previous_forecasts
            and question_type not in self.force_reforecast_in_conditional
        ):
            # TODO: add option to not affirm current parent/child forecasts, create new forecast
            previous_forecast = previous_forecasts[-1]
            current_utc_time = datetime.now(timezone.utc)
            if (
                previous_forecast.timestamp_end is None
                or previous_forecast.timestamp_end > current_utc_time
            ):
                pretty_value = DataOrganizer.get_readable_prediction(previous_forecast)  # type: ignore
                prediction = ReasonedPrediction(
                    prediction_value=PredictionAffirmed(),
                    reasoning=f"Already existing forecast reaffirmed at {pretty_value}.",
                )
                return (prediction, research)  # type: ignore
        info = await self._make_prediction(question, research)
        full_research = self._add_reasoning_to_research(research, info, question_type)
        return info, full_research  # type: ignore

    def _add_reasoning_to_research(
        self,
        research: str,
        reasoning: ReasonedPrediction[PredictionTypes],
        question_type: str,
    ) -> str:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        question_type = question_type.title()
        return clean_indents(
            f"""
            {research}
            ---
            ## {question_type} Question Information
            You have previously forecasted the {question_type} Question to the value: {DataOrganizer.get_readable_prediction(reasoning.prediction_value)}
            This is relevant information for your current forecast, but it is NOT your current forecast, but previous forecasting information that is relevant to your current forecast.
            The reasoning for the {question_type} Question was as such:
            ```
            {reasoning.reasoning}
            ```
            This is absolutely essential: do NOT use this reasoning to re-forecast the {question_type} question.
            """
        )

    def _get_conditional_disclaimer_if_necessary(
        self, question: MetaculusQuestion
    ) -> str:
        if question.conditional_type not in ["yes", "no"]:
            return ""
        return clean_indents(
            """
            As you are given a conditional question with a parent and child, you are to only forecast the **CHILD** question, given the parent question's resolution.
            You never re-forecast the parent question under any circumstances, but you use probabilistic reasoning, strongly considering the parent question's resolution, to forecast the child question.
            """
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the template forecasting bot")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["seasonal-check", "seasonal", "minibench", "tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="What to forecast on (default: tournament)",
    )
    parser.add_argument("--test-size", type=int, choices=range(1, 11), default=1,
                        help="Number of testing-area questions for a dry run")
    args = parser.parse_args()
    run_mode: Literal["seasonal-check", "seasonal", "minibench", "tournament", "metaculus_cup", "test_questions"] = args.mode

    protected_run = run_mode in {"seasonal", "minibench", "tournament"}
    if protected_run:
        from prize_publication import ProviderAuditHandler, VerifiedPrizeClient, finish_prize_run, open_audit_stream
        client = VerifiedPrizeClient(open_audit_stream())
        logger.addHandler(ProviderAuditHandler(client.audit))
    else:
        client = MetaculusClient()
    if client.get_current_user_id() <= 0:
        raise RuntimeError("Metaculus did not return an authenticated user")
    if protected_run:
        print("Metaculus authentication verified", file=client.audit)
    logger.info("Metaculus authentication verified")
    if run_mode == "seasonal-check":
        questions = client.get_all_open_questions_from_tournament(client.CURRENT_AI_COMPETITION_ID)
        logger.info("FutureEval seasonal monitor: %d open, %d unanswered; publication disabled",
                    len(questions), sum(not q.already_forecasted for q in questions))
        raise SystemExit(0)

    check_environment(strict=True)
    publish_to_metaculus = run_mode != "test_questions"
    print_startup_banner(run_mode, will_publish=publish_to_metaculus)

    free_model = FreeForecastLlm()
    template_bot = FallTemplateBot2026(
        metaculus_client=client,
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=False,
        enable_summarize_research=False,
        publish_reports_to_metaculus=publish_to_metaculus,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        llms={
            "default": free_model,
            "summarizer": free_model,
            "researcher": free_model,
            "parser": free_model,
        },
    )



    # Per-mode tournament URL shown in the summary banner footer. These
    # piggyback on the forecasting_tools SDK constants and need updating
    # whenever those rotate seasons.
    TOURNAMENT_URLS = {
        "seasonal": "https://www.metaculus.com/tournament/fall-futureeval-2026/",
        "minibench": "https://www.metaculus.com/futureeval/minibench/",
        "tournament": "https://www.metaculus.com/tournament/fall-futureeval-2026/",
        "metaculus_cup": "https://www.metaculus.com/tournament/metaculus-cup-fall-2026/",
        "test_questions": "https://www.metaculus.com/tournament/bot-testing-area/",
    }

    # Dispatch on mode. Each branch produces a list of ForecastReport (or
    # exceptions, since return_exceptions=True) which then flows into the
    # summary printers below.
    if run_mode == "seasonal":
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
    elif run_mode == "minibench":
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
        )
    elif run_mode == "tournament":
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
        forecast_reports.extend(asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
        ))
    elif run_mode == "metaculus_cup":
        # The Metaculus Cup may be uninitialized near the start of a season
        # (Jan/May/Sep). MetaculusClient.ACX_2025_TOURNAMENT = 32564 and
        # MetaculusClient.AI_2027_TOURNAMENT_ID = "ai-2027" are also valid
        # targets here.
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        template_bot.skip_previously_forecasted_questions = False  # dry run only
        questions = client.get_all_open_questions_from_tournament("bot-testing-area")
        binary = next((q for q in questions if type(q) is BinaryQuestion), None)
        if binary is None or len(questions) < args.test_size:
            raise RuntimeError("Not enough open testing-area questions for this test")
        sample = [binary, *(q for q in questions if q is not binary)][:args.test_size]
        forecast_reports = []
        # Stop at each checkpoint on error, and process each selected question only once.
        for target in sorted({min(n, len(sample)) for n in (1, 2, 5, 10)}):
            batch = sample[len(forecast_reports):target]
            logger.info("Testing-area checkpoint %d/%d; %d new questions",
                        target, len(sample), len(batch))
            reports = asyncio.run(template_bot.forecast_questions(batch))
            if len(reports) != len(batch) or any(r.errors for r in reports):
                raise RuntimeError("Incomplete testing-area forecast")
            if not forecast_reports:
                first = reports[0]
                if not isinstance(first, BinaryReport) or not 0 < first.prediction < 1:
                    raise RuntimeError("Expected a valid binary probability")
                logger.info("Validated dry run: question=%s probability=%.4f reports=1",
                            first.question.id_of_question, first.prediction)
            forecast_reports.extend(reports)
            logger.info("Validated %d/%d testing-area reports", len(forecast_reports), len(sample))

    if protected_run:
        finish_prize_run(forecast_reports, client)
        raise SystemExit(0)
    template_bot.log_report_summary(forecast_reports)
    print_run_summary_banner(
        forecast_reports,
        will_publish=publish_to_metaculus,
        tournament_url=TOURNAMENT_URLS.get(run_mode),
    )
