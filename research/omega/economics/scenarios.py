"""Reproduce the 24 Nov 2026 planning scenarios. Offline; never calls a provider."""
from decimal import Decimal as D
import json

PTAX = D("5.8573")
MILLION = D(1_000_000)
STANDARD_PRICES = {
    "sol": (D("4"), D("20")),
    "mini": (D("0.75"), D("4.50")),
    "astra_6_2_sensitivity_only": (D("8"), D("40")),
    "groq_oss_120b_developer_reference_only": (D("0.15"), D("0.60")),
}
PROFILES = {"short": (800, 250), "standard": (3000, 1000), "long": (8000, 2500)}
BUDGETS = (200, 1000, 2000)
BASIC_SEARCH = D("0.008")


def token_cost(arm, input_tokens, output_tokens):
    inp, out = STANDARD_PRICES[arm]
    return (inp * input_tokens + out * output_tokens) / MILLION


def make_report():
    profiles = {}
    for name, (inp, out) in PROFILES.items():
        costs = {arm: token_cost(arm, inp, out) for arm in STANDARD_PRICES}
        profiles[name] = {"input_tokens_per_arm": inp, "output_tokens_per_arm": out,
                          "usd_by_model": {a: str(cost) for a, cost in costs.items()},
                          "usd_paid_pair": str(costs["sol"] + costs["mini"])}
    budgets = []
    mid = D(profiles["standard"]["usd_paid_pair"])
    long = D(profiles["long"]["usd_paid_pair"])
    for brl in BUDGETS:
        usd = D(brl) / PTAX
        stress_usd = D(brl) / (PTAX * D("1.10"))
        stress_pair = long + 3 * BASIC_SEARCH
        attempts = int(stress_usd / stress_pair)
        budgets.append({
            "cap_brl": brl,
            "usd_ptax": str(usd.quantize(D("0.01"))),
            "token_only_paid_pairs_by_profile": {k: int(usd / D(v["usd_paid_pair"]))
                                                 for k, v in profiles.items()},
            "standard_with_three_paid_basic_searches_per_pair": int(usd / (mid + 3 * BASIC_SEARCH)),
            "stress_attempted_pairs": attempts,
            "stress_scorable_pairs_if_nine_in_ten_matched": int(D(attempts) * D("0.9")),
            "single_round_base_questions_standard_max": min(500, int(usd / mid)),
            "single_round_base_questions_stress_max_if_nine_in_ten_matched":
                int(D(min(500, attempts)) * D("0.9")),
        })
    return {"as_of": "2026-11-24", "live_paid_calls": 0, "ptax_brl_per_usd_sell": str(PTAX),
            "profiles": profiles, "budgets": budgets,
            "standard_cost_brl": {str(n): str((n * PTAX * mid).quantize(D("0.01")))
                                  for n in (12, 100, 500, 1000)},
            "gemini_claim_single_sol_calls_at_1cent_to_4cents_R200":
                [int(D(200) / PTAX / cost) for cost in (D("0.04"), D("0.01"))],
            "illustrative_95pct_halfwidth_at_paired_sd_0.10":
                {str(n): round(1.96 * 0.1 / n ** 0.5, 5) for n in (100, 500, 1000)},
            "future_price_policy": "Refresh official price after 2026-11-30; no future price assumed."}


if __name__ == "__main__":
    print(json.dumps(make_report(), indent=2, ensure_ascii=False))
