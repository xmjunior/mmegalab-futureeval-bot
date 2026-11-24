"""Single independent Groq forecast, allowed only on a confirmed Free account."""
import argparse
import http.client
import json
import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import datetime, timedelta, timezone

import runner
from source_trials import GROQ_MODEL, fetch


def free_tier():
    # Listing batches cannot start work or incur token charges. The endpoint may
    # return an explicit account-tier denial. A generic 403 or a successful GET
    # says nothing conclusive about the account's billing tier.
    connection = http.client.HTTPSConnection("api.groq.com", timeout=15)
    try:
        connection.request("GET", "/openai/v1/batches?limit=1",
                           headers={"Authorization":"Bearer " + os.environ["GROQ_API_KEY"]})
        response = connection.getresponse()
        if response.status not in (400, 402, 403):
            return False, "read_http_" + str(response.status)
        data = json.loads(response.read(8192))
        error = data.get("error", {})
        message = str(error.get("message","")).lower()
        code = str(error.get("code","")).lower()
        free = (code in {"free_tier_not_supported","free_tier_not_allowed"} or
                bool(re.search(r"(?:your (?:organization|account) is (?:on|in) (?:the )?free (?:tier|plan)|"
                               r"(?:not available|not supported|unavailable) (?:for|on) (?:the )?free (?:tier|plan)|"
                               r"upgrade (?:your (?:account|organization) )?to (?:the )?developer (?:plan|tier))", message)))
        vocab = {"the","this","on","only","tier","tiers","free","paid","developer","plan","plans",
                 "organization","organizations","org","orgs","your","upgrade","upgraded","requires",
                 "required","billing","available","unavailable","batch","api","not","have","access",
                 "please","enabled","endpoint","is","a","for","users","denied","permission",
                 "feature","supported","subscription","to","higher"}
        words = [w for w in re.findall(r"[a-z]+",message) if w in vocab][:45]
        evidence = " ".join(words)
        if re.search(r"free (?:organizations?|orgs?|users)", evidence) and any(
            w in words for w in ("unavailable","not","upgrade")
        ):
            free = True
        return free, ("explicit_free_or_upgrade_required" if free else
                      "indeterminate_http_" + str(response.status) + ": " + evidence)
    finally:
        connection.close()


def groq_answer(text):
    data, _ = fetch("api.groq.com","/openai/v1/chat/completions",
                    {"Authorization":"Bearer " + os.environ["GROQ_API_KEY"]},
                    {"model":GROQ_MODEL, "messages":[{"role":"user","content":text}],
                     "tools":[], "tool_choice":"none", "reasoning_effort":"low",
                     "max_completion_tokens":1200,"service_tier":"on_demand","stream":False})
    choice = data["choices"][0]
    runner.require(data.get("model") == GROQ_MODEL and choice.get("finish_reason") == "stop"
                   and not choice["message"].get("tool_calls"), "unexpected_model_or_finish")
    usage = data["usage"]
    runner.require(all(type(usage.get(k)) is int and usage[k]>=0 for k in ("prompt_tokens","completion_tokens")),
                   "invalid_token_accounting")
    return choice["message"]["content"], usage


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--owner-free-until", default="")
    p.add_argument("--entitlement-only", action="store_true")
    opts = p.parse_args()
    now = runner.utc()
    panel = runner.load_panel()
    deadline = datetime.fromisoformat(panel["issue_deadline_utc"])
    runner.require(now + timedelta(minutes=2) < deadline,"prospective_window_closed")
    # Optional short-lived owner confirmation for the account linked to GROQ_API_KEY.
    # Only set by a reviewed workflow after the owner identifies their actual Free tier.
    if opts.owner_free_until:
        expiry = datetime.fromisoformat(opts.owner_free_until)
        runner.require(now < expiry <= min(now+timedelta(hours=2),deadline),"invalid_confirmation_window")
        eligible, reason = True, "owner_verified_free_tier"
    else:
        eligible, reason = free_tier()
    print(json.dumps({"groq_account_evidence":reason,"eligible":eligible}),flush=True)
    if opts.entitlement_only:
        return
    if not eligible:
        raise SystemExit(2)
    with TemporaryDirectory() as tmp:
        journal = runner.Journal(Path(tmp))
        runner.lease(journal,"groq",None,{"single_panel_request":0})
        group = journal.book["arms"]["groq"]
        try:
            answer, usage = groq_answer(runner.prompt())
            issued = runner.utc()
            runner.require(issued < deadline,"forecast_arrived_late")
            group["records"] = runner.probabilities(answer,panel)
            group["request"] = {"issued_at":issued.isoformat(),"prompt_sha256":runner.PROMPT_SHA,
                                "model":GROQ_MODEL,"output_sha256":runner.sha(answer.encode()),
                                "input_tokens":usage["prompt_tokens"],"output_tokens":usage["completion_tokens"],
                                "free_tier_evidence":reason}
            group["status"] = "completed"
        except Exception as err:
            group["status"] = "stopped_no_retry"
            group["error_type"] = type(err).__name__
        journal.commit("Seal Groq probabilities or retain its nonrepeatable failed lease")
        print(json.dumps({"groq_panel":group["status"]}))
        if group["status"] != "completed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
