"""Bounded, prospective four-case experiment. Prerequisites are checked before I/O."""
import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import uuid
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shadow

PANEL = Path(__file__).with_name("panel.json")
PANEL_SHA = "173e2c3ade7fa75d0ee51014e22c25fc33959a48a8610905b0690ca4e4791471"
# Pin only an independently checked, redacted owner-dashboard audit. Empty means denied.
# No environment switch, model response or self-assertion can enable spending.
BILLING_AUDIT_SHA = None
COMMON = Path(__file__).with_name("common-prompt.txt")
PROMPT_SHA = "feb6205b0bfd71fb34eeb9e304505cf5dccb6d4dff4d105d1d769e264b7c755a"
MODEL = "gpt-5.6-sol"
REPO = "https://github.com/xmjunior/mmegalab-futureeval-bot.git"
STATE = "research/openai-promo-ledger-v2"
OAI = "https://api.openai.com/v1/responses"
LIMIT_MICRO_USD = 5_500_000
MAX_INPUT = 2048
MAX_OUTPUT = 448
# Conservative ceiling for this global, standard-tier request. Official price is
# $4/M input, $20/M output. Reserve $20/M on *every* input too, including cache writes.
RATE = 20
SCHEMA = "openai-promo-v2"
ARMS = ("gemma", "openai")


def utc():
    return datetime.now(timezone.utc)


def require(ok, code):
    if not ok:
        raise ValueError(code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_panel():
    raw = PANEL.read_bytes()
    require(sha(raw) == PANEL_SHA, "panel_hash_changed")
    return json.loads(raw)


def prompt():
    raw = COMMON.read_bytes()
    require(sha(raw) == PROMPT_SHA, "common_prompt_hash_changed")
    return raw.decode()


def probabilities(answer, panel):
    def unique(pairs):
        out = dict(pairs)
        require(len(out) == len(pairs), "duplicate_case_id")
        return out
    text = answer.strip()
    fenced = re.fullmatch(r"`{3}(?:json)?[ \t]*\r?\n(.*?)\r?\n`{3}", text, re.S | re.I)
    if fenced:
        text = fenced[1]
    values = json.loads(text, object_pairs_hook=unique)
    require(isinstance(values, dict) and values.keys() == panel["cases"].keys() and
            all(type(v) in {int, float} and 0 < v < 1 for v in values.values()), "invalid_probability_panel")
    return values


def audited_grant(raw, expected_sha, now):
    require(expected_sha is not None and raw is not None and sha(raw) == expected_sha,
            "BLOCKED_NO_VERIFIED_GRANT_ISOLATION")
    p = json.loads(raw)
    require(p["source"] == "OpenAI owner billing dashboard" and p["active_positive_promotions"] == 1
            and (p["cash_fallback_available"] is False or p.get("other_positive_balance_verified_purchased") is True)
            and p["other_org_usage_isolated"] is True,
            "BLOCKED_GRANT_NOT_ISOLATED")
    require(p["currency"] == "USD" and type(p["grant_remaining_micro_usd"]) is int and p["grant_remaining_micro_usd"] > 0
            and p["org"].startswith("org-") and p["project"].startswith("proj_")
            and bool(p["target_grant_reference"]), "invalid_grant_scope")
    expires = datetime.fromisoformat(p["grant_expires_at"])
    checked = datetime.fromisoformat(p["checked_at"])
    require(expires.date().isoformat() == "2026-11-26" and now < expires and
            checked <= now < checked + timedelta(minutes=30), "grant_evidence_expired")
    return p


def git(folder, *args):
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    token = env.get("GITHUB_TOKEN")
    if token:
        count = int(env.get("GIT_CONFIG_COUNT", "0"))
        auth = base64.b64encode(("x-access-token:" + token).encode()).decode()
        settings = [(f"http.{REPO}.extraHeader", "AUTHORIZATION: basic " + auth),
                    (f"http.{REPO}.followRedirects", "false")]
        for ix, (key, value) in enumerate(settings, count):
            env[f"GIT_CONFIG_KEY_{ix}"] = key
            env[f"GIT_CONFIG_VALUE_{ix}"] = value
        env["GIT_CONFIG_COUNT"] = str(count + len(settings))
    p = subprocess.run(["git", "-C", str(folder), *args], capture_output=True, text=True, env=env)
    require(p.returncode == 0, "durable_state_unavailable_or_conflict")
    return p.stdout.strip()


class Journal:
    """Compare-and-swap a shared run lease onto an existing, dedicated Git ref."""
    def __init__(self, path, remote=REPO):
        self.path = path
        self.remote = remote
        git(path, "init", "--quiet")
        git(path, "remote", "add", "origin", remote)
        git(path, "fetch", "--quiet", "origin", "refs/heads/" + STATE)
        self.head = git(path, "rev-parse", "FETCH_HEAD")
        git(path, "checkout", "--quiet", "--detach", self.head)
        self.file = path / "promo-ledger.json"
        self.book = json.loads(self.file.read_bytes())
        require(self.book["schema"] == SCHEMA and self.book["panel_hash"] == PANEL_SHA, "ledger_mismatch")
        require(isinstance(self.book["arms"], dict), "invalid_ledger")

    def commit(self, message):
        self.file.write_text(json.dumps(self.book, sort_keys=True, indent=2, allow_nan=False) + "\n")
        git(self.path, "add", "--", "promo-ledger.json")
        git(self.path, "-c", "user.name=Forecast Research", "-c", "user.email=forecast@users.noreply.github.com",
            "commit", "-m", message)
        # Fails on any parallel writer, including a stale check-out. Never retry an inference.
        git(self.path, "push", "--quiet", "--force-with-lease=refs/heads/" + STATE + ":" + self.head,
            "origin", "HEAD:refs/heads/" + STATE)
        self.head = git(self.path, "rev-parse", "HEAD")
        remote_head = git(self.path, "ls-remote", "--heads", "origin", STATE).split()[0]
        require(remote_head == self.head, "reservation_not_confirmed")


def plan(cap):
    require(type(cap) is int and 0 < cap <= MAX_INPUT, "input_over_cap")
    return (cap + MAX_OUTPUT) * RATE


def lease(journal, arm, source, estimates):
    book = journal.book
    require(arm not in book["arms"], "duplicate_arm_lease")
    require(not any(v["status"] == "reserved" for v in book["arms"].values()), "prior_run_in_flight_or_ambiguous")
    total = sum(v["max_micro_usd"] for v in book["arms"].values())
    permitted = min(LIMIT_MICRO_USD, source["grant_remaining_micro_usd"]) if arm == "openai" else LIMIT_MICRO_USD
    reserved = sum(estimates.values())
    # Retain one entire additional request reserve if the organization has paid fallback.
    # This buffer is credit availability only; it never raises the authorized spend ceiling.
    buffer = reserved if (arm == "openai" and source.get("cash_fallback_available")) else 0
    require(total + reserved <= permitted and
            (arm != "openai" or total + reserved + buffer <= source["grant_remaining_micro_usd"]),
            "insufficient_protected_credit")
    book["arms"][arm] = {"lease_nonce": uuid.uuid4().hex, "status": "reserved", "max_micro_usd": reserved, "requests": estimates,
                         "started_at": utc().isoformat(), "records": {}}
    journal.commit("Reserve entire panel before external model requests")


def oai(path, params, scope):
    headers = {"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
               "OpenAI-Organization": scope["org"], "OpenAI-Project": scope["project"]}
    return json.loads(shadow.request(OAI + path, params, headers, timeout=50))


def prepare_openai(text, grant):
    public = shadow.request("https://developers.openai.com/api/docs/models/gpt-5.6-sol.md", timeout=15).decode()
    require("| Input | $4 | 1M tokens |" in public and "| Output | $20 | 1M tokens |" in public,
            "published_pricing_changed")
    return count_input(text, grant)


def count_input(text, scope=None):
    inp = {"model": MODEL, "input": text, "tools": []}
    if scope is None:
        headers = {"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
        count = json.loads(shadow.request(OAI + "/input_tokens", inp, headers, timeout=30))
    else:
        count = oai("/input_tokens", inp, scope)
    require(count.get("object") == "response.input_tokens", "invalid_input_count")
    n = count["input_tokens"]
    return inp, n, plan(n)


def try_openai(inp, expected, scope):
    body = {**inp, "reasoning": {"effort":"low"}, "max_output_tokens": MAX_OUTPUT, "store": False,
            "tool_choice":"none", "service_tier":"default", "truncation":"disabled"}
    data = oai("", body, scope)
    raw_usage = data.get("usage") or {}
    receipt = {"billing_reconciliation":"pending_grant_level_statement"}
    identity = data.get("model")
    if isinstance(identity,str) and re.fullmatch(r"[A-Za-z0-9._:-]{1,120}",identity):
        receipt["model"] = identity
    if isinstance(data.get("id"), str):
        receipt["response_id_sha256"] = sha(data["id"].encode())
    for key in ("input_tokens","output_tokens"):
        v = raw_usage.get(key)
        if type(v) is int and v >= 0:
            receipt[key] = v
    rt = (raw_usage.get("output_tokens_details") or {}).get("reasoning_tokens")
    if type(rt) is int and rt >= 0:
        receipt["reasoning_tokens"] = rt
    status = data.get("status")
    if status in {"completed", "failed", "incomplete", "in_progress", "cancelled", "queued"}:
        receipt["provider_status"] = status
    try:
        require(identity and re.fullmatch(re.escape(MODEL) + r"(?:-20[0-9]{2}-[0-9]{2}-[0-9]{2})?", identity)
                and data.get("service_tier") == "default", "unexpected_model_tier_or_status")
        ni, no = receipt.get("input_tokens"), receipt.get("output_tokens")
        require(type(ni) is int and type(no) is int and ni <= expected and no <= MAX_OUTPUT,
                "provider_usage_outside_reservation")
        receipt["usage_cost_upper_micro_usd"] = (ni + no)*RATE
        require(status == "completed", "provider_incomplete_or_failed")
        require(all(m["type"] in {"reasoning","message"} for m in data["output"]), "unexpected_tool_call")
        text = "\n".join(c["text"] for m in data["output"] if m["type"] == "message"
                         for c in m["content"] if c["type"] == "output_text")
        require(text.strip(), "empty_output")
    except (ValueError, KeyError, TypeError) as exc:
        return None, {**receipt, "status":shadow.safe_status(exc)}
    return text, receipt

def run(arm, journal, panel, scope=None):
    require(arm in ARMS, "unknown_arm")
    deadline = datetime.fromisoformat(panel["issue_deadline_utc"])
    require(utc() + timedelta(minutes=2) < deadline, "prospective_window_closed")
    require(arm not in journal.book["arms"], "duplicate_arm_lease")
    text = prompt()
    if arm == "openai":
        inp, count, reserve = prepare_openai(text, scope)
    else:
        reserve = 0
    lease(journal,arm,scope,{"single_panel_request":reserve})
    group = journal.book["arms"][arm]
    if utc() + timedelta(seconds=65) >= deadline:
        group["status"] = "deadline"
    elif scope and utc() >= min(datetime.fromisoformat(scope["checked_at"]) + timedelta(minutes=30),
                                 datetime.fromisoformat(scope["grant_expires_at"])) - timedelta(seconds=65):
        group["status"] = "grant_check_expired"
    else:
        try:
            if arm == "openai":
                answer, meta = try_openai(inp,count,scope)
            else:
                answer,ver,usage = shadow.Routes().call(shadow.GEMMAS[0],text)
                meta = {"model":ver,"usage_cost_upper_micro_usd":0,
                        "input_tokens":usage.get("promptTokenCount"),"output_tokens":usage.get("candidatesTokenCount")}
            at = utc()
            group["request"] = {**meta, "received_at":at.isoformat(),"prompt_sha256":PROMPT_SHA}
            require(at < deadline, "forecast_arrived_late")
            require(isinstance(answer,str),meta.get("status","empty_output"))
            group["request"]["output_sha256"] = sha(answer.encode())
            group["records"] = probabilities(answer,panel)
            group["request"]["issued_at"] = at.isoformat()
            group["status"] = "completed"
        except (OSError, ValueError, TypeError, KeyError, IndexError, ArithmeticError) as exc:
            group.setdefault("request", {}).update(status=shadow.safe_status(exc), request_ambiguous=True)
            group["status"] = "stopped_no_retry"
    journal.commit("Record first panel forecasts and outcomes of provider requests")
    return group["status"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("dry","count","gemma","openai"))
    a = p.parse_args()
    panel = load_panel()
    if a.action == "count":
        require(bool(os.getenv("OPENAI_API_KEY")), "missing_openai_key")
        _, n, reserve = count_input(prompt())
        print(json.dumps({"model":MODEL, "cases":len(panel["cases"]), "input_tokens":n,
                          "max_output_tokens":MAX_OUTPUT, "reservation_micro_usd":reserve,
                          "prompt_sha256":PROMPT_SHA, "inference_requests":0, "billed_usd":0}))
        return
    if a.action == "dry":
        print(json.dumps({"panel_sha":PANEL_SHA, "cases":len(panel["cases"]), "model":MODEL,
                          "panel_reservation_upper_usd":plan(MAX_INPUT)/1e6,
                          "inference_requests":1,
                          "openai_gate":"blocked" if BILLING_AUDIT_SHA is None else "audit_required",
                          "prompt_sha256":sha(prompt().encode())}))
        return
    scope = None
    if a.action == "openai":
        audit_file = os.getenv("OPENAI_PROMO_VERIFIED_AUDIT")
        raw = Path(audit_file).read_bytes() if audit_file and BILLING_AUDIT_SHA else None
        scope = audited_grant(raw,BILLING_AUDIT_SHA,utc())
        require(bool(os.getenv("OPENAI_API_KEY")), "missing_openai_key")
    else:
        require(bool(os.getenv("GEMINI_API_KEY")), "missing_google_key")
    with TemporaryDirectory() as folder:
        status = run(a.action,Journal(Path(folder)),panel,scope)
    print(json.dumps({"arm":a.action,"status":status}))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        print(json.dumps({"status":"blocked","reason": str(exc) if isinstance(exc, ValueError) and
                          str(exc).replace("_","").isalnum() else type(exc).__name__}))
        raise SystemExit(2)
