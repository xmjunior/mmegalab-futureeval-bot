"""Bounded fallback for the native tournament schedules. Never creates environments."""

import argparse
import base64
import json
import re
import math
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError

import watchdog

NAME = "forecast-recovery"
WAIT = 50
STOP = datetime(2027, 1, 6, tzinfo=UTC)
MAX_CYCLES = 1420
MAX_FAILURES = 3
CONTROL = "champion_control.yaml"
MARKET = "market_pulse.yaml"
MAIN = {"ref": "main"}
REPO = "xmjunior/mmegalab-futureeval-bot"
ORIGINALS = {
    "Forecast on new AI tournament questions": "run_bot_on_tournament.yaml",
    "Forecast on MiniBench": "run_bot_on_minibench.yaml",
    "Market Pulse 26Q4 spot updates": MARKET,
}
ACTIVE = ("waiting", "pending", "in_progress", "queued", "requested")


def guard(condition, message):
    if not condition:
        raise ValueError(message)


def utc(value):
    parsed = datetime.fromisoformat(value)
    guard(parsed.utcoffset() is not None, "missing timezone")
    return parsed.astimezone(UTC)


def timer_id(environment):
    wait_rules = [
        r.get("wait_timer")
        for r in environment.get("protection_rules", [])
        if r.get("type") == "wait_timer"
    ]
    guard(
        environment.get("name") == NAME
        and type(environment.get("id")) is int
        and environment["id"] > 0,
        "unverified environment",
    )
    guard(wait_rules == [WAIT], "expected a single 50-minute wait timer")
    guard(
        not any(
            rule.get("type") == "required_reviewers"
            for rule in environment["protection_rules"]
        ),
        "environment requires interactive review",
    )
    return environment["id"]


def check_elapsed(environment, expected_id, since, now):
    guard(timer_id(environment) == expected_id, "environment identity changed")
    guard(
        since > 0 and now.timestamp() - since >= WAIT * 60,
        "server wait was not observed; no dispatch permitted",
    )


def market_window(now):
    slots = [
        (now - timedelta(days=back)).replace(hour=h, minute=43, second=0, microsecond=0)
        for back in (0, 1)
        for h in (2, 19)
    ]
    eligible = [t for t in slots if t <= now and t.year == 2026 and t.month in (11, 12)]
    return max(eligible, default=None)


def genuine(run, file, cutoff):
    return (
        run.get("event") == "schedule"
        and run.get("head_branch") == "main"
        and run.get("path") == ".github/workflows/" + file
        and utc(run["created_at"]) >= cutoff
    )


class Dispatcher:
    def __init__(self):
        self.token = os.environ["GH_TOKEN"]

    def github(self, endpoint, **kwargs):
        return watchdog.call(watchdog.GH, endpoint, self.token, **kwargs)

    def environment(self):
        return self.github("/environments/" + NAME)

    def runs(self, file, **params):
        return self.github(
            "/actions/workflows/" + file + "/runs", branch="main", per_page=30, **params
        )["workflow_runs"]

    def send(self, file, inputs=None):
        self.github(
            "/actions/workflows/" + file + "/dispatches",
            method="POST",
            payload={"ref": "main", "inputs": inputs} if inputs else MAIN,
        )

    def evaluation_receipts(self, cutoff):
        rows = self.github(
            "/commits",
            sha="research/champion-observations",
            path="records/control.md",
            since=cutoff.isoformat(),
            per_page=30,
        )
        receipts = []
        if len(rows) >= 30:
            raise RuntimeError("Cannot verify every score update in this window")
        for commit in rows:
            # This file is copied in the same commit as records/control.md by the record job.
            sha = commit["sha"]
            guard(re.fullmatch(r"[0-9a-f]{40}", sha), "invalid score commit")
            receipt_message = re.fullmatch(
                r"Record autonomous forecast checks for run ([1-9][0-9]*)/([1-9][0-9]*)",
                commit["commit"]["message"].strip(),
            )
            if not receipt_message:
                continue
            rid, attempt = receipt_message.groups()
            content = self.github(
                f"/contents/records/runs/{rid}-{attempt}/run.json", ref=sha
            )
            guard(content["encoding"] == "base64", "unexpected score receipt")
            record = json.loads(base64.b64decode(content["content"]))
            guard(
                str(record["GITHUB_RUN_ID"]) == rid
                and str(record["GITHUB_RUN_ATTEMPT"]) == attempt,
                "score receipt identifiers mismatch",
            )
            run = self.github(f"/actions/runs/{rid}/attempts/{attempt}")
            if (
                run.get("head_branch") != "main"
                or run.get("path") != ".github/workflows/" + CONTROL
                or str(run["id"]) != rid
                or run["event"] != record["GITHUB_EVENT_NAME"]
                or run.get("head_sha") != record["GITHUB_SHA"]
                or utc(run["created_at"]) < cutoff
            ):
                continue
            if all(
                record.get(k) == "success"
                for k in ("AUDIT", "PROMO", "TREASURY", "MINI")
            ):
                receipts.append((sha, run))
        return receipts

    def evaluation_attempted(self, cutoff):
        rows = self.github(
            "/actions/workflows/" + CONTROL + "/runs",
            branch="main",
            created=">=" + cutoff.isoformat(),
            per_page=100,
        )["workflow_runs"]
        guard(len(rows) < 100, "Cannot verify all control attempts in this window")
        for run in rows:
            if (
                run.get("path") != ".github/workflows/" + CONTROL
                or run.get("head_branch") != "main"
                or utc(run["created_at"]) < cutoff
            ):
                continue
            title = run.get("display_title")
            # Only this workflow's evaluation modes receive these names. The hourly
            # coverage job and this recovery clock share the workflow id, not the title.
            if run.get("event") == "workflow_dispatch" and title == "Champion control":
                return True
            if (
                run.get("event") == "schedule"
                and title == "Champion schedule 17 7,19 * * *"
                and run.get("status") != "completed"
            ):
                return True
        return False

    def recent_clock_runs(self, now):
        # Usually the first page includes the current chain or its recent seed.
        params = {
            "event": "workflow_dispatch",
            "branch": "main",
            "created": ">=" + (now - timedelta(hours=26)).isoformat(),
            "per_page": 100,
        }
        rows = self.github("/actions/workflows/" + CONTROL + "/runs", **params)[
            "workflow_runs"
        ]
        guard(len(rows) < 100, "ambiguous clock-run history")
        return rows

    def older_active_clock(self):
        for status in ACTIVE:
            rows = self.github(
                "/actions/workflows/" + CONTROL + "/runs",
                event="workflow_dispatch",
                branch="main",
                status=status,
                per_page=100,
            )["workflow_runs"]
            guard(len(rows) < 100, "ambiguous active clock history")
            if any(is_clock(r) and r["status"] != "completed" for r in rows):
                return True
        return False

    def original_coverage(self):
        # Preserve the normal identity, open-window, own-history and concurrency checks.
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("watchdog.py")),
                "--dispatch",
            ],
            check=True,
            timeout=220,
        )

    def market_has_open_candidate(self):
        data = watchdog.call(
            watchdog.MC, "/posts/46010/", os.environ["METACULUS_TOKEN"], with_cp="false"
        )
        guard(
            data["id"] == 46010 and data["projects"]["default_project"]["id"] == 33131,
            "incorrect Market Pulse group",
        )
        return any(
            q["status"] == "open" for q in data["group_of_questions"]["questions"]
        )


def native_healthy(engine, now):
    # Two independent, recent native starts per 20-minute polling workflow. A manual
    # recovery or a cron from another workflow never certifies this scheduler.
    for f in watchdog.TARGETS.values():
        runs = engine.runs(f, event="schedule")
        recent = {r["id"] for r in runs if genuine(r, f, now - timedelta(minutes=85))}
        if len(recent) < 2:
            return False
    if now < datetime(2027, 1, 1, tzinfo=UTC):
        # GitHub may enqueue a scheduled job late; accept the prior slot during the grace period.
        slot = market_window(now - timedelta(minutes=25))
        if slot and not any(
            genuine(r, MARKET, slot) for r in engine.runs(MARKET, event="schedule")
        ):
            return False
    return any(
        run["event"] == "schedule"
        for _, run in engine.evaluation_receipts(control_slot(now))
    )


def control_slot(now):
    # A normal GitHub schedule gets 25 minutes to start before a backup dispatch.
    due = now - timedelta(minutes=25)
    return max(
        t
        for days in (0, 1)
        for h in (7, 19)
        if (
            t := (due - timedelta(days=days)).replace(
                hour=h, minute=17, second=0, microsecond=0
            )
        )
        <= due
    )


def recover_control(engine, now):
    cutoff = control_slot(now)
    receipts = engine.evaluation_receipts(cutoff)
    if receipts:
        return "score already recorded in " + receipts[0][0]
    if engine.evaluation_attempted(cutoff):
        return "evaluation already attempted or active"
    engine.send(CONTROL, {"mode": "control"})
    return "missing semi-daily score dispatched"


def recover_market(engine, now):
    slot = market_window(now)
    if slot is None or not timedelta(minutes=25) <= now - slot <= timedelta(
        minutes=120
    ):
        return "outside slot"

    def already_handled():
        return any(
            r.get("path") == ".github/workflows/" + MARKET
            and r.get("head_branch") == "main"
            and (r["status"] != "completed" or utc(r["created_at"]) >= slot)
            for r in engine.runs(MARKET)
        )

    if already_handled():
        return "already active or attempted in slot"
    if not engine.market_has_open_candidate():
        return "no approved group open"
    if already_handled():
        return "another run observed before dispatch"
    engine.send(MARKET)
    return "original Market Pulse dispatched"


def is_clock(run):
    title = run.get("display_title") or ""
    return (
        run.get("path") == ".github/workflows/" + CONTROL
        and run.get("event") == "workflow_dispatch"
        and run.get("head_branch") == "main"
        and (
            title == "Champion recovery"
            or re.fullmatch(r"Champion recovery cycle [0-9]+ errors [0-9]+", title)
            is not None
        )
    )


def suppressed(engine, now):
    for run in engine.recent_clock_runs(now):
        if not is_clock(run):
            continue
        if run.get("status") != "completed":
            return "an existing clock is active"
        finished = utc(run["updated_at"])
        if (
            run.get("conclusion") in ("failure", "timed_out", "cancelled")
            and (run.get("display_title") or "").endswith(" errors 2")
            and now - finished < timedelta(hours=24)
        ):
            return "24-hour cooldown after repeated clock failures"
        if now - finished < timedelta(minutes=110):
            return "a clock recently completed"
    return "an older clock is still active" if engine.older_active_clock() else None


def trusted_completion(event, now):
    r = event.get("workflow_run") or {}
    if not (
        event.get("action") == "completed"
        and (event.get("repository") or {}).get("full_name") == REPO
        and (r.get("head_repository") or {}).get("full_name") == REPO
        and r.get("head_branch") == "main"
        and r.get("event") in ("schedule", "workflow_dispatch")
        and r.get("status") == "completed"
        and r.get("conclusion") in ("success", "failure", "timed_out")
        and type(r.get("id")) is int
        and r["id"] > 0
        and re.fullmatch(r"[a-f0-9]{40}", r.get("head_sha") or "")
    ):
        return False
    name = ORIGINALS.get(r.get("name"))
    if not name or r.get("path") != ".github/workflows/" + name:
        return False
    age = now - utc(r["updated_at"])
    return timedelta(minutes=-1) <= age <= timedelta(hours=2)


def seed(engine, now, dry=False):
    if now + timedelta(minutes=WAIT) >= STOP:
        return "outside tournament window"
    try:
        ident = timer_id(engine.environment())
    except HTTPError as e:
        if e.code != 404:
            raise
        return "environment not configured; no dispatch"
    except ValueError:
        return "environment has no valid 50-minute rule; no dispatch"
    if native_healthy(engine, now):
        return "native schedules healthy"
    reason = suppressed(engine, now)
    if reason:
        return reason
    if dry:
        return "eligible for one recovery dispatch (read only)"
    guard(timer_id(engine.environment()) == ident, "environment changed before seeding")
    engine.send(
        CONTROL, {"mode": "recovery", "recovery_cycle": "0", "recovery_errors": "0"}
    )
    return "seeded one bounded recovery clock"


def output(**values):
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        for k, v in values.items():
            print(f"{k}={v}", file=f)


def inside_bounds(now, cycle, errors):
    return now < STOP and 0 <= cycle < MAX_CYCLES and 0 <= errors < MAX_FAILURES


def wake(engine, now, clock, cycle, errors, env_id, anchor):
    # Re-read the rule before doing anything on the repository or Metaculus.
    check_elapsed(engine.environment(), env_id, anchor, now)
    if not inside_bounds(now, cycle, errors) or native_healthy(engine, now):
        return "ended"
    failures = []
    try:
        engine.original_coverage()
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        failures.append(type(exc).__name__)
    try:
        print("market", recover_market(engine, now))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        failures.append(type(exc).__name__)
    try:
        print("scores", recover_control(engine, now))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        failures.append(type(exc).__name__)
    new_errors = errors + 1 if failures else 0
    next_time = clock()
    if inside_bounds(
        next_time + timedelta(minutes=WAIT * (2 if new_errors else 1)),
        cycle + 1,
        new_errors,
    ):
        check_elapsed(engine.environment(), env_id, anchor, next_time)
        engine.send(
            CONTROL,
            {
                "mode": "recovery",
                "recovery_cycle": str(cycle + 1),
                "recovery_errors": str(new_errors),
            },
        )
    if failures:
        raise RuntimeError("original workflow checks failed: " + ",".join(failures))
    return "completed"


def main():
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("phase", choices=("preflight", "backoff", "wake", "seed"))
    phase = arg.parse_args().phase
    guard(
        os.environ["GITHUB_REPOSITORY"] == REPO
        and os.environ["GITHUB_REF"] == "refs/heads/main"
        and os.environ["GITHUB_EVENT_NAME"]
        == ("workflow_run" if phase == "seed" else "workflow_dispatch"),
        "unsupported trigger",
    )
    if phase == "seed":
        now = datetime.now(UTC)
        payload = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        if os.environ["GITHUB_RUN_ATTEMPT"] != "1" or not trusted_completion(
            payload, now
        ):
            print("untrusted, stale or repeated completion; no dispatch")
            return
        print(seed(Dispatcher(), now))
        return
    mode = os.environ["RECOVERY_MODE"]
    guard(mode in ("recovery", "recovery_check"), "invalid mode")
    errs = int(os.environ["RECOVERY_ERRORS"])
    cycle = int(os.environ["RECOVERY_CYCLE"])

    def clock():
        return datetime.now(UTC)

    now = clock()
    engine = Dispatcher()
    if phase == "preflight":
        if os.environ["GITHUB_RUN_ATTEMPT"] != "1" or not inside_bounds(
            now, cycle, errs
        ):
            print("closed or re-run; no lease")
            return
        if mode == "recovery_check":
            print(seed(engine, now, dry=True))
            return
        try:
            env_id = timer_id(engine.environment())
        except (ValueError, HTTPError) as e:
            if isinstance(e, HTTPError) and e.code != 404:
                raise
            print(
                "blocked: existing forecast-recovery environment with 50-minute timer required"
            )
            if mode == "recovery":
                raise SystemExit(1)
            return
        print("verified environment", env_id, "wait_minutes", WAIT)
        if mode != "recovery" or native_healthy(engine, now):
            return
        output(
            ready="true",
            environment_id=env_id,
            checked_epoch=math.ceil(clock().timestamp()),
        )
        return
    guard(
        mode == "recovery" and os.environ["GITHUB_RUN_ATTEMPT"] == "1",
        "read-only or rerun",
    )
    env_id = int(os.environ["CHECKED_ENV_ID"])
    anchor = int(os.environ["CHECKED_EPOCH"])
    if phase == "backoff":
        check_elapsed(engine.environment(), env_id, anchor, now)
        output(checked_epoch=math.ceil(clock().timestamp()))
    else:
        print(wake(engine, now, clock, cycle, errs, env_id, anchor))


if __name__ == "__main__":
    main()
