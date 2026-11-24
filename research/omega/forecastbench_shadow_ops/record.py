"""Publish shadow forecasts to the public research ledger and attest their public GitHub time."""
import argparse
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.request import Request, urlopen

REPO = "xmjunior/mmegalab-futureeval-bot"
BRANCH = "research/omega-forecastbench-shadow"
CONTEXT = "megalab/forecastbench-shadow-published"
ROUND = re.compile(r"\d{4}-\d{2}-\d{2}\.json")
FIRST = date(2026, 12, 6)
LAST = date(2027, 3, 31)
UTC = timezone.utc


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def git_text(repo, *args):
    return git(repo, *args).decode().strip()


def save(repo, files, title):
    if not files:
        return
    git(repo, "add", "--", *[str(f.relative_to(repo)) for f in files])
    if git(repo, "diff", "--cached", "--name-only"):
        git(repo, "-c", "user.name=MEGALAB research", "-c",
            "user.email=forecast@users.noreply.github.com", "commit", "-m", title)
        # An ordinary fast-forward push rejects a concurrent branch advance.
        git(repo, "push", "origin", f"HEAD:refs/heads/{BRANCH}")


def write_once(path, obj):
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open("x") as stream:
        stream.write(json.dumps(obj, sort_keys=True, indent=2) + "\n")


def timely_status(row, commit, day):
    timestamp = str(row.get("created_at", ""))
    if not timestamp:
        return False
    when = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    status_url = f"https://api.github.com/repos/{REPO}/statuses/{commit}"
    return (row.get("state") == "success" and row.get("context") == CONTEXT
            and row.get("creator", {}).get("login") == "github-actions[bot]"
            and row.get("url") == status_url
            and re.fullmatch(rf"https://github\.com/{REPO}/actions/runs/[1-9][0-9]*", row.get("target_url") or "")
            and when.utcoffset() == timedelta(0)
            and when.date() == day)


def status_api(commit, payload=None):
    endpoint = f"statuses/{commit}" if payload else f"commits/{commit}/statuses?per_page=100"
    request = Request(f"https://api.github.com/repos/{REPO}/{endpoint}",
                      headers={"Authorization": "Bearer " + os.environ["GH_TOKEN"],
                               "Accept": "application/vnd.github+json",
                               "X-GitHub-Api-Version": "2022-11-28"},
                      data=json.dumps(payload).encode() if payload else None)
    with urlopen(request, timeout=12) as response:
        return json.load(response)


def historical_issue(repo, issue):
    path = str(issue.relative_to(repo))
    changes = git_text(repo, "log", "--format=%H", "--", path).splitlines()
    if len(changes) != 1 or git(repo, "show", changes[0] + ":" + path) != issue.read_bytes():
        raise ValueError(f"Immutable issue changed: {path}")
    return changes[0]


def stamp(repo, issue, now, run_id, api):
    commit = historical_issue(repo, issue)
    day = date.fromisoformat(issue.stem)
    existing = api(commit)
    if not isinstance(existing, list) or len(existing) == 100:
        raise ValueError("Cannot exhaustively verify the commit's statuses")
    valid = [item for item in existing if timely_status(item, commit, day)]
    if not valid and now.date() == day and not any(x.get("context") == CONTEXT for x in existing):
        created = api(commit, {"state": "success", "context": CONTEXT,
                         "description": "Published immutable shadow forecast",
                         "target_url": f"https://github.com/{REPO}/actions/runs/{run_id}"})
        if timely_status(created, commit, day):
            valid.append(created)
    if not valid:
        return None
    row = min(valid, key=lambda s: s["created_at"])
    return {"commit_sha": commit, "issue_sha256": sha256(issue.read_bytes()).hexdigest(),
            "published_at_utc": row["created_at"], "status_id": row["id"],
            "status_url": row["url"], "workflow_run": row["target_url"]}


def issues_in(directory):
    return {p.name: p for p in directory.glob("*.json") if ROUND.fullmatch(p.name)}


def eligible(day):
    return FIRST <= day <= LAST and (day - FIRST).days % 14 == 0


def operate(repo, science_sha, run_id, now, run_science, api=status_api, validate_issue=None):
    if git_text(repo, "branch", "--show-current") != BRANCH or git(repo, "status", "--porcelain"):
        raise ValueError("The ledger requires a clean checkout of the dedicated branch")
    folder = repo / "forecastbench"
    issues = issues_in(folder)
    today = now.date().isoformat() + ".json"
    active = eligible(now.date())
    if not active and not issues:
        print("Outside first issue window, no outstanding shadow rounds.")
        return
    if active:
        lease = folder / "leases" / today
        if not lease.exists():
            write_once(lease, {"round": now.date().isoformat(), "first_run_id": run_id,
                               "science_sha": science_sha})
            save(repo, [lease], "Reserve the ForecastBench shadow round " + today[:-5])
        if json.loads(lease.read_bytes())["science_sha"] != science_sha and today not in issues:
            raise ValueError("The reserved forecast requires its original frozen code")

    before = {name: path.read_bytes() for name, path in issues.items()}
    receipts = folder / "receipts"
    failure = None
    try:
        run_science(folder, receipts)
    except subprocess.SubprocessError as exc:
        # The scientist publishes today's issue atomically before fetching any old outcomes.
        if not active or today in before or not (folder / today).is_file() or validate_issue is None:
            raise
        failure = exc
    after = issues_in(folder)
    if any(name not in after or after[name].read_bytes() != raw for name, raw in before.items()):
        raise ValueError("Scientific code attempted to change a previously published forecast")
    new = set(after) - set(before)
    if new and (not active or new != {today}):
        raise ValueError("Only the eligible current UTC round can be issued")
    if new:
        if validate_issue:
            validate_issue(after[today])
        save(repo, [after[today]], "Seal the ForecastBench shadow round " + today[:-5])

    errors = [f"Scientific settlement failed after an intact issue: {failure}"] if failure else []
    changed_scores = []
    new_receipts = []
    for name, issue in sorted(after.items()):
        proof = receipts / name
        score = folder / "scores" / name
        changed = failure is None and score.is_file() and bool(git(repo, "status", "--porcelain", "--", str(score.relative_to(repo))))
        if proof.exists() and not changed:
            continue
        try:
            receipt = stamp(repo, issue, now, run_id, api)
            if receipt is None:
                print(f"{name}: no eligible GitHub attestation")
                if changed:
                    raise ValueError("Score without a timely GitHub receipt")
                continue
            if proof.exists():
                if receipt != json.loads(proof.read_bytes()):
                    raise ValueError("Receipt differs from the GitHub attestation")
            else:
                write_once(proof, receipt)
                new_receipts.append(proof)
            if changed:
                changed_scores.append(score)
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    stray = set(issues_in(folder / "scores")) - set(after)
    if stray:
        errors.append("Scores for unknown issue dates")
    save(repo, new_receipts + changed_scores, "Record verified shadow receipts and official settlements")
    if errors:
        raise ValueError("; ".join(errors))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--driver", type=Path, required=True)
    args = parser.parse_args()
    env = os.environ
    if env.get("GITHUB_REPOSITORY") != REPO:
        raise ValueError("Unexpected repository")
    driver = args.driver.resolve()
    science_sha = env["SCIENCE_SHA"]
    if git_text(driver.parent, "rev-parse", "HEAD") != science_sha:
        raise ValueError("The research checkout differs from its frozen SHA")

    def run_science(directory, receipts):
        public = {k: v for k, v in env.items() if k not in {"GH_TOKEN", "GITHUB_TOKEN"}}
        subprocess.run([sys.executable, str(driver), "run", "--state-dir", str(directory),
                        "--receipts-dir", str(receipts)], check=True, env=public, timeout=135)

    sys.path.insert(0, str(driver.parent))
    from shadow import read_seal
    operate(args.ledger.resolve(), science_sha, env["GITHUB_RUN_ID"], datetime.now(UTC),
            run_science, validate_issue=read_seal)


if __name__ == "__main__":
    main()
