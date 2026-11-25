"""Check whether the pinned Market Pulse validator has a prospective rehearsal window."""

from datetime import datetime, timedelta, timezone
import os


def rehearsal_windows(market, account, now):
    # The validator intersects its Q4 candidates with this Q3 menu. An empty
    # intersection can never be promoted by this scheduling check.
    groups = market.cross_quarter_menu()
    if not groups or not all(groups.values()):
        raise ValueError("No family qualified in the pinned Q3 scorecard")

    count = 0
    for group_id in groups:
        questions = account.group(int(group_id))
        if not questions:
            raise ValueError(f"Missing numeric questions for Market Pulse group {group_id}")
        for question in questions.values():
            row = question.api_json["question"]
            status = row["status"]
            if status in {"closed", "resolved", "annulled"}:
                continue
            if status not in {"open", "upcoming"}:
                raise ValueError(f"Unrecognized Market Pulse status {status!r}")
            deadline = min(market.timestamp(row[field]) for field in ("spot_scoring_time", "scheduled_close_time"))
            if deadline > now + timedelta(minutes=5):
                count += 1
    return count


if __name__ == "__main__":
    import market_pulse

    count = rehearsal_windows(market_pulse, market_pulse.Account(), datetime.now(timezone.utc))
    print(f"Market Pulse Q3-qualified families: {count} open or upcoming rehearsal windows")
    if count == 0:
        print("Waiting for a prospective window. Publication stays locked.")
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        print(f"available={'true' if count else 'false'}", file=output)
