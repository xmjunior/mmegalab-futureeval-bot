"""Check whether the pinned Market Pulse validator has a prospective rehearsal window."""

from collections import Counter
from datetime import datetime, timedelta, timezone
import os


def rehearsal_windows(market, account, now):
    # The validator intersects its Q4 candidates with this Q3 menu. An empty
    # intersection can never be promoted by this scheduling check.
    groups = market.cross_quarter_menu()
    if not groups or not all(groups.values()):
        raise ValueError("No family qualified in the pinned Q3 scorecard")

    count = 0
    supported = getattr(market, "MARKET_QUESTION_TYPES", {"numeric"})
    for group_id in groups:
        post = account.get(f"/posts/{int(group_id)}/")
        projects = post["projects"]
        if (post["id"] != int(group_id) or projects["default_project"]["id"] != market.PROJECT
                or market.PROJECT not in {p["id"] for p in projects["tournament"]}):
            raise ValueError(f"Unexpected Market Pulse project or post for {group_id}")
        questions = post["group_of_questions"]["questions"]
        if not questions:
            raise ValueError(f"Missing questions for Market Pulse group {group_id}")
        types = Counter(row["type"] for row in questions)
        windows = Counter()
        for row in questions:
            status = row["status"]
            if status in {"closed", "resolved", "annulled"}:
                continue
            if status not in {"open", "upcoming"}:
                raise ValueError(f"Unrecognized Market Pulse status {status!r}")
            deadline = min(market.timestamp(row[field]) for field in ("spot_scoring_time", "scheduled_close_time"))
            if deadline > now + timedelta(minutes=5):
                windows[row["type"]] += 1
        print(f"Project {market.PROJECT} group {group_id}: types={dict(types)} rehearsal_windows={dict(windows)}")
        unsupported = set(types).difference(supported)
        if unsupported:
            raise ValueError(f"Pinned Market Pulse adapter does not support {sorted(unsupported)} in group {group_id}")
        if "discrete" in types and int(group_id) not in getattr(market, "MARKET_DISCRETE_GROUPS", set()):
            raise ValueError(f"Discrete geometry is unqualified for group {group_id}")
        if windows:
            # Production parses every row of the group before selecting a window.
            # Exercise that same read-only contract before any hosted review.
            parser = getattr(market, "parse_market_question", None)
            if parser is not None:
                for row in questions:
                    if parser(post, row) is None:
                        raise ValueError(f"Adapter rejected a question in group {group_id}")
        count += sum(windows.values())
    return count


if __name__ == "__main__":
    import market_pulse

    count = rehearsal_windows(market_pulse, market_pulse.Account(), datetime.now(timezone.utc))
    print(f"Market Pulse Q3-qualified families: {count} open or upcoming rehearsal windows")
    if count == 0:
        print("Waiting for a prospective window. Publication stays locked.")
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        print(f"available={'true' if count else 'false'}", file=output)
