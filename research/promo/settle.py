"""Settle the preregistered Treasury panel. Reads sealed forecasts; never calls a model."""

import argparse
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shadow

HERE = Path(__file__).resolve().parent
PANEL_SHA = "173e2c3ade7fa75d0ee51014e22c25fc33959a48a8610905b0690ca4e4791471"
PROMPT_SHA = "feb6205b0bfd71fb34eeb9e304505cf5dccb6d4dff4d105d1d769e264b7c755a"
REGISTRATION = "4cd8725625727e87e1980aaa726e083a66973e39"
SCHEMA = "ust-promo-settlement-v1"


def require(valid, message):
    if not valid:
        raise ValueError(message)


def unique(pairs):
    obj = dict(pairs)
    require(len(obj) == len(pairs), "duplicate JSON key")
    return obj


def read_json(path):
    return json.loads(path.read_bytes(), object_pairs_hook=unique)


def fingerprint(value):
    return shadow.digest(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )


def sealed_arms(folder, panel, journal):
    cases = panel["cases"]
    start = datetime.fromisoformat(
        "2026-11-24T19:33:50+00:00"
    )  # Git commit of REGISTRATION.
    deadline = datetime.fromisoformat(panel["issue_deadline_utc"])
    arms = {}

    def add(name, values, stamp, prompt_hash):
        require(
            isinstance(name, str)
            and re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]{0,63}", name)
            and name not in arms,
            "duplicate or invalid arm name",
        )
        at = datetime.fromisoformat(stamp)
        require(
            at.tzinfo is not None
            and start <= at <= deadline
            and prompt_hash == PROMPT_SHA,
            "forecast timestamp or prompt mismatch",
        )
        require(
            isinstance(values, dict)
            and values.keys() == cases.keys()
            and all(
                type(p) in (float, int) and math.isfinite(p) and 0 < p < 1
                for p in values.values()
            ),
            "invalid sealed probability panel",
        )
        arms[name] = {
            "probabilities": values,
            "issued_at": at.isoformat(),
            "forecast_sha256": fingerprint(values),
        }

    add(
        "C",
        {k: row["baseline_C"] for k, row in cases.items()},
        start.isoformat(),
        PROMPT_SHA,
    )
    for path in sorted(folder.glob("*.json")):
        data = read_json(path)
        require(data["case_preregistration"] == REGISTRATION, "unregistered panel")
        add(
            data["arm"],
            data["probabilities"],
            data["observed_utc"],
            data["common_prompt_sha256"],
        )
    if journal is not None:
        data = read_json(journal)
        require(
            data["schema"] == "openai-promo-v2" and data["panel_hash"] == PANEL_SHA,
            "unexpected provider journal",
        )
        for name, row in data["arms"].items():
            require(name in {"gemma", "openai", "groq"}, "unexpected provider arm")
            if row["status"] == "completed":
                add(
                    name,
                    row["records"],
                    row["request"]["issued_at"],
                    row["request"]["prompt_sha256"],
                )
    return arms


class OfficialTreasury:
    def __init__(self):
        self.cache = {}

    def observations(self, year):
        if year not in self.cache:
            url = shadow.TREASURY + str(year)
            raw = shadow.request(url)
            days = {}
            for item in ET.fromstring(raw).findall(".//{*}properties"):
                day = date.fromisoformat(item.findtext("{*}NEW_DATE")[:10])
                require(day.year == year, "unexpected year in official document")
                row = {}
                for name, field in (("2y", "BC_2YEAR"), ("10y", "BC_10YEAR")):
                    value = item.findtext("{*}" + field)
                    if not value:
                        continue
                    value = Decimal(value)
                    require(
                        value.is_finite() and -20 <= value <= 100,
                        "out of range Treasury yield",
                    )
                    row[name] = value
                require(
                    day not in days or days[day] == row, "conflicting Treasury rows"
                )
                days[day] = row
            require(bool(days), "empty Treasury feed")
            self.cache[year] = days, shadow.digest(raw)
        return self.cache[year]


def resolution(case, source, now):
    start = date.fromisoformat(case["target"])
    today = now.date()
    last = start + timedelta(days=6)
    if today < start + timedelta(days=2):
        return None
    observations = {}
    hashes = {}
    for year in range(start.year, min(today, last).year + 1):
        rates, hashes[year] = source.observations(year)
        observations.update(rates)
    require(case["leg"] in ("2y", "spread"), "unknown leg")
    for day, row in sorted(observations.items()):
        if not start <= day <= min(last, today - timedelta(days=2)) or "2y" not in row:
            continue
        if case["leg"] == "spread" and "10y" not in row:
            continue
        value = row["2y"] if case["leg"] == "2y" else row["10y"] - row["2y"]
        return {
            "status": "resolved",
            "observed_date": day.isoformat(),
            "value": str(value),
            "threshold": case["threshold"],
            "unit": "percent" if case["leg"] == "2y" else "percentage_points",
            "outcome": int(value > Decimal(case["threshold"])),
            "recorded_utc": now.isoformat(),
            "source_url": shadow.TREASURY + str(day.year),
            "source_sha256": hashes[day.year],
        }
    # A later observation proves the feed advanced beyond the window; with transport
    # errors or a truncated/stale year we cannot declare missing data to be a resolution.
    if today >= last + timedelta(days=2) and max(observations) > last:
        return {
            "status": "void",
            "reason": "no_evaluable_official_observation_in_window",
            "recorded_utc": now.isoformat(),
            "source_sha256_by_year": hashes,
        }
    if today > last + timedelta(days=14) and max(observations) <= last:
        raise RuntimeError(
            "Treasury series did not advance beyond the resolution window"
        )
    return None


def scores(arms, resolved):
    distinct = {
        k: r["outcome"] for k, r in resolved.items() if r["status"] == "resolved"
    }
    result = {}
    for name, arm in arms.items():
        losses = {}
        for key, truth in sorted(distinct.items()):
            p = arm["probabilities"][key]
            losses[key] = {
                "brier": (p - truth) ** 2,
                "log_loss_nats": -math.log(p if truth else 1 - p),
            }
        result[name] = {
            "submitted": len(arm["probabilities"]),
            "paired_scored": len(losses),
            "scores_by_case": losses,
            "mean_brier": sum(v["brier"] for v in losses.values()) / len(losses)
            if losses
            else None,
            "mean_log_loss_nats": sum(v["log_loss_nats"] for v in losses.values())
            / len(losses)
            if losses
            else None,
        }
    return result


def settle(book, panel, arms, source, now):
    require(
        book["schema"] == SCHEMA and book["panel_sha256"] == PANEL_SHA,
        "state does not match preregistration",
    )
    for name, record in arms.items():
        require(
            name not in book["arms"] or book["arms"][name] == record,
            "sealed forecast changed",
        )
    require(set(book["arms"]) <= arms.keys(), "a previously sealed arm is missing")
    book["arms"].update(arms)
    require(
        book["resolutions"].keys() <= panel["cases"].keys(), "unknown question in state"
    )
    for key, case in panel["cases"].items():
        if key in book["resolutions"]:
            continue
        recorded = resolution(case, source, now)
        if recorded:
            book["resolutions"][key] = recorded
    book["scores"] = scores(book["arms"], book["resolutions"])
    count = sum(r["status"] == "resolved" for r in book["resolutions"].values())
    book["coverage"] = {
        "cases": len(panel["cases"]),
        "resolved": count,
        "void": len(book["resolutions"]) - count,
        "pending": len(panel["cases"]) - len(book["resolutions"]),
        "target_dates": len({c["target"] for c in panel["cases"].values()}),
        "correlated_panel": True,
        "promotion_allowed": False,
    }
    book["status"] = "finished" if not book["coverage"]["pending"] else "pending"
    return book


def new_state():
    return {"schema": SCHEMA, "panel_sha256": PANEL_SHA, "arms": {}, "resolutions": {}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument(
        "--journal",
        type=Path,
        help="Read-only provider ledger from research/openai-promo-ledger-v2",
    )
    args = parser.parse_args()
    out = args.state.resolve()
    require(
        out != HERE
        and HERE not in out.parents
        and (args.journal is None or out != args.journal.resolve()),
        "output overlaps sealed input",
    )
    require(
        shadow.digest((HERE / "panel.json").read_bytes()) == PANEL_SHA
        and shadow.digest((HERE / "common-prompt.txt").read_bytes()) == PROMPT_SHA,
        "panel or prompt changed",
    )
    panel = read_json(HERE / "panel.json")
    arms = sealed_arms(HERE / "sealed", panel, args.journal)
    previous = read_json(args.state) if args.state.exists() else None
    book = settle(
        json.loads(json.dumps(previous)) if previous is not None else new_state(),
        panel,
        arms,
        OfficialTreasury(),
        datetime.now(UTC),
    )
    if book != previous:
        shadow.write_ledger(args.state, book)
    print(
        json.dumps(
            {
                "status": book["status"],
                "coverage": book["coverage"],
                "arms": sorted(book["arms"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
