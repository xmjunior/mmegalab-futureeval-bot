"""Extract the preregistered resolved cohort from an official Git checkout."""
import argparse
from collections import Counter
from datetime import date, timedelta
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "binary"))
import derive

REVISION = "86b92ba06ed34d606d633bff3e99af54dd447d3a"
CUTOFF = "2026-11-22"
ROUNDS = derive.ROUNDS
TRAIN_PANEL = HERE.parent / "binary" / "rows.jsonl.gz"
PANEL_SHA = "6d7b464141d5c0c5a141df2fdc8b3ec25dcb0ab1f2299950324795c5b6776860"


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def previous_events():
    raw = gzip.decompress(TRAIN_PANEL.read_bytes())
    if digest(raw) != PANEL_SHA:
        raise ValueError("Original panel hash mismatch")
    return {(row[2], row[3], row[4]) for line in raw.splitlines() if (row := json.loads(line))}


def family(source, identifier, item):
    if source == "acled":
        country = re.search(r" in (.+?) for the 30 days before", item["question"].splitlines()[0], re.I)
        return source + "|" + ("country:" + country[1].strip().casefold() if country else "unparsed-acled")
    if source == "dbnomics" and identifier.startswith("meteofrance_TEMPERATURE"):
        return source + "|weather:france"
    return source + "|" + identifier


def assigned_targets(questions):
    result = {}
    for q in questions:
        source, identifier = q["source"], q["id"]
        if source not in derive.SOURCES or not isinstance(identifier, str) or q.get("direction"):
            continue
        group = family(source, identifier, q)
        for target in q["resolution_dates"]:
            key = source, identifier, target[:10]
            if key in result:
                raise ValueError("Duplicate question and date within issue")
            result[key] = group
    return result


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def derive_panel(repo):
    old = previous_events()
    embargo = set()
    emitted = set()
    records, vintages = [], {}
    counts = Counter()
    for due in ROUNDS:
        path = "datasets/question_sets/" + due + "-llm.json"
        before = (date.fromisoformat(due) + timedelta(days=1)).isoformat() + "T00:00:00Z"
        revision = git(repo, "rev-list", "-1", "--before=" + before, REVISION, "--", path).decode().strip()
        if not revision:
            raise ValueError("Missing issue-time version: " + due)
        qraw = git(repo, "show", revision + ":" + path)
        qdoc = json.loads(qraw)
        if qdoc["forecast_due_date"] != due:
            raise ValueError("Mismatched issue")
        metadata = assigned_targets(qdoc["questions"])
        info = {"issue_revision": revision, "issue_sha256": digest(qraw)}
        vintages[due] = info
        if due < "2026-01-04":
            embargo.update(metadata)
            continue
        lraw = git(repo, "show", REVISION + ":datasets/resolution_sets/" + due + "_resolution_set.json")
        ldoc = json.loads(lraw)
        if ldoc["forecast_due_date"] != due:
            raise ValueError("Mismatched label issue")
        labels = {}
        for item in ldoc["resolutions"]:
            if item["source"] not in derive.SOURCES or not isinstance(item["id"], str) or item.get("direction"):
                continue
            key = (item["source"], item["id"], item["resolution_date"][:10])
            if key in labels:
                raise ValueError("Duplicate outcome row")
            labels[key] = item
        info["resolution_sha256"] = digest(lraw)
        stats = Counter()
        for key, group in sorted(metadata.items()):
            source, identifier, target = key
            stats["listed"] += 1
            if not due < target <= CUTOFF:
                stats["outside_scoring_window"] += 1
                continue
            if key in embargo:
                stats["earlier_training_issue"] += 1
                continue
            if key in old:
                stats["prior_archived_event"] += 1
                continue
            if key in emitted:
                stats["later_reissue"] += 1
                continue
            # Assign the earliest issue before inspecting its label.
            emitted.add(key)
            label = labels.get(key, {})
            if label.get("resolved") is not True or label.get("resolved_to") not in (0, 1):
                stats["unresolved_or_nonbinary"] += 1
                continue
            stats["scored"] += 1
            records.append([due, source, identifier, target, group, int(label["resolved_to"])])
        info["counts"] = dict(stats)
        counts.update(stats)
    assert len({tuple(row[1:4]) for row in records}) == len(records)
    assert not ({tuple(row[1:4]) for row in records} & (old | embargo))
    raw = b"".join((json.dumps(row, separators=(",", ":")) + "\n").encode() for row in records)
    return raw, {"protocol_commit":"35fffcdb115e3828a17f131809cb4ffd52327feb",
                 "official_repo":"https://github.com/forecastingresearch/forecastbench-datasets",
                 "resolution_revision": REVISION, "complete_rows_sha256":digest(raw),
                 "columns":["issued","source","id","target","family","y"],
                 "prior_unique_events":len(old),"all_training_issued_events_embargoed":len(embargo),
                 "counts":dict(counts),"rounds":vintages,"licence":derive.LICENSE}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("official_git", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    raw, manifest = derive_panel(args.official_git)
    args.out.mkdir(exist_ok=True, parents=True)
    (args.out / "panel.jsonl.gz").write_bytes(gzip.compress(raw, mtime=0))
    (args.out / "inputs.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(manifest["counts"])
