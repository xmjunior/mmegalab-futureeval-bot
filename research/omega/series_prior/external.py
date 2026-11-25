"""Read only the preregistered external forecasts matching the closed cohort."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile

from data import HERE, ROUNDS, derive
from evaluate import read_panel

MODEL = derive.ARCHIVE_NAMES["mini"]


def extract(archive):
    selected = {tuple(row[:4]) for row in read_panel()}
    expected = json.loads((HERE.parent / "binary" / "inputs.json").read_bytes())["forecast_sha256"]
    members = {d+"."+MODEL: d for d in ROUNDS if d >= "2026-01-04"}
    hashes, matched = {}, {}
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            name = Path(member.name).name
            if name not in members:
                continue
            due = members[name]
            if due in hashes:
                raise ValueError("Duplicate forecast member")
            raw = tar.extractfile(member).read()
            hashes[due] = hashlib.sha256(raw).hexdigest()
            if hashes[due] != expected[name]:
                raise ValueError("Forecast revision differs from original archive: "+due)
            doc = json.loads(raw)
            if doc["forecast_due_date"] != due or doc["leaderboard_eligible"] is not True:
                raise ValueError("Mismatched forecast issue")
            for r in doc["forecasts"]:
                if not isinstance(r["id"], str) or r.get("direction"):
                    continue
                key = due, r["source"], r["id"], r["resolution_date"][:10]
                if key not in selected:
                    continue
                p = r.get("forecast")
                if r.get("imputed") is not False or type(p) not in (float, int):
                    continue
                if r["forecast_due_date"][:10] != due or not math.isfinite(p) or not 0 <= p <= 1:
                    raise ValueError("Invalid archived probability")
                if key in matched:
                    raise ValueError("Duplicate archived forecast for a scored event")
                matched[key] = p
    if set(hashes) != set(members.values()):
        raise ValueError("Missing expected model results")
    return {"archive_url":derive.ARCHIVE_URL, "model":MODEL,"forecasts_sha256":hashes,
            "rows":[[*key,p] for key,p in sorted(matched.items())]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    doc = extract(args.archive)
    args.out.write_text(json.dumps(doc, separators=(",", ":"), allow_nan=False)+"\n")
    print(f"Verified {len(doc['forecasts_sha256'])} original files and {len(doc['rows'])} matched probabilities")
