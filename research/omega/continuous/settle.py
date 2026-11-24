"""Incrementally score the sealed gap-tolerant UNRATE panel against ALFRED's first release."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

import download
import evaluate as e

SEAL_SHA = "f8d8999400885bdef8be404e997226c6acac25bb3f26a509b204d43c2b0ee4e3"
INPUT_SHA = "300ab4a9dd46009aa73effe710705697eeeac0549b733ed675dc6c95fd12e2d5"
PROTOCOL_SHA = "6aa6c96631c5540248a3fc2549817e4badace11c3a8f5b94004c31a7e2cbeba2"
# BLS Employment Situation public schedule; wait for ALFRED if its release lags.
RELEASE = {"2026-11-01":"2026-12-04T13:30:00+00:00",
           "2027-01-01":"2027-02-05T13:30:00+00:00",
           "2027-04-01":"2027-05-07T12:30:00+00:00"}


def read_seal():
    for file, expected in (("shadow_gap_20261124.json", SEAL_SHA),
                           ("shadow_gap_protocol.json", PROTOCOL_SHA),
                           ("shadow_input_20261123.csv", INPUT_SHA)):
        if hashlib.sha256((e.HERE/file).read_bytes()).hexdigest()!=expected:
            raise ValueError("The sealed input, protocol or prediction bytes changed")
    seal=json.loads((e.HERE/"shadow_gap_20261124.json").read_text())
    if (seal["source_sha256"]!=INPUT_SHA or seal["prereg_sha256"]!=PROTOCOL_SHA or
            {q["target_month"] for q in seal["questions"]}!=set(RELEASE)):
        raise ValueError("Invalid panel")
    return seal


def first_releases(raw, targets, issued, today):
    result={}
    reader=csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    if reader.fieldnames != ["period_start_date","UNRATE","realtime_start_date","realtime_end_date"]:
        raise ValueError("ALFRED observations format changed")
    for row in reader:
        target=row["period_start_date"]
        if target not in targets or not row["UNRATE"]:
            continue
        observed=row["realtime_start_date"]
        if observed<=issued:
            raise ValueError("Label predates issue")
        if observed<=today:
            y=float(row["UNRATE"])
            if not 0<=y<=100:
                raise ValueError("Invalid observed unemployment rate")
            if target not in result or observed<result[target][0]:
                result[target]=(observed,y)
            elif observed==result[target][0] and y!=result[target][1]:
                raise ValueError("Conflicting first release")
    return result


def advance(state, now, fetch=download.fetch):
    seal=read_seal()
    if state.get("seal_sha256")!=SEAL_SHA or not set(state["cases"]).issubset(RELEASE):
        raise ValueError("State is bound to a different panel")
    due=[q for q in seal["questions"] if q["target_month"] not in state["cases"]
         and now>=datetime.fromisoformat(RELEASE[q["target_month"]])]
    if due:
        raw=fetch(now.date())
        issued=datetime.fromisoformat(seal["issued_utc"]).date().isoformat()
        labels=first_releases(raw,{q["target_month"] for q in due},issued,now.date().isoformat())
        for q in due:
            target=q["target_month"]
            if target not in labels:
                continue
            released, y=labels[target]
            scores={}
            for name,p in q["forecasts"].items():
                dist=e.Distribution(p["center"],p["scale"],p["knots"],p["mix"])
                scores[name]=e.metrics(dist, y, q["threshold"])
            state["cases"][target]=dict(value=y,first_release=released,scores=scores)
    pending=sorted(set(RELEASE)-set(state["cases"]))
    state.update(targets=3, scored=len(state["cases"]), distributions_scored=2*len(state["cases"]),
                 status="waiting_official_publication" if pending else "complete",
                 next_scheduled_release=RELEASE[pending[0]] if pending else None)
    return state


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--state",required=True,type=Path)
    path=parser.parse_args().state
    path.parent.mkdir(parents=True,exist_ok=True)
    # Actions runs on Linux. Lock the existing directory; no extra file is persisted.
    import fcntl
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        write_state(path)
    finally:
        os.close(descriptor)


def write_state(path):
    old=path.read_bytes() if path.exists() else None
    state=json.loads(old) if old else {"seal_sha256":SEAL_SHA,"cases":{}}
    updated=advance(state,datetime.now(timezone.utc))
    data=(json.dumps(updated,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    if data!=old:
        with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:
            tmp=Path(f.name)
            try:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
        try:
            if (path.read_bytes() if path.exists() else None) != old:
                raise ValueError("Concurrent state update; retry with latest state")
            os.replace(tmp,path)
        finally:
            tmp.unlink(missing_ok=True)
    print(json.dumps(updated,sort_keys=True,allow_nan=False))


if __name__ == "__main__":
    main()
