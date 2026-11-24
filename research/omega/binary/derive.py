"""Build compact, auditable rows from ForecastBench's official archive and historical Git vintages."""
import argparse
import gzip
import hashlib
import json
import math
import re
import tarfile
import urllib.request
from urllib.error import HTTPError
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCHIVE_URL = "https://www.forecastbench.org/assets/data/processed-forecast-sets/processed_forecast_sets.tar.gz"
ARCHIVE_HASH = "3b3b9630f2ccf4da401e703266a116cdc7de535eabb4cb44c9a10d1c01315b0f"
ARCHIVE_NAMES = {"haiku":"Anthropic.claude-haiku-4-5-20251001_zero_shot.json",
                 "mini":"OpenAI.gpt-5-mini-2025-08-07_zero_shot.json",
                 "deepseek":"DeepSeek.DeepSeek-V3.1_zero_shot.json"}
SOURCES = {"acled","dbnomics","fred","wikipedia","yfinance"}
PHASE = [
    ("train", "2026-01-04", "f2fc9404dc32c8b99f902b265aae3182ba16648b", "2026-01-02T04:30:39Z"),
    ("valid", "2026-03-01", "1bd50bae3559460987e695bf32c7854a6d4adb7b", "2026-02-28T06:29:11Z"),
    ("test", "2026-11-24", "86b92ba06ed34d606d633bff3e99af54dd447d3a", "2026-11-22T00:06:38Z")]
ROUNDS = [(date(2025,10,26)+timedelta(days=14*i)).isoformat() for i in range(15)]
LICENSE = "ForecastBench data © Forecasting Research Institute, CC BY-SA 4.0; derived dataset and results under the same license. https://www.forecastbench.org/datasets/ and https://creativecommons.org/licenses/by-sa/4.0/"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_hash(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def stable_id(value):
    return value if isinstance(value,str) else json.dumps(value, separators=(",",":"), sort_keys=True)


def signature(row):
    return (row["source"], stable_id(row["id"]), row["resolution_date"][:10], stable_id(row["direction"] or []))


def fetch_label(pair):
    directory, rd = pair
    phase, cutoff, git, at = next(p for p in PHASE if rd < p[1])
    path = directory/git/(rd+"_resolution_set.json")
    url = f"https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/{git}/datasets/resolution_sets/{path.name}"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(url,timeout=60) as response:
                raw = response.read()
        except HTTPError as exc:
            # Official trees at these historical commits do not yet contain
            # these two rounds. Keep them in coverage; no modern fill-in.
            if exc.code == 404 and rd in {"2025-12-21", "2026-02-15"}:
                return rd, None
            raise
        if json.loads(raw)["forecast_due_date"] != rd:
            raise ValueError("wrong resolution set")
        path.write_bytes(raw)
    return rd, path


def question_metadata(pair):
    directory,rd=pair
    name=rd+"-llm.json"
    path=directory/name
    if not path.exists():
        directory.mkdir(parents=True,exist_ok=True)
        url=f"https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/{PHASE[-1][2]}/datasets/question_sets/{name}"
        with urllib.request.urlopen(url,timeout=60) as response:
            path.write_bytes(response.read())
    raw=path.read_bytes()
    meta=json.loads(raw)
    if meta["forecast_due_date"]!=rd:
        raise ValueError("wrong metadata round")
    group={}
    for item in meta["questions"]:
        src=item["source"]
        ident=stable_id(item["id"])
        if src not in SOURCES:
            continue
        family=ident
        if src=="acled":
            match=re.search(r" in (.+?) for the 30 days before",item["question"].splitlines()[0],re.I)
            family="country:"+match[1].strip().casefold() if match else "unparsed-acled"
        elif src=="dbnomics" and ident.startswith("meteofrance_TEMPERATURE"):
            family="weather:france"
        group[src,ident]=src+"|"+family
    return rd,(group,digest(raw))


def probabilities(archive):
    names = {rd+"."+name:(rd,model) for rd in ROUNDS for model,name in ARCHIVE_NAMES.items()}
    docs = defaultdict(dict)
    hashes = {}
    with tarfile.open(archive,"r|gz") as tar:
        for member in tar:
            name = Path(member.name).name
            if name not in names or not member.isfile():
                continue
            rd, model = names[name]
            raw = tar.extractfile(member).read()
            data = json.loads(raw)
            if data["forecast_due_date"] != rd or data["leaderboard_eligible"] is not True or model in docs[rd]:
                raise ValueError("unexpected forecast metadata")
            rows = {}
            for item in data["forecasts"]:
                key = signature(item)
                if key[0] not in SOURCES:
                    continue
                if key in rows or item["forecast_due_date"][:10] != rd:
                    raise ValueError("duplicate record or round mismatch")
                p = item.get("forecast")
                native = item.get("imputed") is False and type(p) in (int,float) and math.isfinite(p) and 0 <= p <= 1
                rows[key] = p if native else None
            docs[rd][model] = rows
            hashes[name] = digest(raw)
    if len(hashes) != 3*len(ROUNDS):
        raise ValueError("missing base forecast file")
    return docs, hashes


def build(archive, labels):
    if file_hash(archive) != ARCHIVE_HASH:
        raise ValueError("source archive differs from pinned hash")
    docs, hashes = probabilities(archive)
    rows, seen = [],set()
    coverage, label_hashes = {},{}
    for rd in ROUNDS:
        part, cutoff, revision, stamp = next(p for p in PHASE if rd < p[1])
        path = labels[rd]
        raw = path.read_bytes() if path else b""
        doc = json.loads(raw) if path else {"forecast_due_date":rd, "resolutions":[]}
        if doc["forecast_due_date"] != rd:
            raise ValueError("wrong historical resolution file")
        label_hashes[f"{revision}/{rd}_resolution_set.json"] = digest(raw) if path else None
        resolutions = {}
        for item in doc["resolutions"]:
            key = signature(item)
            if key in resolutions:
                raise ValueError("duplicate historical label")
            resolutions[key] = item
        sources = docs[rd]
        counts = Counter()
        for key in sorted(set.union(*(set(a) for a in sources.values()))):
            source, underlying, target, direction = key
            if target <= rd:
                continue
            counts["source_forecast_entries"] += 1
            event = key[:3]
            if event in seen:
                counts["duplicate_event_or_direction"] += 1
                continue
            # Embargo even unavailable, immature or missing first issues to stop
            # recycling an earlier event as a held-out target.
            seen.add(event)
            label = resolutions.get(key,{})
            if target >= cutoff or target > stamp[:10] or not label.get("resolved") or label.get("resolved_to") not in (0,1):
                counts["unavailable_in_vintage"] += 1
                continue
            counts["eligible_first_issue"] += 1
            ps = [sources[model].get(key) for model in ARCHIVE_NAMES]
            for model, p in zip(ARCHIVE_NAMES,ps):
                counts["native_"+model] += int(p is not None)
            if None in ps:
                continue
            counts["paired"] += 1
            rows.append([part,rd,source,underlying,target,direction,*ps,int(label["resolved_to"])])
        coverage[rd] = dict(counts)
    return rows, {"archive_url":ARCHIVE_URL, "archive_sha256":ARCHIVE_HASH, "license":LICENSE,
        "models":ARCHIVE_NAMES, "split_vintages":PHASE, "forecast_sha256":hashes,
        "resolution_sha256":label_hashes, "coverage":coverage,
        "columns":["split","due","source","id","target","direction",*ARCHIVE_NAMES,"y"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("label_cache",type=Path)
    parser.add_argument("--output",type=Path,default=HERE)
    parser.add_argument("--question-cache",type=Path)
    args=parser.parse_args()
    with ThreadPoolExecutor(max_workers=5) as pool:
        paths=dict(pool.map(fetch_label,[(args.label_cache,rd) for rd in ROUNDS]))
        meta=dict(pool.map(question_metadata,[(args.question_cache or args.label_cache/"questions",rd) for rd in ROUNDS]))
    rows,manifest = build(args.archive,paths)
    for row in rows:
        mapping=meta[row[1]][0]
        row.append(mapping[(row[2],row[3])])
    manifest["question_set_sha256"]={rd:info[1] for rd,info in meta.items()}
    manifest["question_metadata_revision"]=PHASE[-1][2]
    manifest["columns"].append("dependence_family")
    args.output.mkdir(parents=True,exist_ok=True)
    payload="".join(json.dumps(r,separators=(",",":"))+"\n" for r in rows).encode()
    out=args.output/"rows.jsonl.gz"
    out.write_bytes(gzip.compress(payload, mtime=0))
    manifest["rows_jsonl_sha256"] = digest(payload)
    (args.output/"inputs.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print("Derived",len(rows),"rows;",out.stat().st_size,"compressed bytes")


if __name__=="__main__":
    main()
