"""Closed-only paired trial. Persist each response before attempting the next call."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path

import groq_free
import market_prices as prices
from market_pulse import reviewer_prompt
from capture import HERE, PROTOCOL, sha

LEVELS = prices.LEVELS


def load_inputs():
    path = HERE / "inputs.json"
    if sha(path.read_bytes()) != (HERE / "inputs.sha256").read_text().strip():
        raise ValueError("Inputs differ from the registered digest")
    packets = json.loads(path.read_bytes())
    expected = [str(q) for q in PROTOCOL["historical_controls"]["abstentions"]] + PROTOCOL["reserve_spots_utc"]
    if [p["key"] for p in packets] != expected or len(expected) != len(set(expected)):
        raise ValueError("Wrong trial cohort")
    for p in packets:
        if not datetime.fromisoformat(p["cutoff"]) < datetime.fromisoformat(p["target"]) < datetime.now(timezone.utc):
            raise ValueError("Trial requires closed measurement windows")
        if {k:v for k,v in p["candidate"].items() if k!="evidence"} != p["control"]:
            raise ValueError("Candidate changed control inputs")
        for arm in ("control","candidate"):
            if sha(reviewer_prompt([p[arm]]).encode()) != p["prompt_sha256"][arm]:
                raise ValueError("Input differs from registered prompt")
    return packets


def schedule(packets):
    for packet in packets:
        if packet["group"] == "diagnostic":
            yield packet, "candidate"
    for index, packet in enumerate(p for p in packets if p["group"] == "reserve"):
        arms = ("control", "candidate") if index % 2 == 0 else ("candidate", "control")
        for arm in arms:
            yield packet, arm


def parse(answer):
    text = answer.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    result = json.loads(text)
    if not isinstance(result,list) or len(result)!=1:
        raise ValueError("Wrong number of decisions")
    choice = result[0]
    if choice is not None and (type(choice) not in {int,float} or choice != 1.25):
        raise ValueError("Response outside locked width menu")
    return choice


def save(path, record):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")
    temp.replace(path)


def run_trial(packets, complete, output):
    if output.exists():
        raise FileExistsError(output)
    record = {"status":"running","git_sha":os.getenv("GITHUB_SHA"),"run_id":os.getenv("GITHUB_RUN_ID"),
              "protocol_sha256":sha((HERE/"protocol.json").read_bytes()),"source_input_sha256":sha((HERE/"inputs.json").read_bytes()),
              "model":groq_free.ROUTE,"max_calls":15,"attempted":0,"new_paid_cost_usd":0,"published":0,"rows":[]}
    save(output, record)
    for packet,arm in schedule(packets):
        record["attempted"]+=1
        if record["attempted"]>record["max_calls"]:
            raise AssertionError("Trial exceeded budget")
        row={"key":packet["key"],"arm":arm,"prompt_sha256":packet["prompt_sha256"][arm]}
        try:
            raw,usage=complete(reviewer_prompt([packet[arm]]),max_tokens=512)
            row.update(width=parse(raw), usage=usage)
        except Exception as exc:
            row["error_type"]=type(exc).__name__
            record["rows"].append(row)
            record["status"]="incomplete"
            save(output,record)
            return record
        record["rows"].append(row)
        save(output,record)
    record["status"]="completed"
    save(output,record)
    return record


def pinball(values, y):
    if len(values) != len(LEVELS):
        raise ValueError("Unsupported quantiles")
    return sum(max(p*(y-v),(p-1)*(y-v)) for p,v in zip(LEVELS,values))/len(LEVELS)


def sign_p(deltas):
    nonzero=[d for d in deltas if abs(d)>1e-12]
    n=len(nonzero)
    wins=sum(d>0 for d in nonzero)
    return sum(math.comb(n,i) for i in range(wins,n+1))/2**n


def score(packets, result, labels):
    if result["status"]!="completed" or len(result["rows"])!=15:
        return {"status":"incomplete", "advance":False}
    choices={(row["key"],row["arm"]):row["width"] for row in result["rows"]}
    if choices.keys() != {(packet["key"],arm) for packet,arm in schedule(packets)}:
        raise ValueError("Decision record is missing an arm")
    reserve=[p for p in packets if p["group"]=="reserve"]
    if set(labels)!=set(p["key"] for p in reserve):
        raise ValueError("Reserve settlements differ from the registered cohort")
    records=[]
    for p in reserve:
        key=p["key"]
        y=float(labels[key]["value"])
        row={"key":key}
        for arm in ("control","candidate"):
            chosen=choices[key,arm]
            quantiles=p[arm]["choices"][0]["quantiles"] if chosen is not None else p["naive_quantiles"]
            row[arm]={"covered":chosen is not None,"loss":pinball(quantiles,y)}
        records.append(row)
    deltas=[r["control"]["loss"]-r["candidate"]["loss"] for r in records]
    jointly_covered=[d for d,r in zip(deltas,records) if r["control"]["covered"] and r["candidate"]["covered"]]
    coverage={arm:sum(r[arm]["covered"] for r in records) for arm in ("control","candidate")}
    diagnostics=[p for p in packets if p["group"]=="diagnostic"]
    gains=[]
    for packet in diagnostics:
        h=next(h for h in prices.HISTORY if str(h["question_id"])==packet["key"])
        def mass(values):
            return prices._mass(dict(zip(LEVELS,values)),h["low"],h["high"],h["outcome"],h["bins"])
        pick=choices[packet["key"],"candidate"]
        gains.append(None if pick is None else math.log(mass(packet["candidate"]["choices"][0]["quantiles"])/mass(packet["naive_quantiles"])))
    official=all(g is not None for g in gains) and sum(gains[:-1])>0 and gains[-1]>0
    mean=sum(deltas)/len(deltas)
    prob=sign_p(deltas)
    return {"status":"completed","advance":bool(official and coverage["candidate"]>coverage["control"] and mean>0 and prob<=.05),
            "q4_archived_control":{"covered":0,"n":3,"run_id":45612967315},"q4_candidate":{"covered":sum(g is not None for g in gains),"n":3,"log_gain":gains,"passes_historical_gate":official},
            "reserve":{"n":len(reserve),"coverage":coverage,"mean_loss":{a:sum(r[a]["loss"] for r in records)/len(records) for a in coverage},
                       "paired_mean_loss_reduction":mean,"paired_one_sided_sign_p":prob,
                       "jointly_covered":len(jointly_covered),"jointly_covered_mean_reduction":sum(jointly_covered)/len(jointly_covered) if jointly_covered else None,"cases":records},
            "prospective_proof":False,"publishable":False}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["review","score"])
    p.add_argument("--receipts",type=Path,required=True)
    p.add_argument("--labels",type=Path)
    p.add_argument("--result",type=Path)
    args=p.parse_args()
    packets=load_inputs()
    if args.mode=="review":
        result=run_trial(packets,groq_free.GroqFreeRoute().complete,args.receipts)
        print(json.dumps({k:result[k] for k in ["status","attempted","published"]}))
        raise SystemExit(result["status"]!="completed")
    result=score(packets,json.loads(args.receipts.read_bytes()),json.loads(args.labels.read_bytes()))
    save(args.result,result)
    print(json.dumps({k:result[k] for k in ["status","advance"]}))
