"""Finite operator recovery for the one approved November 24 window; no question content is emitted."""
import base64
import json
import os
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

GH = "https://api.github.com/repos/xmjunior/mmegalab-futureeval-bot"
MC = "https://www.metaculus.com/api"
HEAD = "8d52a632256a34d87df28e9124769fb14e69db55"
MARKET = "199e336d333f74b47b71a1df607cc430107dab39"
SEASON = "run_bot_on_tournament.yaml"
PULSE = "market_pulse.yaml"


def stamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def wait_until(hour):
    target = datetime.fromisoformat("2026-11-24T"+hour+"+00:00").timestamp()
    remaining = target-time.time()
    if remaining>0:
        print("Waiting for",hour,"UTC",flush=True)
        time.sleep(remaining)


def request(base, path, *, data=None):
    uri = base+path
    au = "Token "+os.environ["METACULUS_TOKEN"] if base==MC else "Bearer "+os.environ["GH_TOKEN"]
    for attempt in range(3):
        req = Request(uri,method="POST" if data is not None else "GET", data=json.dumps(data).encode() if data is not None else None,
                      headers={"Authorization":au,"User-Agent":"tournament-operations","Content-Type":"application/json"})
        try:
            with urlopen(req,timeout=20) as response:
                return None if response.status==204 else json.load(response)
        except (TimeoutError, HTTPError, URLError) as exc:
            if data is not None or attempt == 2 or isinstance(exc,HTTPError) and exc.code not in (408,429,500,502,503,504): raise
            retry=exc.headers.get("Retry-After","") if isinstance(exc,HTTPError) else ""
            time.sleep(min(50,int(retry) if retry.isdigit() else 3**attempt))


def runs(file):
    return request(GH,f"/actions/workflows/{file}/runs?per_page=30")["workflow_runs"]


def emit(items):
    print("observed_runs",[(r["id"],r["event"],r["status"],r["conclusion"],r["head_sha"]) for r in items],flush=True)


def dispatch(file):
    request(GH,f"/actions/workflows/{file}/dispatches",data={"ref":"main"})
    print("Accepted one workflow_dispatch for",file,"at",stamp(),flush=True)


def guard_head(head, allow_research_control):
    if head == HEAD:
        return True
    if not allow_research_control:
        return False
    def protected_tree(sha):
        tree_sha = request(GH, "/git/commits/" + sha)["tree"]["sha"]
        response = request(GH, "/git/trees/" + tree_sha + "?recursive=1")
        if response["truncated"]:
            raise ValueError("Cannot establish a complete production tree")
        return {e["path"]: (e["mode"], e["type"], e["sha"]) for e in response["tree"]
                if e["type"] != "tree" and e["path"] != ".github/workflows/champion_control.yaml"}
    return protected_tree(HEAD) == protected_tree(head)


def seasonal(recent_since="2026-11-24T19:03:00Z", allow_research_control=False):
    recent=[r for r in runs(SEASON) if r["status"]!="completed" or r["created_at"]>=recent_since]
    if recent: emit(recent); return
    head = request(GH,"/commits/main")["sha"]
    if not guard_head(head, allow_research_control):
        print("Default branch changed. Skipping seasonal."); return
    if request(GH,"/actions/runs/45467213084")["conclusion"]!="success": raise ValueError("Failed transport gate")
    me=request(MC,"/users/me")
    if me["id"]!=313890 or me["is_primary_bot"] is not True: raise ValueError("Wrong bot identity")
    time.sleep(4)
    rows=request(MC,"/questions/bulk-forecast-read/?project=33121")["results"]
    matches=[r for r in rows if r["question_id"]==46894]
    if len(matches)!=1: raise ValueError("Missing question account membership")
    if matches[0]["forecasts"]:
        print("Already forecast by account. Skipping seasonal."); return
    time.sleep(4)
    q=request(MC,"/questions/46894/")
    open_now=(q["id"]==46894 and q["status"]=="open" and datetime.now(timezone.utc)<datetime.fromisoformat(q["scheduled_close_time"]))
    print("seasonal id=46894 open=",open_now,"own_history_count=0")
    if open_now:
        if request(GH, "/commits/main")["sha"] != head:
            raise ValueError("Main changed while confirming participation")
        latest = [r for r in runs(SEASON) if r["status"] != "completed" or r["created_at"] >= recent_since]
        if latest:
            emit(latest)
            return
        print("Verified active production revision", head)
        dispatch(SEASON)


def market():
    already=[r for r in runs(PULSE) if r["name"]=="Market Pulse 26Q4 spot updates" and
             (r["status"]!="completed" or r["created_at"]>="2026-11-24T19:31:00Z")]
    if already: emit(already); return
    active=[r for r in runs(SEASON) if r["status"]!="completed"]
    if active:
        emit(active); print("Seasonal has priority. Checking its completion once at 19:53 UTC.")
        wait_until("19:53:00")
        if any(r["status"]!="completed" for r in runs(SEASON)):
            print("Seasonal ongoing. Retaining the next native Market Pulse schedule."); return
        if any(r["name"]=="Market Pulse 26Q4 spot updates" and r["created_at"]>="2026-11-24T19:31:00Z" for r in runs(PULSE)):
            print("Market Pulse has already run."); return
    file=request(GH,"/contents/.github/workflows/market_pulse.yaml?ref=main")
    if f'MARKET_CODE_SHA: "{MARKET}"' not in base64.b64decode(file["content"]).decode():
        raise ValueError("Unexpected Market Pulse source pin")
    print("Verified Market Pulse source",MARKET,"branch default head",request(GH,"/commits/main")["sha"])
    dispatch(PULSE)


if __name__=="__main__":
    if not "2026-11-24T19:00" <= stamp() < "2026-11-24T19:54": raise SystemExit("One-off window has expired")
    failed=[]
    for when,fn in [("19:33:02",seasonal),("19:49:02",market)]:
        wait_until(when)
        try: fn()
        except Exception as error:
            print("FAIL",fn.__name__,type(error).__name__,str(error)[:220],flush=True)
            failed.append(fn.__name__)
    if failed: raise RuntimeError(f"Recovery incomplete: {failed}")
