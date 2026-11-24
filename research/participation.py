"""Account operations audit. GET only. Never persist question text, predictions, comments, or public aggregates."""
import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

BASE = "https://www.metaculus.com/api"
BOT = "megalab-mario-bot"
UTC = timezone.utc


def read(path, **params):
    url = BASE + path + ("?" + urlencode(params) if params else "")
    for attempt in range(4):
        time.sleep(3.7)
        try:
            with urlopen(Request(url, method="GET", headers={
                "Authorization": "Token " + os.environ["METACULUS_TOKEN"],
                "Accept-Language": "en", "User-Agent": "FutureEval-participation-audit"
            }), timeout=35) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError) as exc:
            if attempt == 3 or isinstance(exc, HTTPError) and exc.code not in (408,429,500,502,503,504):
                raise
            delay = int(exc.headers.get("Retry-After", "60")) if isinstance(exc, HTTPError) and exc.code == 429 else 2 ** attempt
            time.sleep(min(120, max(1,delay)))


def pages(path, **filters):
    offset = 0
    while True:
        filters.update(limit=100, offset=offset)
        page = read(path, **filters)
        items = page["results"]
        yield from items
        if not page.get("next") or not items:
            return
        offset += len(items)
        if offset > 5000:
            raise ValueError("Pagination guard reached")


def utc(value):
    if value is None:
        return None
    if isinstance(value, (float,int)):
        return datetime.fromtimestamp(value,tz=UTC).isoformat()
    return datetime.fromisoformat(value.replace("Z","+00:00")).astimezone(UTC).isoformat()


def extract(post):
    # Only whitelisted metadata leaves this function. Do not inspect the question titles, values or text.
    if post.get("group_of_questions"):
        # A list response can truncate the individual subquestions. Request the full group without CP.
        post = read(f"/posts/{post['id']}/", with_cp="false")
        qs = post["group_of_questions"]["questions"]
    elif post.get("question"):
        qs = [post["question"]]
    elif post.get("conditional"):
        qs = [q for q in post["conditional"].values() if isinstance(q,dict) and q.get("id") and "my_forecasts" in q]
    else:
        return []
    out = []
    for q in qs:
        mf = q.get("my_forecasts")
        history = mf.get("history") if isinstance(mf,dict) else None
        entry = {"id":q["id"], "post":post["id"], "state":q["status"], "opens":utc(q.get("open_time")),
                 "closes":utc(q.get("actual_close_time") or q.get("scheduled_close_time")),
                 "own_count":len(history) if isinstance(history,list) else None}
        if history:
            entry["own_times"] = [utc(f["start_time"]) for f in history]
            entry["resolved_own_scores"] = ({k:v for k,v in mf.get("score_data",{}).items()
                                            if isinstance(v,(float,int)) and "score" in k}
                                           if q["status"]=="resolved" else {})
        out.append(entry)
    return out


def audit(slug, key, userid, created, now):
    project = read(f"/projects/tournaments/{slug}/")
    if "id" not in project:
        raise ValueError(f"Project identifier missing: {list(project)}")
    pid = project["id"]
    questions = {}
    filtered_post_ids = {p["id"] for p in pages("/posts/",tournaments=pid,forecaster_id=userid,
                                                include_descriptions="false",with_cp="false")}
    for p in pages("/posts/", tournaments=pid, include_descriptions="false", with_cp="false"):
        for q in extract(p):
            questions[q["id"]] = q
    # This endpoint returns only the token owner's authoritative timestamped forecasts.
    # Read only the identifiers and timestamps; never log or store forecast distributions.
    response = read("/questions/bulk-forecast-read/", project=pid)
    if not isinstance(response.get("results"),list):
        raise RuntimeError("Authenticated forecast history unavailable")
    history = {r["question_id"]:[utc(f["start_time"]) for f in r["forecasts"]] for r in response["results"]}
    for q in questions.values():
        if q["id"] in history:
            q["own_count"] = len(history[q["id"]])
            if history[q["id"]]:
                q["own_times"] = history[q["id"]]
    missing_history = [q["id"] for q in questions.values() if q["id"] not in history]
    missing_own = {k:v for k,v in history.items() if v and k not in questions}
    own = [q for q in questions.values() if q["own_count"]]
    mismatch = sorted(filtered_post_ids ^ {q["post"] for q in own})
    counts = Counter(q["state"] for q in questions.values())
    comments = []
    for post_id in {q["post"] for q in own} | filtered_post_ids:
        for private in ("true","false"):
            for c in pages("/comments/",post=post_id,author=userid,is_private=private):
                if c.get("author_id",userid) != userid:
                    raise ValueError("Incorrect author in filtered results")
                comments.append({"id":c["id"], "created_at":utc(c.get("created_at")), "post":post_id,
                                 "is_private":private == "true",
                                 "included_forecast":bool(c.get("included_forecast"))})
    boards = read(f"/leaderboards/project/{pid}/",primary_only="true",with_entries="true")
    ranking = []
    for b in boards:
        row = b.get("userEntry")
        ranking.append({"leaderboard":b.get("id"),"score_type":b.get("score_type"),"ranked_entries":len(b.get("entries",[])),
                        "coverage_max":b.get("max_coverage"),"own_entry":None if row is None else
                        {k:row.get(k) for k in ("rank","score","coverage","exclusion_status","prize","percent_prize")}})
    open_now = [q for q in questions.values() if q["state"]=="open"]
    next_times = sorted({q["opens"] for q in questions.values() if q["state"]=="upcoming" and q["opens"] and q["opens"]>now})
    since = [q for q in questions.values() if q["opens"] and q["opens"]<=now and q["closes"] and q["closes"]>=created]
    return {"own_filter_post_ids": sorted(filtered_post_ids)[:20], "own_filter_post_total":len(filtered_post_ids), "mismatching_post_ids":mismatch,"question_history_missing_count":len(missing_history),"own_history_without_post":missing_own,
            "project":pid,"slug":slug,"canonical_slug":project.get("slug"),"dates":{k:project.get(k) for k in ("start_date","close_date","forecasting_end_date")},
            "states":dict(counts),"questions":len(questions),"forecasted_questions":len(own),
            "forecast_events":sum(q["own_count"] for q in own),"duplicate_questions":[q["id"] for q in own if q["own_count"]>1],
            "opportunities_overlapping_account_lifetime":len(since),"unanswered_current":sum(not q["own_count"] for q in open_now),
            "current_windows":[{k:q[k] for k in ("id","opens","closes","own_count")} for q in open_now],
            "next_visible_open":next_times[:1],"submitted":own,"comments":comments,"leaderboards":ranking}


me = read("/users/me")
if me.get("username")!=BOT or me.get("is_bot") is not True: raise RuntimeError("Wrong account")
created = utc(me.get("date_joined"))
now = datetime.now(UTC).isoformat()
out = {"observed_at":now,"account":{k:me.get(k) for k in ("id","username","is_bot","is_primary_bot","date_joined")},
       "projects":{}}
args = argparse.ArgumentParser()
args.add_argument("--prior-minibench-file", type=Path, help="JSON list of previously observed MiniBench slugs")
opts = args.parse_args()
projects = [("FutureEval","fall-futureeval-2026"),("MiniBench","minibench"),("MarketPulse","market-pulse-26q4")]
if opts.prior_minibench_file:
    import re
    slugs = json.loads(opts.prior_minibench_file.read_text())
    if not isinstance(slugs,list) or len(slugs)>10 or not all(isinstance(s,str) and re.fullmatch(r"minibench-[a-z0-9-]{4,60}",s) for s in slugs):
        raise ValueError("Invalid historical MiniBench list")
    projects += [(s,s) for s in sorted(set(slugs))]
for name,slug in projects:
    try:
        out["projects"][name]=audit(slug,name,me["id"],created or now,now)
    except (HTTPError,URLError,TimeoutError,ValueError,RuntimeError,KeyError,TypeError) as e:
        out["projects"][name]={"error":f"{type(e).__name__}: {str(e)[:280]}"}
    print("AUDIT",name,json.dumps(out["projects"][name],sort_keys=True),flush=True)
print("ACCOUNT",json.dumps(out["account"],sort_keys=True))
Path("participation-audit.json").write_text(json.dumps(out,indent=2)+"\n")
if any("error" in p or p.get("mismatching_post_ids") or p.get("own_history_without_post") for p in out["projects"].values()): raise RuntimeError("One or more project audits incomplete")
