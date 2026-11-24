"""Account operations audit. GET only. Never persist question text, predictions, comments, or public aggregates."""
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
    for attempt in range(3):
        try:
            with urlopen(Request(url, method="GET", headers={
                "Authorization": "Token " + os.environ["METACULUS_TOKEN"],
                "Accept-Language": "en", "User-Agent": "FutureEval-participation-audit"
            }), timeout=35) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError) as exc:
            if attempt == 2 or isinstance(exc, HTTPError) and exc.code not in (408,429,500,502,503,504):
                raise
            time.sleep(2 ** attempt)


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
    own = [q for q in questions.values() if q["own_count"]]
    # Verify detail where filters confirm an own forecast even if the list hid history.
    for post_id in filtered_post_ids - {q["post"] for q in own}:
        for q in extract(read(f"/posts/{post_id}/",with_cp="false")):
            questions[q["id"]] = q
    own = [q for q in questions.values() if q["own_count"]]
    if filtered_post_ids != {q["post"] for q in own}:
        raise RuntimeError("Authenticated post filter and own-subquestion history do not agree")
    for q in questions.values():
        if q["own_count"] is None and q["post"] not in filtered_post_ids:
            q["own_count"] = 0  # Exhaustive, separately paginated own-forecaster filter.
    counts = Counter(q["state"] for q in questions.values())
    comments = []
    for post_id in filtered_post_ids:
        for c in pages("/comments/",post=post_id,author=userid):
            if c.get("author_id",userid) != userid:
                raise ValueError("Incorrect author in filtered results")
            comments.append({k:c.get(k) for k in ("id","created_at","is_private","included_forecast")}|{"post":post_id})
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
    return {"project":pid,"slug":slug,"dates":{k:project.get(k) for k in ("start_date","close_date","forecasting_end_date")},
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
for name,slug in (("FutureEval","fall-futureeval-2026"),("MiniBench","minibench"),("MarketPulse","market-pulse-26q4")):
    try:
        out["projects"][name]=audit(slug,name,me["id"],created or now,now)
    except (HTTPError,URLError,TimeoutError,ValueError,RuntimeError,KeyError,TypeError) as e:
        out["projects"][name]={"error":f"{type(e).__name__}: {str(e)[:280]}"}
    print("AUDIT",name,json.dumps(out["projects"][name],sort_keys=True),flush=True)
print("ACCOUNT",json.dumps(out["account"],sort_keys=True))
Path("participation-audit.json").write_text(json.dumps(out,indent=2)+"\n")
if any("error" in p for p in out["projects"].values()): raise RuntimeError("One or more project audits incomplete")
