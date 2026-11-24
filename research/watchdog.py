"""Only dispatch the existing main workflows for missed polling windows. No forecast data is posted here."""
import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
import time
from urllib.request import HTTPRedirectHandler, Request, build_opener


class SameHost(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        from urllib.parse import urlsplit
        requested, redirected = urlsplit(request.full_url), urlsplit(newurl)
        if requested.scheme != redirected.scheme or requested.netloc != redirected.netloc:
            return None
        return super().redirect_request(request, fp, code, msg, headers, newurl)


OPENER = build_opener(SameHost)

TARGETS = {"33121": "run_bot_on_tournament.yaml", "minibench": "run_bot_on_minibench.yaml"}
GH = "https://api.github.com/repos/xmjunior/mmegalab-futureeval-bot"
MC = "https://www.metaculus.com/api"


def call(base, suffix, token, method="GET", payload=None, **params):
    prefix = "Bearer " if base == GH else "Token "
    url = base + suffix + ("?" + urlencode(params,doseq=True) if params else "")
    req = Request(url, method=method, data=json.dumps(payload).encode() if payload else None,
                  headers={"Authorization":prefix+token,"Content-Type":"application/json","User-Agent":"tournament-operations"})
    for attempt in range(3):
        if base == MC: time.sleep(3.7)
        try:
            with OPENER.open(req, timeout=35) as response:
                return json.load(response) if response.status!=204 else None
        except (HTTPError,URLError,TimeoutError) as err:
            if method != "GET" or attempt==2: raise
            if isinstance(err,HTTPError) and err.code not in (408,429,500,502,503,504): raise
            retry=err.headers.get("Retry-After","") if isinstance(err,HTTPError) else ""
            time.sleep(min(120, int(retry) if retry.isdigit() else 2**attempt))



def main():
    args = argparse.ArgumentParser()
    args.add_argument("--dispatch", action="store_true", help="Call only the whitelisted workflows on main")
    opts = args.parse_args()
    git_token = os.environ["GH_TOKEN"]
    mc_token = os.environ["METACULUS_TOKEN"]
    verified = False
    now = datetime.now(timezone.utc)
    for ident, file in TARGETS.items():
        route = "/actions/workflows/" + file
        runs = call(GH,route+"/runs",git_token,per_page=10)["workflow_runs"]
        relevant = [r for r in runs if r["head_branch"]=="main" or
                    (file=="run_bot_on_tournament.yaml" and r["head_branch"]=="release/futureeval")]
        active = any(r["status"]!="completed" for r in relevant)
        recent = any(now-datetime.fromisoformat(r["created_at"]) < timedelta(minutes=42) for r in relevant)
        native = next((r for r in relevant if r["event"]=="schedule"),None)
        if active or recent:
            print(ident, "original workflow active or recent; no dispatch", "last_native", (native or {}).get('id'))
            continue
        if not verified:
            me = call(MC, "/users/me/", mc_token)
            if (me.get("id"), me.get("username"), me.get("is_primary_bot")) != (313890, "megalab-mario-bot", True):
                raise ValueError("Wrong primary bot")
            verified = True
        project = call(MC, f"/projects/tournaments/{ident}/", mc_token)
        pid = project["id"]
        if type(pid) is not int or (ident.isdigit() and pid != int(ident)):
            raise ValueError("Wrong tournament")
        feed = call(MC,"/posts/",mc_token,tournaments=pid,statuses="open",with_cp="false",
                    include_descriptions="false",limit=100)
        if len(feed["results"]) >= 100: raise RuntimeError("Tournament batch too large for this guard")
        candidates = set()
        for post in feed["results"]:
            if post.get("group_of_questions"):
                post = call(MC, f"/posts/{post['id']}/", mc_token, with_cp="false")
            q = post.get("question")
            if q and q["status"]=="open": candidates.add(q["id"])
            for q in (post.get("group_of_questions") or {}).get("questions",[]):
                if q["status"]=="open": candidates.add(q["id"])
            for q in (post.get("conditional") or {}).values():
                if isinstance(q,dict) and q.get("status")=="open" and q.get("id"): candidates.add(q["id"])
        if not candidates:
            print(ident, "no open subquestions; no dispatch")
            continue
        history = call(MC,"/questions/bulk-forecast-read/",mc_token, question_ids=sorted(candidates))["results"]
        if any(item.get("author_id") != 313890 for row in history for item in row["forecasts"]):
            raise ValueError("Unverified forecast owner")
        own = {row["question_id"]: bool(row["forecasts"]) for row in history}
        if candidates-own.keys(): raise RuntimeError("Account history incomplete; refusing dispatch")
        missed = sorted(qid for qid in candidates if not own[qid])
        print(ident,"unanswered",missed,"dispatch",bool(opts.dispatch and missed))
        if missed and opts.dispatch:
            call(GH,route+"/dispatches",git_token,method="POST",payload={"ref":"main"})


if __name__=="__main__":
    main()
