"""Read published leaderboards of completed seasonal competitions; no question endpoints."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from time import sleep
from urllib.request import Request, urlopen

PROJECTS={32916:"spring-aib-2026",33022:"summer-futureeval-2026",32813:"fall-aib-2025"}
out={}
for pid,slug in PROJECTS.items():
    sleep(4)
    url=f"https://www.metaculus.com/api/leaderboards/project/{pid}/?primary_only=true&with_entries=true"
    req=Request(url,method="GET",headers={"Authorization":"Token "+os.environ["METACULUS_TOKEN"],"User-Agent":"FutureEval-archive-audit"})
    with urlopen(req,timeout=35) as response:
        board,=json.load(response)
    if board.get("project_id")!=pid or board.get("is_primary_leaderboard") is not True:
        raise RuntimeError("Unexpected historical leaderboard")
    def summary(e):
        u=e.get("user") or {}
        return {"name":u.get("username") or e.get("aggregation_method"),"is_bot":u.get("is_bot"),
                "rank":e["rank"],"score":e["score"],"prize_usd":e.get("prize"),"exclusion_status":e.get("exclusion_status")}
    rows=board["entries"]
    def sequence(e): return -float(e["score"])
    eligible=sorted((x for x in rows if x["exclusion_status"]==0 and (x.get("user") or {}).get("is_bot") is True),key=sequence)
    paid=sorted((x for x in eligible if (x.get("prize") or 0)>0), key=sequence)
    refs=sorted((x for x in rows if (x.get("exclusion_status")!=0 or (x.get("user") or {}).get("is_bot") is not True)), key=sequence)
    out[str(pid)]={"url":f"https://www.metaculus.com/tournament/{slug}/","api":url,
            "finalized":board["finalized"],"finalize_time":board["finalize_time"],"end_time":board["end_time"],
            "prize_pool":board["prize_pool"],"score_type":board["score_type"],
            "eligible_top_three_by_score":[summary(x) for x in eligible[:3]],
            "eligible_top_three_with_published_prize":[summary(x) for x in paid[:3]],
            "highest_noncompeting_or_nonbot_references":[summary(x) for x in refs[:3]],"total_records":len(rows)}
    print("PODIUM",pid,json.dumps(out[str(pid)],sort_keys=True),flush=True)
Path("past-season-podiums.json").write_text(json.dumps({"observed_at":datetime.now(timezone.utc).isoformat(),"seasons":out},indent=2)+"\n")
