"""Exercise actual checkpoint writes and restoration in separate processes, without any credentials."""
import argparse
import asyncio
import copy
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from market_checkpoints import Checkpoints, ProviderCooldown, fingerprint, qualify_series

ROUTES = ["gemini/gemma-4-31b-it", "gemini/gemma-4-26b-a4b-it"]
REVISION = "a" * 40


def make_case(i):
    spot = datetime(2025, 1, 1 + i, tzinfo=timezone.utc)
    row = {"sample": i, "choices": [{"width": 1.25}]}
    return {"group": 30, "question": 50+i, "as_of": spot-timedelta(seconds=1), "spot": spot,
            "resolve": spot+timedelta(hours=2), "row": row,
            "prompt_hash": fingerprint(["synthetic", row]), "evidence_hash": fingerprint(["sealed", i])}


async def exercise(phase, path):
    cases = [make_case(i) for i in range(3)]
    store = Checkpoints(path, REVISION, 33131)
    seen = []
    async def reviewer(routes, rows):
        assert len(routes) == len(rows) == 1
        i = rows[0]["sample"]
        seen.append((routes[0], i))
        if phase == "interrupt" and i == 1:
            # Kill this Python process while waiting for decision 2. A separate process,
            # and in CI a separate hosted runner, must be able to finish the sequence.
            os._exit(86)
        if phase == "recover":
            assert i != 0, "Already validated work was called again"
        return routes[0], [1.25]

    if phase == "interrupt":
        await qualify_series(ROUTES, cases, store, reviewer)
        raise AssertionError("The deliberate interruption was missed")
    files = list(Path(path).glob("*.json"))
    assert len(files) == 1, "Expected exactly the decision saved before process termination"
    redacted = json.loads(files[0].read_text())
    assert set(redacted) == {"context","decision"}
    assert set(redacted["context"]) == {"format","project","revision","route","group","question","as_of",
                                       "spot","resolve","input_hash","prompt_hash","evidence_hash"}
    assert "1.25" not in files[0].read_text()
    result = await qualify_series(ROUTES, cases, store, reviewer)
    assert result == (ROUTES[0], [1.25] * 3)
    assert seen == [(ROUTES[0], 1), (ROUTES[0], 2)]
    assert all(store.read(ROUTES[0], c) == (True, 1.25) for c in cases)
    assert store.read(ROUTES[1], cases[0]) == (False,None)
    assert Checkpoints(path, "b"*40, 33131).read(ROUTES[0], cases[0]) == (False,None)
    assert Checkpoints(path, REVISION, 3).read(ROUTES[0], cases[0]) == (False,None)
    for field, value in (("question",99),("group",31),("as_of",cases[0]["as_of"]-timedelta(days=1)),
                         ("spot",cases[0]["spot"]+timedelta(minutes=1)),
                         ("resolve",cases[0]["resolve"]+timedelta(seconds=1)),("evidence_hash",fingerprint("new")),
                         ("prompt_hash",fingerprint("different")),("row",{"sample":12,"choices":[{"width":1.25}]})):
        changed = {**cases[0], field:value}
        assert store.read(ROUTES[0], changed) == (False,None), field
    future = {**cases[0], "resolve": datetime.now(timezone.utc) + timedelta(hours=1)}
    try:
        store.read(ROUTES[0], future)
    except ValueError:
        pass
    else:
        raise AssertionError("Prospective input entered historical cache")
    with tempfile.TemporaryDirectory() as other:
        backup = Checkpoints(other, REVISION, 33131)
        calls = []
        async def intermittent(routes, rows):
            route, i = routes[0], rows[0]["sample"]
            calls.append((route,i))
            if route == ROUTES[0] and i == 1: raise RuntimeError("offline")
            return route, [None if route == ROUTES[1] else 1.25]
        assert await qualify_series(ROUTES, cases, backup, intermittent) == (ROUTES[1], [None]*3)
        assert calls == [(ROUTES[0],0),(ROUTES[0],1),(ROUTES[1],0),(ROUTES[1],1),(ROUTES[1],2)]
        assert backup.read(ROUTES[1],cases[0]) == (True,None)
        assert backup.read(ROUTES[0],cases[1]) == (False,None)
        calls.clear()
        async def quota(routes, rows):
            calls.append(routes[0])
            raise ProviderCooldown("quota")
        try:
            await qualify_series(ROUTES, [make_case(5)], backup, quota)
        except ProviderCooldown:
            pass
        else:
            raise AssertionError("Provider quota did not abort its other routes")
        assert calls == [ROUTES[0]]
    corrupt = copy.deepcopy(redacted)
    corrupt["decision"] = "x"
    files[0].write_text(json.dumps(corrupt))
    try:
        store.read(ROUTES[0],cases[0])
    except ValueError:
        pass
    else:
        raise AssertionError("A corrupted decision was accepted")
    print("Interrupted after decision 1. Restored on a new process, called only records 2 and 3. "
          "Verified identity, inputs, prompt, evidence, windows, redaction, abstentions, provider quota and corruption handling.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["interrupt","recover"])
    p.add_argument("path")
    args = p.parse_args()
    asyncio.run(exercise(args.phase, args.path))
