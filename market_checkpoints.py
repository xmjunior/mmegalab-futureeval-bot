"""Historical, closed-case model decisions only. No network access or forecasts."""
import hashlib
import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

LOG = logging.getLogger("market_pulse.checkpoints")


class ProviderCooldown(RuntimeError):
    """Stop trying other names on the same provider while its free quota is exhausted."""


def fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


class Checkpoints:
    def __init__(self, directory, revision, project):
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Historical validation must be bound to a full git SHA")
        self.directory = Path(directory)
        self.revision = revision
        self.project = project

    def context(self, *, route, case):
        # Dates, identifiers and hashes are the whole on-disk allowlist. Do not
        # store prompts, input values, outcome labels or raw provider responses.
        if not case["as_of"] < case["spot"] <= case["resolve"] <= datetime.now(timezone.utc):
            raise ValueError("Only a resolved historical window may be checkpointed")
        return {"format": 1, "project": self.project, "revision": self.revision,
                "route": route, "group": case["group"], "question": case["question"],
                "as_of": case["as_of"].isoformat(), "spot": case["spot"].isoformat(),
                "resolve": case["resolve"].isoformat(), "input_hash": fingerprint(case["row"]),
                "prompt_hash": case["prompt_hash"], "evidence_hash": case["evidence_hash"]}

    def choices(self, case):
        return [None] + [o["width"] for o in case["row"]["choices"]]

    def read(self, route, case):
        context = self.context(route=route, case=case)
        key = fingerprint(context)
        path = self.directory / (key + ".json")
        if not path.is_file():
            return False, None
        entry = json.loads(path.read_text())
        if not isinstance(entry, dict) or set(entry) != {"context", "decision"} or entry["context"] != context:
            raise ValueError("Historical checkpoint context mismatch")
        matches = [c for c in self.choices(case) if fingerprint([key, c]) == entry["decision"]]
        if len(matches) != 1:
            raise ValueError("Historical checkpoint decision mismatch")
        LOG.info("Restored historical decision model=%s question=%s key=%s", route, case["question"], key)
        return True, matches[0]

    def write(self, route, case, choice):
        context = self.context(route=route, case=case)
        key = fingerprint(context)
        choices = self.choices(case)
        if choice is not None and (type(choice) not in (int, float) or choice not in choices):
            raise ValueError("Invalid historical selection")
        # Canonicalize numeric type, so 1 and 1.0 cannot have different fingerprints.
        canonical = choices[choices.index(choice)]
        entry = {"context": context, "decision": fingerprint([key, canonical])}
        self.directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=self.directory, mode="w", delete=False) as f:
            tmp = Path(f.name)
            try:
                json.dump(entry, f, sort_keys=True, allow_nan=False)
                f.flush()
                os.fsync(f.fileno())
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
        try:
            os.replace(tmp, self.directory / (key + ".json"))
        finally:
            tmp.unlink(missing_ok=True)
        LOG.info("Saved historical decision model=%s question=%s key=%s", route, case["question"], key)


async def qualify_series(routes, cases, store, review):
    """Require one named route for a complete blind series; recover only that route's own decisions."""
    for route in routes:
        decisions = []
        for case in cases:
            found, choice = store.read(route, case)
            if not found:
                try:
                    name, [choice] = await review([route], [case["row"]])
                except ProviderCooldown:
                    raise
                except RuntimeError:
                    # Model unavailable. Its earlier decisions survive; a second fixed
                    # identity, if configured, must establish its own entire series.
                    break
                if name != route:
                    raise ValueError("Historical reviewer identity mismatch")
                store.write(route, case, choice)
            decisions.append(choice)
        else:
            return route, decisions
    raise RuntimeError("No named free reviewer completed all historical dates")
