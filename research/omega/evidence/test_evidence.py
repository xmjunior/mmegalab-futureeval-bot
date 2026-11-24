import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import experiment
import prospective as future
from sources import links


class ArchivedEvidence(unittest.TestCase):
    def test_archived_reproduction_and_correlated_counts(self):
        data = json.loads((experiment.ROOT / "measurements.json").read_bytes())
        design = json.loads((experiment.ROOT / "design.json").read_bytes())
        expected = json.loads((experiment.ROOT / "result.json").read_bytes())
        summary = experiment.evaluate(data, design)["2025"]
        self.assertEqual(summary, expected["2025"])
        self.assertEqual((summary["editions"], summary["meetings"], summary["pairs"]), (24, 8, 276))
        self.assertEqual(summary["confusion"]["ncd_echo"]["fp"], 23)
        self.assertEqual(summary["distinct_different_rate_false_ncd_echoes"], 5)
        self.assertEqual(summary["effective_dated_correct"], 8)
        self.assertEqual(summary["announced_only_correct"], 4)

    def test_trusted_publisher_identity_preserves_conflicts(self):
        en, es = links()[:2]
        self.assertEqual(experiment.provenance(en, es, ["2.50"], ["2.50"]), "same_release")
        self.assertEqual(experiment.provenance(en, es, ["2.50"], ["2.25"]), "conflict")
        self.assertEqual(experiment.provenance(en, es, None, ["2.50"]), "unverified")
        spoof = en.replace("ecb.europa.eu", "ecb.europa.eu.invalid")
        self.assertEqual(experiment.provenance(spoof, es, ["2.50"], ["2.50"]), "unknown")


class Prospective(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "ecb.json"

    @staticmethod
    def fake_csv(start, end, rate="2.25", status="A"):
        return (f"KEY,UNIT,UNIT_MULT,OBS_STATUS,TIME_PERIOD,OBS_VALUE\n"
                f"{future.KEY},PCPA,0,A,{start},2.5\n"
                f"{future.KEY},PCPA,0,{status},{end},{rate}\n").encode()

    def test_certificate_reproduction_and_cutoff(self):
        expected = future.SEAL.read_bytes()
        self.assertEqual(future.encode(future.create_seal()), expected)
        cert = json.loads(expected)
        self.assertEqual(cert["estimates"]["counterevidence_dossier"]["p_cut"], .36)
        self.assertEqual(cert["last_training_observation"], "2026-11-05")
        self.assertEqual(cert["events"][0]["after"], "2026-12-24")
        self.assertEqual(len({event["meeting"] for event in cert["events"]}), 3)

    def test_pending_uses_no_network_and_is_idempotent(self):
        early = datetime(2026, 12, 25, 11, 59, tzinfo=timezone.utc)
        no_fetch = lambda *_: self.fail("premature or duplicate network request")
        self.assertEqual(future.settle(self.path, early, no_fetch)["pending"], 3)
        first = self.path.read_bytes()
        self.assertEqual(json.loads(first)["status"], "waiting")
        future.settle(self.path, early, no_fetch)
        self.assertEqual(self.path.read_bytes(), first)

    def test_missing_or_provisional_observations_remain_pending(self):
        now = datetime(2026, 12, 26, tzinfo=timezone.utc)
        missing = lambda start, end: self.fake_csv(start, "2026-12-23")
        preliminary = lambda start, end: self.fake_csv(start, end, status="E")
        for fetch in (missing, preliminary):
            summary = future.settle(self.path, now, fetch)
            self.assertEqual(summary["pending"], 3)

    def test_freeze_score_once_and_refuse_tampered_state(self):
        now = datetime(2026, 12, 26, tzinfo=timezone.utc)
        self.assertEqual(future.settle(self.path, now, self.fake_csv)["settled"], 1)
        frozen = self.path.read_bytes()
        receipt = json.loads(frozen)["settled"]["2026-12-17"]
        self.assertEqual(receipt["event"], 1)
        self.assertAlmostEqual(receipt["scores"]["counterevidence_dossier"]["brier"], .64**2)
        future.settle(self.path, now, lambda *_: self.fail("duplicate network read"))
        self.assertEqual(frozen, self.path.read_bytes())
        self.path.write_bytes(frozen.replace(b'"event": 1', b'"event": 0'))
        with self.assertRaises(ValueError):
            future.settle(self.path, now, self.fake_csv)

    def test_catch_up_scores_all_and_closes(self):
        now = datetime(2027, 3, 29, tzinfo=timezone.utc)
        summary = future.settle(self.path, now, lambda a,b: self.fake_csv(a,b,rate="2.50"))
        self.assertEqual(summary["pending"], 0)
        state = json.loads(self.path.read_bytes())
        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["seal_sha256"], hashlib.sha256(future.SEAL.read_bytes()).hexdigest())
        self.assertEqual({r["event"] for r in state["settled"].values()}, {0})

    def test_incorrect_units_are_rejected(self):
        raw = self.fake_csv("2026-12-16","2026-12-24").replace(b"PCPA", b"USD")
        with self.assertRaises(ValueError):
            future.observations(raw, {"2026-12-16", "2026-12-24"})


if __name__ == "__main__":
    unittest.main()
