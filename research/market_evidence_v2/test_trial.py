from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from evidence import prior_coverage
from trial import load_inputs, parse, run_trial, schedule, score, sign_p


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = load_inputs()

    def test_original_diagnostics_and_balanced_independent_pairs(self):
        seq = [(p["key"], a) for p,a in schedule(self.inputs)]
        self.assertEqual(len(seq),15)
        self.assertEqual(len(set(seq)),15)
        self.assertEqual([a for _,a in seq[:3]],["candidate"]*3)
        self.assertEqual(sum(a=="control" for _,a in seq[3::2]),3)
        periods=[p for p in self.inputs if p["group"]=="reserve"]
        self.assertTrue(all(p["target"]<s["cutoff"] for p,s in zip(periods,periods[1:])))
        for p in self.inputs:
            self.assertEqual(p["control"]["choices"], p["candidate"]["choices"])
            self.assertEqual(len(p["candidate"]["evidence"]["prior_interval_checks"]),1)

    def test_disjoint_reference_counts_do_not_treat_overlaps_as_independent(self):
        values=np.sin(np.arange(505)/25)+3
        proof=prior_coverage(values,10,1.25)
        self.assertEqual(proof["prior_disjoint_targets"],34)
        self.assertLessEqual(proof["inside_interval"],proof["prior_disjoint_targets"])

    def test_absolute_null_and_ineligible_width(self):
        self.assertIsNone(parse("[null]"))
        for value in ("[false]","[1]","[1.5]","[1.25,null]"):
            with self.assertRaises(ValueError):
                parse(value)

    def test_transport_failure_preserves_receipt_without_retry(self):
        calls=[]
        def respond(prompt,max_tokens):
            calls.append(1)
            if len(calls)==2:
                raise TimeoutError
            return "[null]",{}
        with TemporaryDirectory() as out:
            r=run_trial(self.inputs,respond,Path(out)/"receipts.json")
            self.assertEqual((r["status"],r["attempted"]),("incomplete",2))
            self.assertIsNone(r["rows"][0]["width"])
            self.assertFalse(score(self.inputs,r,{})["advance"])

    def test_abstention_cannot_pass_and_common_policy_is_matched(self):
        with TemporaryDirectory() as out:
            r=run_trial(self.inputs,lambda *a,**kw:("[null]",{}),Path(out)/"r.json")
            labels={p["key"]:{"value":3.4} for p in self.inputs if p["group"]=="reserve"}
            s=score(self.inputs,r,labels)
            self.assertFalse(s["advance"])
            self.assertFalse(s["q4_candidate"]["passes_historical_gate"])
            self.assertEqual(s["reserve"]["paired_mean_loss_reduction"],0)
            self.assertIsNone(s["reserve"]["jointly_covered_mean_reduction"])

    def test_forward_and_score_support(self):
        self.assertEqual(sign_p([.2]*6),1/64)
        self.assertEqual(sign_p([.2]*4+[0,0]),1/16)
        self.assertEqual(sign_p([-.2]*4),1)
