import copy
import unittest
from unittest.mock import Mock

import groq_cohort as groq


class FreeCohortTests(unittest.TestCase):
    def setUp(self):
        self.journal = Mock()
        self.journal.book = {"arms": {"openai": {"max_micro_usd": 19200, "status":"completed"}}}
        self.snapshots = []
        self.journal.commit.side_effect = lambda _: self.snapshots.append(copy.deepcopy(self.journal.book))
        self.complete = Mock(return_value=('{"p":0.4}', {"prompt_tokens":110,"completion_tokens":48}))
        self.pause = Mock()

    def calculate(self):
        return groq.run(self.journal, self.complete, self.pause)

    def test_identical_inputs_all_matched_and_no_cost_on_replay(self):
        old = copy.deepcopy(self.journal.book)
        report = self.calculate()
        self.assertEqual(report["scored"],12)
        self.assertEqual(report["baseline"],groq.study.dry()["baseline"])
        self.assertEqual(report["inference_cost_usd"],0)
        self.assertEqual(report["openai_paired_n"],0)
        self.assertEqual(self.journal.book["arms"],old["arms"])
        self.assertEqual(groq.study.prior_reservations(self.journal.book),19200)
        cases = [groq.study.case(groq.study.load_values(),i)[0] for i in groq.study.ORIGINS]
        self.assertEqual([c.args[0] for c in self.complete.call_args_list],cases)
        self.assertEqual(self.pause.call_count,11)
        self.assertEqual(self.calculate(),report)
        self.assertEqual(self.complete.call_count,12)

    def test_no_call_before_committed_origin_lease(self):
        self.journal.commit.side_effect = [None, ValueError("conflict")]
        with self.assertRaisesRegex(ValueError,"conflict"):
            self.calculate()
        self.complete.assert_not_called()

    def test_lost_receipt_never_replays_origin_or_fabricates_a_score(self):
        self.calculate()
        # Durable prefix immediately after the first pre-call lease.
        self.journal.book = copy.deepcopy(self.snapshots[1])
        self.complete.reset_mock()
        report = self.calculate()
        self.assertEqual((self.complete.call_count, report["scored"], report["missing"]), (11,11,1))
        self.assertNotEqual(self.complete.call_args_list[0].args[0],
                            groq.study.case(groq.study.load_values(),groq.study.ORIGINS[0])[0])

    def test_nonfree_account_or_quota_stops_new_inference(self):
        error = RuntimeError("sensitive provider message")
        error.status_code = 403
        self.complete.side_effect = error
        report = self.calculate()
        self.assertEqual((report["scored"], self.complete.call_count), (0,1))
        self.assertNotIn("sensitive provider message",str(self.journal.book))
        self.assertEqual(self.calculate(),report)
        self.assertEqual(self.complete.call_count,1)

    def test_invalid_probability_excluded_from_both_comparators(self):
        self.complete.side_effect = [('{"p":true}',{})] + [self.complete.return_value]*11
        report = self.calculate()
        self.assertEqual(report["baseline"]["n"],11)
        self.assertEqual(report["groq"]["n"],11)

    def test_openai_join_only_on_same_origin_data_and_prompt(self):
        self.calculate()
        vals = groq.study.load_values()
        origin = groq.study.ORIGINS[3]
        prompt, base = groq.study.case(vals, origin)
        receipt = dict(origin_index=origin, prompt_sha256=groq.r.sha(prompt.encode()),
                       baseline=base, outcome=int(vals[origin+5]>vals[origin]), probability=0.75)
        self.journal.book["experiments"] = {groq.study.STUDY: {"source_sha256":groq.study.DATA_SHA,
                                                              "receipts":[receipt]}}
        result = groq.summary(self.journal.book)
        self.assertEqual(result["openai_paired_n"],1)
        self.assertEqual(result["equal_model_pool_on_shared"]["mean_probability"],(0.4+0.75)/2)
        receipt["prompt_sha256"]="wrong"
        with self.assertRaisesRegex(ValueError,"mismatched_peer_origin"):
            groq.summary(self.journal.book)


if __name__ == "__main__":
    unittest.main()
