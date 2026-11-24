import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from review import cases, FIXTURE, parse_response, trial


class NumericReview(unittest.TestCase):
    def test_source_boundary(self):
        for entry in cases():
            row, = json.loads(entry["prompt"].rsplit("\n", 1)[1])
            self.assertEqual(set(row), {"observations", "days_until_measurement", "quote_age_days", "choices"})
            self.assertNotIn(str(entry["id"]),entry["prompt"])
            self.assertEqual(len(row["choices"][0]["quantiles"]), 5)
            self.assertLess(entry["source_date"], entry["cutoff"][:10])
            self.assertEqual([choice["width"] for choice in row["choices"]], [1.25])

    def test_detect_fixture_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "amended.json"
            path.write_bytes(FIXTURE.read_bytes() + b" ")
            with self.assertRaises(ValueError):
                cases(path)

    def test_reject_bad_responses(self):
        entry = cases()[0]
        for text in ("[true]", "[1.5]", "[]", "[1.25,null]", "{}", "[NaN]", "[1.25] extra"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_response(text, entry)
        self.assertEqual(parse_response("```json\n[1.25]\n```", entry), 1.25)
        self.assertIsNone(parse_response("[null]", entry))

    def test_three_distinct_trials_report_abstention(self):
        transport = Mock(side_effect=[("[1.25]", {}), ("[null]", {}), ("[1.25]", {})])
        result = trial(transport)
        self.assertEqual((result["numeric_decisions"], result["abstentions"], result["published"]), (2, 1, 0))
        self.assertEqual(len({call.args[0] for call in transport.call_args_list}), 3)

    def test_partial_failure_keeps_receipt_and_stops(self):
        transport = Mock(side_effect=[("[1.25]", {}), TimeoutError()])
        result = trial(transport)
        self.assertEqual((result["status"], result["hosted_requests_attempted"], len(result["cases"])),
                         ("incomplete", 2, 1))
        self.assertEqual(transport.call_count, 2)
