import copy
import json
import os
import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, patch
from tempfile import TemporaryDirectory
from pathlib import Path
from urllib.error import HTTPError

import account_activity as billing
import closed_cohort as cohort

NOW = datetime(2026, 11, 24, 23, tzinfo=timezone.utc)


class ClosedCohort(unittest.TestCase):
    def setUp(self):
        empty = TemporaryDirectory()
        self.addCleanup(empty.cleanup)
        root = patch.object(cohort, "ROOT", Path(empty.name))
        root.start()
        self.addCleanup(root.stop)
        self.book = {"schema": cohort.r.SCHEMA, "panel_hash": cohort.r.PANEL_SHA,
                     "arms": {"openai": {"status": "completed", "max_micro_usd": 19200}}}
        self.scope = {
            "org": "org-test", "project": "proj_test", "covered_from": cohort.r.OWNER_CHECKED,
            "valid_through": cohort.EXPIRY.isoformat(), "verified_orgwide_including_pending": True,
            "external_upper_micro_usd": 0, "snapshot_sha256": cohort.r.OWNER_CAPTURE_SHA256,
        }
        self.memory = Mock()
        self.memory.book = self.book
        self.calls = []
        self.memory.commit.side_effect = lambda _: self.calls.append(copy.deepcopy(self.book))

    def test_future_does_not_enter_any_prompt(self):
        values = cohort.load_values()
        for origin in cohort.ORIGINS:
            distorted = values[:origin + 1] + [2400] * 10
            self.assertEqual(cohort.case(values, origin), cohort.case(distorted, origin))
        self.assertTrue(all(a + 5 < b for a, b in zip(cohort.ORIGINS, cohort.ORIGINS[1:])))
        self.assertEqual(cohort.dry()["baseline"]["n"], 12)

    def test_grant_evidence_missing_changed_or_expired(self):
        raw = json.dumps(self.scope)
        with patch.dict(os.environ, {"ARENA_EXPOSURE_RECORD": raw,
                                     "ARENA_EXPOSURE_SHA256": cohort.r.sha(raw.encode())}):
            self.assertEqual(cohort.reviewed_scope(NOW), self.scope)
            with self.assertRaises(ValueError):
                cohort.reviewed_scope(cohort.EXPIRY)
        for changes in ({"verified_orgwide_including_pending": False},
                        {"external_upper_micro_usd": -1}, {"external_upper_micro_usd": True},
                        {"covered_from": NOW.isoformat()}):
            raw = json.dumps(dict(self.scope, **changes))
            with patch.dict(os.environ, {"ARENA_EXPOSURE_RECORD": raw,
                                         "ARENA_EXPOSURE_SHA256": cohort.r.sha(raw.encode())}):
                with self.assertRaises(ValueError):
                    cohort.reviewed_scope(NOW)
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, "unbounded"):
            cohort.reviewed_scope(NOW)

    @patch.object(cohort.r, "try_openai")
    @patch.object(cohort.r, "prepare_openai")
    def test_first_reserves_all_twelve_and_replay_costs_nothing(self, prepare, invoke):
        prepare.side_effect = lambda text, _: ({"input": text}, 100, cohort.r.plan(100))
        invoke.return_value = ('{"p":0.51}', {"input_tokens": 100, "output_tokens": 12})
        self.assertEqual(cohort.run(self.memory, self.scope, lambda: NOW), "completed")
        first = self.calls[0]["experiments"][cohort.STUDY]
        self.assertEqual((first["status"], first["max_micro_usd"]), ("reserved", 12 * cohort.r.plan(100)))
        self.assertEqual(len(first["receipts"]), 0)
        self.assertEqual(self.book["experiments"][cohort.STUDY]["paired_model"]["n"], 12)
        self.assertEqual(invoke.call_count, 12)
        self.assertEqual(len({c.args[0]["input"] for c in invoke.call_args_list}), 12)
        with self.assertRaisesRegex(ValueError, "already_reserved"):
            cohort.run(self.memory, self.scope, lambda: NOW)
        self.assertEqual(invoke.call_count, 12)

    @patch.object(cohort.r, "try_openai")
    @patch.object(cohort.r, "prepare_openai")
    def test_external_and_journal_liabilities_pay_no_overage(self, prepare, invoke):
        prepare.return_value = ({}, 100, 1000)
        self.scope["external_upper_micro_usd"] = cohort.VERIFIED_GRANT_FLOOR - 31200 + 1
        with self.assertRaisesRegex(ValueError, "insufficient"):
            cohort.run(self.memory, self.scope, lambda: NOW)
        self.assertFalse(invoke.called)
        self.assertFalse(self.memory.commit.called)
        self.scope["external_upper_micro_usd"] -= 1
        invoke.side_effect = TimeoutError()
        self.assertEqual(cohort.run(self.memory, self.scope, lambda: NOW), "stopped_no_retry")
        self.assertEqual(self.book["experiments"][cohort.STUDY]["max_micro_usd"], 12000)
        self.assertEqual(invoke.call_count, 1)
        with self.assertRaisesRegex(ValueError, "already_reserved"):
            cohort.run(self.memory, self.scope, lambda: NOW)
        self.assertEqual(invoke.call_count, 1)

    def test_cas_failure_prevents_inference(self):
        self.memory.commit.side_effect = ValueError("conflict")
        with patch.object(cohort.r, "prepare_openai", return_value=({}, 100, 1000)), \
             patch.object(cohort.r, "try_openai") as invoke, self.assertRaisesRegex(ValueError, "conflict"):
            cohort.run(self.memory, self.scope, lambda: NOW)
        invoke.assert_not_called()

    def test_owner_window_binds_only_authorization_commit_and_user_time(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            git = lambda *args: cohort.r.git(root, *args)
            git("init", "--quiet")
            git("config", "user.email", "local@example.test")
            git("config", "user.name", "TEST")
            (root / "code").write_text("reviewed code")
            git("add", "code")
            git("commit", "-qm", "reviewed code")
            parent = git("rev-parse", "HEAD")
            record = {**self.scope, "study": cohort.STUDY, "prior_run_id": "45493080434",
                      "reviewed_code_sha": parent, "attested_at": NOW.isoformat()}
            del record["org"], record["project"]
            file = root / cohort.OWNER_WINDOW
            file.parent.mkdir(parents=True)
            file.write_text(json.dumps(record))
            git("add", cohort.OWNER_WINDOW.as_posix())
            git("commit", "-qm", "review owner time window")
            head = git("rev-parse", "HEAD")
            env = {"GITHUB_REPOSITORY": "xmjunior/mmegalab-futureeval-bot",
                   "GITHUB_REF": "refs/heads/verify/arena-closed-grant",
                   "GITHUB_EVENT_NAME": "push", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_SHA": head}
            with patch.dict(os.environ, env, clear=True):
                scope = cohort.owner_scope(NOW, root)
                self.assertEqual(scope["authorization_commit"], head)
                self.assertEqual(scope["credential_scope"], cohort.OWNER_SCOPE)
                with self.assertRaisesRegex(ValueError, "window_closed"):
                    cohort.owner_scope(cohort.EXPIRY, root)
                for bad in ({"GITHUB_SHA": parent}, {"GITHUB_REF": "refs/heads/main"},
                            {"GITHUB_RUN_ATTEMPT": "2"}, {"GITHUB_EVENT_NAME": "workflow_dispatch"}):
                    with patch.dict(os.environ, bad), self.assertRaises(ValueError):
                        cohort.owner_scope(NOW, root)
                (root / "code").write_text("changed code")
                with self.assertRaisesRegex(ValueError, "unreviewed"):
                    cohort.owner_scope(NOW, root)
                git("add", "code")
                git("commit", "--amend", "--no-edit", "--quiet")
                with patch.dict(os.environ, {"GITHUB_SHA": git("rev-parse", "HEAD")}), \
                     self.assertRaisesRegex(ValueError, "unreviewed"):
                    cohort.owner_scope(NOW, root)

    def test_same_project_key_exact_free_count_no_original_panel_shortcut(self):
        scope = {"credential_scope": cohort.OWNER_SCOPE}
        rate = b"| Input | $4 | 1M tokens |\n| Output | $20 | 1M tokens |"
        count = b'{"object":"response.input_tokens","input_tokens":120}'
        with patch.dict(os.environ, {"OPENAI_API_KEY": "example"}, clear=True), \
             patch.object(cohort.r.shadow, "request", side_effect=[rate, count]) as request:
            _, tokens, reserve = cohort.r.prepare_openai("NEW history-only input", scope)
            self.assertEqual((tokens, reserve), (120, cohort.r.plan(120)))
            self.assertEqual(request.call_args.args[0], cohort.r.OAI + "/input_tokens")
            self.assertEqual(request.call_args.args[2], {"Authorization": "Bearer example"})
        scope.update(self.scope, prior_run_id="45493080434", authorization_commit="a" * 40)
        with patch.object(cohort.r, "prepare_openai") as preflight, \
             self.assertRaisesRegex(ValueError, "prior_owner_receipt_missing"):
            cohort.run(self.memory, scope, lambda: NOW)
        preflight.assert_not_called()
        self.book["arms"]["openai"].update(github_run_id="45493080434",
                reviewed_capture_sha256=cohort.r.OWNER_CAPTURE_SHA256,
                request={"response_id_sha256": cohort.REFERENCE_RECEIPT})
        with patch.object(cohort.r, "prepare_openai", side_effect=lambda text, s:
                          ({"input": text}, 120, cohort.r.plan(120))), \
             patch.object(cohort.r, "try_openai", return_value=('{"p":0.55}', {})):
            self.assertEqual(cohort.run(self.memory, scope, lambda: NOW), "completed")
        self.assertEqual(self.book["experiments"][cohort.STUDY]["authorization_commit"], "a" * 40)

    def test_activity_denied_is_sanitized(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "example"}, clear=True), \
             patch.object(billing.runner.shadow, "request",
                          side_effect=HTTPError("url", 403, "private detail", None, None)):
            self.assertEqual(billing.inspect(NOW), {"organization_activity": "http_403"})

    def test_activity_pages_are_checked_and_costs_never_release_funds(self):
        def page(value, more):
            return json.dumps({"object": "page", "data": [{"object": "bucket", "results": [
                {"amount": {"currency": "USD", "value": value}}]}],
                "has_more": more, "next_page": "next" if more else None}).encode()
        with patch.dict(os.environ, {"OPENAI_ADMIN_KEY": "example"}, clear=True), \
             patch.object(billing.runner.shadow, "request", side_effect=[page(0.1, True), page(-1, False)]) as get:
            info = billing.inspect(NOW)
            self.assertEqual(info, {"organization_activity": "positive_costs_reported",
                                    "pending_external_usage_upper_bound": "unavailable_from_costs_endpoint"})
            self.assertEqual(get.call_count, 2)
            self.assertIn("page=next", get.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
