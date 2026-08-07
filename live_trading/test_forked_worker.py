from __future__ import annotations

import os
import unittest

from live_trading.forked_worker import (
    spawn_forked_worker, worker_lifecycle_line,
)


@unittest.skipUnless(hasattr(os, "fork"), "requires fork")
class ForkedWorkerTests(unittest.TestCase):
    def test_lifecycle_line_contains_complete_worker_identity(self):
        self.assertEqual(worker_lifecycle_line(
            "exit", strategy="settlement_value", pid=42,
            launcher="fork_shared_imports", status=0, game_pk=123,
            home_ticker="HOME", away_ticker="AWAY",
        ), (
            "WORKER EXIT strategy=settlement_value pid=42 "
            "launcher=fork_shared_imports status=0 game_pk=123 "
            "home_ticker=HOME away_ticker=AWAY"
        ))

    def test_child_receives_environment_and_output_is_captured(self):
        def child():
            print(f"worker={os.environ['WORKER_TEST_VALUE']}", flush=True)

        worker = spawn_forked_worker(child, {
            **os.environ, "WORKER_TEST_VALUE": "isolated",
        })
        self.assertEqual(worker.wait(timeout=5), 0)
        self.assertEqual(worker.stdout.read().strip(), "worker=isolated")
        worker.stdout.close()

    def test_child_failure_is_reported(self):
        def child():
            raise RuntimeError("boom")

        worker = spawn_forked_worker(child, dict(os.environ))
        self.assertEqual(worker.wait(timeout=5), 1)
        self.assertIn("boom", worker.stdout.read())
        worker.stdout.close()


if __name__ == "__main__":
    unittest.main()
