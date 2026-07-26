from __future__ import annotations

import os
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests

import shared_mlb_feed as feed


class SharedMlbFeedTests(unittest.TestCase):
    def test_poll_intervals_are_adaptive(self):
        with patch.dict(os.environ, {
            "MLB_LIVE_POLL_SECONDS": "1.25",
            "MLB_PREGAME_POLL_SECONDS": "30",
            "MLB_FINAL_POLL_SECONDS": "300",
        }):
            self.assertEqual(feed.FeedState._interval("Live"), 1.25)
            self.assertEqual(feed.FeedState._interval("Preview"), 30)
            self.assertEqual(feed.FeedState._interval("Final"), 300)

    def test_response_preserves_cached_payload_and_error_metadata(self):
        state = feed.FeedState()
        game = feed.GameFeed(
            payload={"gamePk": 123}, received_at="2026-07-21T12:00:00+00:00",
            status="Live", failures=2, last_error="timeout",
        )
        state.games[123] = game
        with patch.object(state, "request", return_value=game):
            response = state.response(123)
        self.assertEqual(response["payload"]["gamePk"], 123)
        self.assertEqual(response["failures"], 2)
        self.assertEqual(response["last_error"], "timeout")
        self.assertIsNone(response["last_error_kind"])
        self.assertFalse(response["websocket_connected"])

    def test_websocket_message_immediately_wakes_poll_worker(self):
        state = feed.FeedState()
        state.games[123] = feed.GameFeed()
        wakeup = Mock()

        class OneMessageSocket:
            def __aiter__(self):
                self.sent = False
                return self

            async def __anext__(self):
                if self.sent:
                    raise StopAsyncIteration
                self.sent = True
                return "changed"

        class Connection:
            async def __aenter__(self):
                return OneMessageSocket()

            async def __aexit__(self, *args):
                return None

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"PAPER_LOG_DIR": directory}
        ), patch.object(feed.websockets, "connect", return_value=Connection()):
            import asyncio
            asyncio.run(state._websocket_game_loop(123, wakeup))
            path = Path(directory) / "mlb_feed_transport_2026-07-25.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertGreaterEqual(wakeup.set.call_count, 2)
        self.assertIsNotNone(state.games[123].websocket_last_message_at)
        self.assertEqual(records[0]["kind"], "mlb_websocket_connected")
        self.assertEqual(records[1]["kind"], "mlb_websocket_message")
        self.assertEqual(
            state.games[123].last_websocket_notification["kind"],
            "websocket_message",
        )

    def test_live_failure_retries_are_capped_at_five_seconds(self):
        self.assertEqual(feed.FeedState._failure_interval("Live", 1), .5)
        self.assertEqual(feed.FeedState._failure_interval("Live", 4), 4.0)
        self.assertEqual(feed.FeedState._failure_interval("Live", 20), 5.0)
        self.assertEqual(feed.FeedState._failure_interval("Preview", 20), 60.0)

    def test_failures_are_classified_for_specific_retry_policies(self):
        timeout = requests.Timeout("slow")
        self.assertEqual(
            feed.FeedState._classify_failure(timeout)[0], "timeout"
        )
        dns = requests.ConnectionError("NameResolutionError: failed to resolve")
        self.assertEqual(feed.FeedState._classify_failure(dns)[0], "dns")
        response = Mock(status_code=429, headers={"Retry-After": "17"})
        limited = requests.HTTPError(response=response)
        self.assertEqual(
            feed.FeedState._classify_failure(limited),
            ("rate_limit", 429, 17.0),
        )
        response = Mock(status_code=503, headers={})
        unavailable = requests.HTTPError(response=response)
        self.assertEqual(
            feed.FeedState._classify_failure(unavailable)[0], "server_error"
        )

    def test_rate_limit_honors_retry_after(self):
        self.assertEqual(
            feed.FeedState._failure_interval(
                "Live", 1, "rate_limit", retry_after=17
            ),
            17,
        )

    def test_persistent_session_has_https_pool(self):
        session = feed.FeedState._new_session()
        try:
            adapter = session.get_adapter("https://statsapi.mlb.com")
            self.assertEqual(adapter._pool_connections, 1)
            self.assertEqual(adapter._pool_maxsize, 2)
        finally:
            session.close()

    def test_stores_result_runners_and_at_bat_progression_without_printing(self):
        state = feed.FeedState()
        game = feed.GameFeed()
        play = {
            "atBatIndex": 4,
            "about": {"isComplete": False},
            "result": {"eventType": "single"},
            "runners": [{"movement": {"end": "1B", "isOut": False}}],
            "playEvents": [{"isPitch": True, "endTime": "2026-07-25T01:00:00Z"}],
        }
        payload = {"liveData": {"plays": {"allPlays": [play]}}}
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"PAPER_LOG_DIR": directory}
        ), patch("builtins.print") as output:
            state._log_play_transitions(
                123, game, payload, "2026-07-25T01:00:01+00:00"
            )
            payload["liveData"]["plays"]["allPlays"].append({
                "atBatIndex": 5, "about": {}, "result": {},
                "runners": [], "playEvents": [],
            })
            state._log_play_transitions(
                123, game, payload, "2026-07-25T01:00:02+00:00"
            )
            path = Path(directory) / "mlb_feed_transitions_2026-07-25.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
        output.assert_not_called()
        self.assertEqual(records[0]["stage"]["event_type"], "single")
        self.assertTrue(records[0]["stage"]["runners_populated"])
        self.assertTrue(records[0]["stage"]["atomic_play_eligible"])
        self.assertTrue(records[1]["at_bat_progressed"])
        self.assertEqual(records[1]["previous_at_bat"], 4)

    def test_persists_raw_transition_evidence_and_upstream_timing(self):
        state = feed.FeedState()
        game = feed.GameFeed()
        play = {
            "atBatIndex": 4,
            "about": {"isComplete": False},
            "result": {"eventType": "single"},
            "runners": [{"movement": {"end": "1B", "isOut": False}}],
            "playEvents": [{"isPitch": True, "endTime": "pitch-end"}],
        }
        payload = {
            "gameData": {"status": {"abstractGameState": "Live"}},
            "liveData": {
                "plays": {"allPlays": [play]},
                "linescore": {"currentInning": 3},
            },
        }
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"PAPER_LOG_DIR": directory}
        ), patch("builtins.print"):
            state._log_play_transitions(
                123, game, payload, "2026-07-25T01:00:01+00:00",
                {"request_elapsed_seconds": 0.125, "age": "0"},
            )
            path = Path(directory) / "mlb_feed_transitions_2026-07-25.jsonl"
            record = json.loads(path.read_text().strip())
        self.assertEqual(record["play"]["result"]["eventType"], "single")
        self.assertEqual(record["linescore"]["currentInning"], 3)
        self.assertEqual(record["upstream"]["request_elapsed_seconds"], .125)
        self.assertEqual(record["upstream"]["age"], "0")

    def test_transition_records_atomic_ball_in_play_rejection_reason(self):
        state = feed.FeedState()
        game = feed.GameFeed()
        play = {
            "atBatIndex": 4,
            "about": {"isComplete": False},
            "result": {}, "runners": [],
            "playEvents": [{
                "isPitch": True, "pitchNumber": 2,
                "endTime": "2026-07-25T01:00:00Z",
                "details": {"isInPlay": True},
            }],
        }
        payload = {"liveData": {"plays": {"allPlays": [play]}}}
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"PAPER_LOG_DIR": directory}
        ):
            state._log_play_transitions(
                123, game, payload, "2026-07-25T01:00:01+00:00"
            )
            path = Path(directory) / "mlb_feed_transitions_2026-07-25.jsonl"
            record = json.loads(path.read_text().strip())
        self.assertTrue(record["stage"]["is_in_play"])
        self.assertFalse(record["stage"]["atomic_play_eligible"])
        self.assertEqual(
            record["stage"]["atomic_rejection_reason"],
            "play is not complete",
        )


if __name__ == "__main__":
    unittest.main()
