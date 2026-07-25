from __future__ import annotations

import os
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

    def test_logs_result_runners_and_at_bat_progression(self):
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
        with patch("builtins.print") as output:
            state._log_play_transitions(123, game, payload, "observed-1")
            payload["liveData"]["plays"]["allPlays"].append({
                "atBatIndex": 5, "about": {}, "result": {},
                "runners": [], "playEvents": [],
            })
            state._log_play_transitions(123, game, payload, "observed-2")
        messages = "\n".join(str(call.args[0]) for call in output.call_args_list)
        self.assertIn("event_type=single", messages)
        self.assertIn("runners_populated=True", messages)
        self.assertIn("MLB_ATBAT_PROGRESSION", messages)


if __name__ == "__main__":
    unittest.main()
