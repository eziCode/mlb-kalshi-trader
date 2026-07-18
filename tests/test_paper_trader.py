from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path
import tempfile

import pandas as pd

from live_trading_engine.paper_trader import (
    is_next_completed_event,
    latest_completed_pitch_token,
    match_games_to_home_markets,
    pitch_token_time,
    Position,
    SharedPaperPortfolio,
    should_surface_worker_line,
)


class PaperTraderTests(unittest.TestCase):
    def test_main_log_surfaces_trade_actions_but_not_every_hold_tick(self):
        self.assertTrue(should_surface_worker_line(
            "18:52:29 0.80/0.81 WATCH_YES_DOUBLE portfolio=$1000.00"
        ))
        self.assertTrue(should_surface_worker_line(
            "18:52:30 0.80/0.81 OPEN_YES_DOUBLE portfolio=$999.72"
        ))
        self.assertFalse(should_surface_worker_line(
            "18:52:31 0.80/0.81 HOLD portfolio=$999.60"
        ))

    def test_workers_share_one_atomic_cash_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "portfolio.sqlite3"
            first = SharedPaperPortfolio(path, starting_cash=1000.0)
            second = SharedPaperPortfolio(path, starting_cash=1000.0)
            position = Position(
                side="yes", contracts=20.0, entry_price=0.50,
                entry_fee=0.35, entry_time=datetime.now(timezone.utc),
                anchor_target=0.55, anchor_fair=0.50, event_id=7,
            )
            self.assertTrue(first.open_position(1, "GAME-ONE", position))
            self.assertTrue(second.open_position(2, "GAME-TWO", position))
            metrics = first.metrics()
            self.assertAlmostEqual(metrics.cash, 979.30)
            self.assertEqual(metrics.open_positions, 2)

            second.update_mark(1, 0.60)
            marked = first.metrics()
            self.assertGreater(marked.equity, metrics.equity)

            self.assertTrue(first.close_position(1, 11.65))
            closed = second.metrics()
            self.assertAlmostEqual(closed.cash, 990.95)
            self.assertEqual(closed.open_positions, 1)

    def test_event_sequence_supports_pregame_and_midgame_startup(self):
        # A pregame baseline has no completed event, so at-bat zero is new.
        self.assertTrue(is_next_completed_event(None, 0))
        # A midgame baseline consumes the currently visible event; only the
        # following event is eligible after startup.
        self.assertTrue(is_next_completed_event(12, 13))
        self.assertFalse(is_next_completed_event(12, 12))
        self.assertFalse(is_next_completed_event(12, 14))

    def test_daily_matcher_selects_home_team_market(self):
        games = [{
            "gamePk": 123,
            "gameDate": "2026-07-18T19:10:00Z",
            "teams": {
                "away": {"team": {"id": 112}},  # CHC
                "home": {"team": {"id": 145}},  # CHW
            },
        }]
        events = [{
            "event_ticker": "KXMLBGAME-26JUL181410CHCCWS",
            "markets": [
                {"ticker": "KXMLBGAME-26JUL181410CHCCWS-CHC"},
                {"ticker": "KXMLBGAME-26JUL181410CHCCWS-CWS"},
            ],
        }]
        matched, warnings = match_games_to_home_markets(games, events)
        self.assertEqual(warnings, [])
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0].game_pk, 123)
        self.assertEqual(matched[0].home_code, "CHW")
        self.assertTrue(matched[0].market_ticker.endswith("-CWS"))

    def test_daily_matcher_skips_ambiguous_doubleheader(self):
        games = [
            {
                "gamePk": game_pk,
                "gameDate": scheduled,
                "teams": {
                    "away": {"team": {"id": 134}},  # PIT
                    "home": {"team": {"id": 114}},  # CLE
                },
            }
            for game_pk, scheduled in [
                (1, "2026-07-18T17:10:00Z"),
                (2, "2026-07-18T23:10:00Z"),
            ]
        ]
        events = [{
            "event_ticker": "KXMLBGAME-26JUL181610PITCLE",
            "markets": [
                {"ticker": "KXMLBGAME-26JUL181610PITCLE-PIT"},
                {"ticker": "KXMLBGAME-26JUL181610PITCLE-CLE"},
            ],
        }]
        matched, warnings = match_games_to_home_markets(games, events)
        self.assertEqual(matched, [])
        self.assertIn("skipping ambiguous matchup", warnings[0])

    def test_latest_completed_pitch_token_ignores_unfinished_pitch(self):
        payload = {
            "liveData": {
                "plays": {
                    "allPlays": [
                        {
                            "atBatIndex": 4,
                            "playEvents": [
                                {
                                    "isPitch": True,
                                    "pitchNumber": 1,
                                    "endTime": "2026-07-18T19:00:01Z",
                                },
                                {
                                    "isPitch": True,
                                    "pitchNumber": 2,
                                    "startTime": "2026-07-18T19:00:20Z",
                                },
                            ],
                        }
                    ]
                }
            }
        }
        token = latest_completed_pitch_token(payload)
        self.assertEqual(token[:2], (4, 1))
        self.assertEqual(
            pitch_token_time(token),
            pd.Timestamp("2026-07-18T19:00:01Z").to_pydatetime(),
        )


if __name__ == "__main__":
    unittest.main()
