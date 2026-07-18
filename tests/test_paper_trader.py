from __future__ import annotations

import unittest

import pandas as pd

from live_trading_engine.paper_trader import (
    latest_completed_pitch_token,
    match_games_to_home_markets,
    pitch_token_time,
)


class PaperTraderTests(unittest.TestCase):
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
