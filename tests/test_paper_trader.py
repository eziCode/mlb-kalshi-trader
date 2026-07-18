from __future__ import annotations

import unittest

import pandas as pd

from live_trading_engine.paper_trader import (
    latest_completed_pitch_token,
    pitch_token_time,
)


class PaperTraderTests(unittest.TestCase):
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
