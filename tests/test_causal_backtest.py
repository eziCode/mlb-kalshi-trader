from __future__ import annotations

import unittest

import pandas as pd

from backtesting.evaluate_strategy import simulate
from data.processed.scripts.apply_feature_preprocessing import preprocess


class CausalBacktestTests(unittest.TestCase):
    def test_signal_cannot_fill_at_same_observation(self):
        frame = pd.DataFrame({
            "game_pk": [1, 1],
            "decision_time": pd.to_datetime([
                "2026-07-01T12:00:00Z", "2026-07-01T12:01:00Z",
            ]),
            "yes_bid_close": [0.19, 0.49],
            "yes_ask_close": [0.20, 0.50],
            "final_prob": [0.60, 0.50],
            "home_win": [1, 1],
        })
        # The first quote has a 40% YES edge, but the next ask no longer does.
        self.assertEqual(simulate(frame).trades, 0)

    def test_preprocessing_is_idempotent_and_keeps_raw_units(self):
        raw = pd.DataFrame({
            "inning_topbot": ["Top", "Bot"],
            "runner_state": ["100", "011"],
            "yes_bid_close": [0.40, 0.55],
            "yes_ask_close": [0.42, 0.57],
            "volume": [125.0, 300.0],
        })
        once = preprocess(raw)
        twice = preprocess(once)
        pd.testing.assert_frame_equal(once, twice)
        self.assertEqual(once["volume"].tolist(), [125.0, 300.0])


if __name__ == "__main__":
    unittest.main()
