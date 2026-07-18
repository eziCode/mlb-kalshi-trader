from __future__ import annotations

import unittest

import pandas as pd

from backtesting.evaluate_strategy import simulate
from data.processed.scripts.apply_feature_preprocessing import preprocess


def add_constant_state(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(
        inning=1,
        inning_topbot=0,
        outs_when_up=0,
        score_diff=0,
        balls=0,
        strikes=0,
        runner_on_first=0,
        runner_on_second=0,
        runner_on_third=0,
    )


class CausalBacktestTests(unittest.TestCase):
    def test_signal_cannot_fill_at_same_observation(self):
        frame = add_constant_state(pd.DataFrame({
            "game_pk": [1, 1],
            "decision_time": pd.to_datetime([
                "2026-07-01T12:00:00Z", "2026-07-01T12:01:00Z",
            ]),
            "yes_bid_close": [0.19, 0.49],
            "yes_ask_close": [0.20, 0.50],
            "final_prob": [0.60, 0.50],
            "home_win": [1, 1],
        }))
        # The first quote has a 40% YES edge, but the next ask no longer does.
        self.assertEqual(simulate(frame).trades, 0)

    def test_exit_requires_meaningfully_negative_edge(self):
        frame = add_constant_state(pd.DataFrame({
            "game_pk": [1, 1, 1, 1],
            "decision_time": pd.to_datetime([
                "2026-07-01T12:00:00Z", "2026-07-01T12:01:00Z",
                "2026-07-01T12:02:00Z", "2026-07-01T12:03:00Z",
            ]),
            "yes_bid_close": [0.39, 0.39, 0.39, 0.39],
            "yes_ask_close": [0.40, 0.40, 0.40, 0.40],
            "final_prob": [0.70, 0.70, 0.38, 0.38],
            "home_win": [1, 1, 1, 1],
        }))
        # A 1% negative executable edge exits with no hysteresis, but not with
        # the configured 4% threshold.
        self.assertEqual(simulate(frame, exit_hysteresis=0).early_exits, 1)
        self.assertEqual(simulate(frame, exit_hysteresis=0.04).early_exits, 0)

    def test_state_change_cancels_pending_entry(self):
        frame = add_constant_state(pd.DataFrame({
            "game_pk": [1, 1],
            "decision_time": pd.to_datetime([
                "2026-07-01T12:00:00Z", "2026-07-01T12:01:00Z",
            ]),
            "yes_bid_close": [0.19, 0.19],
            "yes_ask_close": [0.20, 0.20],
            "final_prob": [0.60, 0.60],
            "home_win": [1, 1],
        }))
        frame.loc[1, "outs_when_up"] = 1

        # The second quote still offers the apparent edge, but it belongs to
        # a new state and may only create a new candidate, not fill the old one.
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
