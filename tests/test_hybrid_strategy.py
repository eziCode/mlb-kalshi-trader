from __future__ import annotations

import unittest

import pandas as pd

from data.processed.scripts.build_event_state_features import (
    add_completed_event_availability,
)
from mlb_kalshi.hybrid import (
    HybridConfig,
    add_event_targets,
    anchored_event_target,
    hybrid_signal,
    simulate_hybrid,
)


class HybridStrategyTests(unittest.TestCase):
    def test_completed_result_is_delayed_until_next_pitch(self):
        frame = pd.DataFrame({
            "game_pk": [1, 1, 1],
            "at_bat_number": [1, 1, 2],
            "pitch_number": [1, 2, 1],
            "events": [None, "single", None],
            "inning_topbot": ["Top", "Top", "Top"],
            "pitch_timestamp_utc": pd.to_datetime([
                "2026-07-01T12:00:00Z",
                "2026-07-01T12:00:20Z",
                "2026-07-01T12:01:00Z",
            ]),
        })
        result = add_completed_event_availability(frame)
        self.assertTrue(pd.isna(result.loc[1, "completed_event"]))
        self.assertEqual(result.loc[2, "completed_event"], "single")
        self.assertEqual(result.loc[2, "completed_event_sequence"], 1)

    def test_target_uses_relative_fair_move_not_absolute_fair_level(self):
        target = float(anchored_event_target(0.40, 0.60, 0.70))
        self.assertAlmostEqual(target, 0.5090909, places=6)
        self.assertEqual(hybrid_signal(target, 0.54, 0.55, 0.02)[0], "no")
        self.assertEqual(hybrid_signal(target, 0.45, 0.46, 0.02)[0], "yes")

    def test_only_one_isolated_hit_creates_an_event_target(self):
        frame = self._event_frame()
        prepared = add_event_targets(frame, HybridConfig())
        self.assertEqual(prepared["hybrid_event"].tolist(), [False, True])

    def test_pre_event_anchor_cannot_include_the_hit_candle(self):
        frame = self._event_frame()
        reacted = frame.iloc[0].copy()
        reacted["decision_time"] = pd.Timestamp("2026-07-01T12:00:40Z")
        reacted["kalshi_price"] = 0.54
        reacted["yes_bid_close"] = 0.53
        reacted["yes_ask_close"] = 0.55
        detection = frame.iloc[1].copy()
        detection["decision_time"] = pd.Timestamp("2026-07-01T12:01:00Z")
        detection["completed_event_time"] = pd.Timestamp(
            "2026-07-01T12:00:50Z"
        )
        prepared = add_event_targets(
            pd.DataFrame([frame.iloc[0], reacted, detection]), HybridConfig()
        )
        event = prepared[prepared["hybrid_event"]].iloc[0]
        self.assertAlmostEqual(event["pre_event_market"], 0.40)

    def test_signal_must_survive_the_next_quote(self):
        frame = self._event_frame()
        # Detection quote overreacts, but the next executable quote has already
        # reverted to target. No fill is allowed at the detection candle.
        third = frame.iloc[-1].copy()
        third["decision_time"] = pd.Timestamp("2026-07-01T12:02:00Z")
        third["yes_bid_close"] = 0.50
        third["yes_ask_close"] = 0.52
        third["kalshi_price"] = 0.51
        frame = pd.concat([frame, third.to_frame().T], ignore_index=True)
        frame["decision_time"] = pd.to_datetime(frame["decision_time"], utc=True)
        frame["completed_event_time"] = pd.to_datetime(
            frame["completed_event_time"], utc=True
        )
        self.assertEqual(
            simulate_hybrid(frame, HybridConfig(minimum_edge=0.02)).trades,
            0,
        )

    @staticmethod
    def _event_frame() -> pd.DataFrame:
        return pd.DataFrame({
            "game_pk": [1, 1],
            "decision_time": pd.to_datetime([
                "2026-07-01T12:00:00Z", "2026-07-01T12:01:00Z",
            ]),
            "completed_event": [None, "single"],
            "completed_event_sequence": [0, 1],
            "completed_event_time": pd.to_datetime([
                None, "2026-07-01T12:00:50Z",
            ], utc=True),
            "completed_event_pitch_start": pd.to_datetime([
                None, "2026-07-01T12:00:20Z",
            ], utc=True),
            "kalshi_price": [0.40, 0.55],
            "yes_bid_close": [0.39, 0.54],
            "yes_ask_close": [0.41, 0.56],
            "fair_prob": [0.60, 0.70],
            "home_win": [1, 1],
        })


if __name__ == "__main__":
    unittest.main()
