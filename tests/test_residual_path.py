from __future__ import annotations

import unittest

import pandas as pd

from mlb_kalshi.residual_path import (
    RESIDUAL_PATH_FEATURES, evaluate_path_policy, residual_path_feature_frame,
)


class ResidualPathTests(unittest.TestCase):
    def test_feature_contract_is_event_agnostic(self):
        self.assertFalse(any("event" in name for name in RESIDUAL_PATH_FEATURES))
        frame = pd.DataFrame({name: [0.0] for name in RESIDUAL_PATH_FEATURES})
        frame["event_type"] = "home_run"
        result = residual_path_feature_frame(frame)
        self.assertNotIn("event_type", result)
        self.assertEqual(tuple(result.columns), RESIDUAL_PATH_FEATURES)

    def test_policy_counts_only_observed_maker_fills(self):
        frame = pd.DataFrame({
            "game_pk": [1, 1, 2],
            "signal_time": pd.to_datetime([
                "2026-06-01T12:00:00Z", "2026-06-01T12:00:05Z",
                "2026-06-01T12:00:00Z",
            ]),
            "maker_filled": [1, 1, 0],
            "net_pnl_10s": [1.0, -1.0, 5.0],
        })
        result = evaluate_path_policy(
            frame, [0.8, 0.8, 0.8], [1.0, 1.0, 1.0],
            horizon_seconds=10, minimum_fill_probability=0.5,
            minimum_expected_pnl=0.0,
        )
        # The second same-game signal overlaps the first position; the third
        # order is attempted but never receives a historical maker fill.
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["trades"], 1)
        self.assertEqual(result["pnl"], 1.0)


if __name__ == "__main__":
    unittest.main()
