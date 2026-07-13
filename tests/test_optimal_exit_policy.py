from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from mlb_kalshi.optimal_exit import (
    EXIT_FEATURES,
    ExitPolicyConfig,
    evaluate_exit_policy,
)


class ConstantModel:
    def __init__(self, value: float):
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value)


class OptimalExitPolicyTests(unittest.TestCase):
    def test_elapsed_time_alone_never_forces_an_exit(self):
        frame = self._trajectory(
            ["2026-07-01T12:00:00Z", "2026-07-01T15:00:00Z"],
            exit_values=[0.20, 0.20],
            terminal_value=1.0,
        )
        result = evaluate_exit_policy(
            frame, ConstantModel(0.50), ExitPolicyConfig()
        )
        self.assertEqual(result.model_exits, 0)
        self.assertEqual(result.settlements, 1)

    def test_exit_requires_confirmation_and_a_later_snapshot(self):
        frame = self._trajectory(
            [
                "2026-07-01T12:00:00Z",
                "2026-07-01T12:00:03Z",
                "2026-07-01T12:00:04Z",
            ],
            exit_values=[0.80, 0.80, 0.80],
            terminal_value=0.0,
        )
        result = evaluate_exit_policy(
            frame,
            ConstantModel(0.50),
            ExitPolicyConfig(
                continuation_margin=0.02,
                confirmation_seconds=2.0,
            ),
        )
        self.assertEqual(result.model_exits, 1)
        self.assertEqual(
            result.records[0].exit_time,
            pd.Timestamp("2026-07-01T12:00:04Z"),
        )

    @staticmethod
    def _trajectory(times, exit_values, terminal_value):
        count = len(times)
        frame = pd.DataFrame({
            "trajectory_id": 1,
            "game_pk": 1,
            "game_date": pd.Timestamp("2026-07-01").date(),
            "side": "yes",
            "event_type": "single",
            "entry_time": pd.Timestamp("2026-07-01T11:59:00Z"),
            "snapshot_time": pd.to_datetime(times, utc=True),
            "entry_price": 0.40,
            "entry_fee": 0.10,
            "contracts": 25.0,
            "terminal_value": terminal_value,
            "exit_price": exit_values,
            "exit_fee": 0.10,
            "exit_value": exit_values,
        })
        for feature in EXIT_FEATURES:
            frame[feature] = 0.0
        return frame


if __name__ == "__main__":
    unittest.main()
