import unittest

import pandas as pd

from mlb_kalshi.runner import (
    RunnerPolicyConfig,
    build_runner_outcomes,
    recovery_contracts,
)
from mlb_kalshi.strategy import taker_fee


class RunnerStrategyTests(unittest.TestCase):
    def test_recovery_sale_covers_capital_and_leaves_a_runner(self):
        capital = 10.35
        contracts = recovery_contracts(capital, 0.60, 20.0)
        proceeds = contracts * 0.60 - taker_fee(contracts, 0.60)
        self.assertGreaterEqual(proceeds, capital - 1e-9)
        self.assertLess(contracts, 20.0)

    def test_trailing_trigger_fills_only_on_a_later_compatible_trade(self):
        candidates = pd.DataFrame([{
            "candidate_id": 1,
            "game_pk": 10,
            "game_date": "2026-07-01",
            "game_end_time": pd.Timestamp("2026-07-01 01:00:00Z"),
            "home_win": 1,
            "entry_time": pd.Timestamp("2026-07-01 00:00:01Z"),
            "entry_side": "yes",
            "entry_price": 0.50,
            "entry_fee": taker_fee(20.0, 0.50),
            "contracts": 20.0,
            "profitable_reversion": 1,
            "reversion_exit_time": pd.Timestamp("2026-07-01 00:00:10Z"),
            "reversion_exit_price": 0.60,
        }])
        trades = pd.DataFrame({
            "game_pk": [10, 10, 10, 10],
            "trade_id": [1, 2, 3, 4],
            "created_time": pd.to_datetime([
                "2026-07-01 00:00:10Z",
                "2026-07-01 00:00:11Z",
                "2026-07-01 00:00:12Z",
                "2026-07-01 00:00:13Z",
            ]),
            "yes_price_dollars": [0.60, 0.65, 0.62, 0.61],
            "no_price_dollars": [0.40, 0.35, 0.38, 0.39],
            "count_fp": [100.0, 100.0, 100.0, 100.0],
            "taker_outcome_side": ["no", "yes", "no", "no"],
        })
        updates = pd.DataFrame(columns=[
            "game_pk", "pitch_end_time", "fair_after"
        ])
        outcomes = build_runner_outcomes(
            candidates,
            trades,
            updates,
            RunnerPolicyConfig(
                trailing_giveback_fraction=0.50,
                trailing_activation_multiple=0.25,
                second_target_multiple=2.0,
            ),
        )
        row = outcomes.iloc[0]
        self.assertGreater(row["runner_contracts"], 0)
        self.assertEqual(row["exit_reason"], "runner_trailing")
        self.assertEqual(
            row["runner_exit_time"], pd.Timestamp("2026-07-01 00:00:13Z")
        )

    def test_elapsed_time_alone_does_not_exit_runner(self):
        candidates = pd.DataFrame([{
            "candidate_id": 2,
            "game_pk": 11,
            "game_date": "2026-07-01",
            "game_end_time": pd.Timestamp("2026-07-01 01:00:00Z"),
            "home_win": 0,
            "entry_time": pd.Timestamp("2026-07-01 00:00:01Z"),
            "entry_side": "yes",
            "entry_price": 0.50,
            "entry_fee": taker_fee(20.0, 0.50),
            "contracts": 20.0,
            "profitable_reversion": 1,
            "reversion_exit_time": pd.Timestamp("2026-07-01 00:00:10Z"),
            "reversion_exit_price": 0.60,
        }])
        trades = pd.DataFrame({
            "game_pk": [11, 11, 11],
            "trade_id": [1, 2, 3],
            "created_time": pd.to_datetime([
                "2026-07-01 00:00:10Z",
                "2026-07-01 00:10:00Z",
                "2026-07-01 00:50:00Z",
            ]),
            "yes_price_dollars": [0.60, 0.60, 0.60],
            "no_price_dollars": [0.40, 0.40, 0.40],
            "count_fp": [100.0, 100.0, 100.0],
            "taker_outcome_side": ["no", "no", "no"],
        })
        updates = pd.DataFrame(columns=[
            "game_pk", "pitch_end_time", "fair_after"
        ])
        outcomes = build_runner_outcomes(
            candidates, trades, updates, RunnerPolicyConfig()
        )
        self.assertEqual(outcomes.iloc[0]["exit_reason"], "runner_settlement")


if __name__ == "__main__":
    unittest.main()
