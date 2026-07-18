import unittest

import numpy as np
import pandas as pd

from mlb_kalshi.empirical_reaction import (
    REACTION_FEATURES,
    EmpiricalStrategyConfig,
    build_reversion_candidates,
    evaluate_candidates,
    expected_home_probability,
    logit,
)


class ConstantReactionModel:
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, frame):
        return np.repeat(self.prediction, len(frame))


class EmpiricalReactionTests(unittest.TestCase):
    def test_expected_probability_is_oriented_back_to_home(self):
        move = float(logit(0.40) - logit(0.25))
        result = expected_home_probability(
            [0.25, 0.25], [move, move], [1, 0]
        )
        np.testing.assert_allclose(result, [0.40, 0.60])

    def test_candidate_requires_causal_fill_and_executable_reversion(self):
        event = {
            "event_id": 1,
            "game_pk": 10,
            "game_date": "2026-06-10",
            "home_win": 1,
            "batting_home": 1,
            "event_end_time": pd.Timestamp("2026-06-10 00:00:05Z"),
            "decision_time": pd.Timestamp("2026-06-10 00:00:10Z"),
            "valid_until": pd.Timestamp("2026-06-10 00:00:20Z"),
            "game_end_time": pd.Timestamp("2026-06-10 01:00:00Z"),
            "pre_batting_price": 0.50,
            "event_type": "single",
            "inning": 4,
            "batting_score_diff": -1,
            "outs_after": 1,
            "runner_on_first_after": 1,
            "runner_on_second_after": 0,
            "runner_on_third_after": 0,
            "pre_trade_count_5s": 3,
            "pre_volume_5s": 25.0,
            "pre_flow_imbalance_5s": 0.1,
            "pre_volatility_5s": 0.01,
            "pitch_duration_seconds": 2.0,
            "post_trade_count_5s": 4,
            "post_volume_5s": 30.0,
            "post_flow_imbalance_5s": 0.2,
            "post_volatility_5s": 0.02,
        }
        self.assertTrue(set(REACTION_FEATURES).issubset(event))
        trades = pd.DataFrame({
            "game_pk": [10, 10, 10],
            "trade_id": [1, 2, 3],
            "created_time": pd.to_datetime([
                "2026-06-10 00:00:10Z",
                "2026-06-10 00:00:11Z",
                "2026-06-10 00:00:12Z",
            ]),
            "yes_price_dollars": [0.70, 0.70, 0.59],
            "no_price_dollars": [0.30, 0.30, 0.41],
            "count_fp": [100.0, 100.0, 100.0],
            "taker_outcome_side": ["no", "no", "yes"],
        })
        model = ConstantReactionModel(float(logit(0.60) - logit(0.50)))
        result = build_reversion_candidates(
            pd.DataFrame([event]), trades, model
        )
        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertEqual(row["entry_time"], pd.Timestamp("2026-06-10 00:00:11Z"))
        self.assertEqual(row["entry_side"], "no")
        self.assertEqual(row["profitable_reversion"], 1)
        self.assertEqual(
            row["reversion_exit_time"], pd.Timestamp("2026-06-10 00:00:12Z")
        )

    def test_economic_break_even_gate_is_enforced(self):
        candidates = pd.DataFrame([
            {
                "candidate_id": 1,
                "game_pk": 1,
                "entry_time": pd.Timestamp("2026-07-01 00:00:00Z"),
                "entry_side": "yes",
                "entry_price": 0.50,
                "entry_fee": 0.20,
                "contracts": 20.0,
                "home_win": 1,
                "probability_residual": 0.10,
                "predicted_reversion_probability": 0.92,
                "breakeven_reversion_probability": 0.85,
                "profitable_reversion": 1,
                "reversion_exit_price": 0.60,
                "reversion_exit_fee": 0.20,
                "reversion_exit_time": pd.Timestamp("2026-07-01 00:01:00Z"),
            },
            {
                "candidate_id": 2,
                "game_pk": 2,
                "entry_time": pd.Timestamp("2026-07-01 00:00:00Z"),
                "entry_side": "yes",
                "entry_price": 0.50,
                "entry_fee": 0.20,
                "contracts": 20.0,
                "home_win": 0,
                "probability_residual": 0.10,
                "predicted_reversion_probability": 0.86,
                "breakeven_reversion_probability": 0.85,
                "profitable_reversion": 0,
                "reversion_exit_price": np.nan,
                "reversion_exit_fee": np.nan,
                "reversion_exit_time": pd.NaT,
            },
        ])
        result = evaluate_candidates(
            candidates,
            EmpiricalStrategyConfig(
                minimum_probability_residual=0.05,
                minimum_reversion_probability=0.50,
                minimum_reversion_probability_margin=0.02,
            ),
        )
        self.assertEqual(result.accepted_ids, [1])
        self.assertEqual(result.reversion_exits, 1)


if __name__ == "__main__":
    unittest.main()
