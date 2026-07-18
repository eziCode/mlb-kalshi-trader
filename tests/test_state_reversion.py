from __future__ import annotations

import unittest

import pandas as pd

from mlb_kalshi.state_overshoot import (
    OvershootConfig,
    build_state_overshoot_candidates,
    signed_logit_residual,
    simulate_state_reversion,
)
from mlb_kalshi.state_reversion import (
    StateReversionConfig,
    accepted_state_reversions,
    add_before_and_delta_state,
    transition_diagnostics,
)


class StateReversionTests(unittest.TestCase):
    @staticmethod
    def updates() -> pd.DataFrame:
        return pd.DataFrame({
            "game_pk": [1, 1, 1],
            "at_bat_number": [1, 1, 2],
            "pitch_number": [1, 2, 1],
            "pitch_end_time": pd.to_datetime([
                "2026-06-01T12:00:01Z", "2026-06-01T12:00:20Z",
                "2026-06-01T12:01:00Z",
            ]),
            "fair_before": [0.50, 0.51, 0.49],
            "fair_after": [0.51, 0.49, 0.55],
            "inning_after": [1, 1, 1],
            "inning_topbot_after": [1, 1, 1],
            "outs_when_up_after": [0, 1, 1],
            "score_diff_after": [0, 0, 1],
            "balls_after": [1, 0, 0],
            "strikes_after": [0, 0, 0],
            "runner_on_first_after": [0, 0, 0],
            "runner_on_second_after": [0, 0, 0],
            "runner_on_third_after": [0, 0, 0],
        })

    def test_state_deltas_and_direction_diagnostics(self):
        prepared = add_before_and_delta_state(self.updates())
        self.assertEqual(prepared.iloc[1]["delta_outs"], 1)
        self.assertEqual(prepared.iloc[2]["delta_score_diff"], 1)
        report, summary = transition_diagnostics(self.updates())
        self.assertFalse(report.empty)
        self.assertEqual(summary["transitions"], 2)
        self.assertEqual(summary["comparable_direction_transitions"], 2)

    def test_acceptance_must_clear_probability_and_break_even(self):
        frame = pd.DataFrame({
            "breakeven_reversion_probability": [0.70, 0.90, 0.70],
        })
        accepted = accepted_state_reversions(
            frame,
            [0.80, 0.80, 0.69],
            StateReversionConfig(
                minimum_probability=0.60,
                minimum_break_even_margin=0.05,
            ),
        )
        self.assertEqual(accepted.tolist(), [True, False, False])

    def test_logit_residual_has_market_rich_direction(self):
        self.assertGreater(signed_logit_residual(0.70, 0.50), 0)
        self.assertLess(signed_logit_residual(0.30, 0.50), 0)

    def test_candidate_requires_persistence_and_later_exit_fill(self):
        updates = pd.DataFrame({
            "game_pk": [1], "game_date": ["2026-06-01"],
            "pitch_start_time": pd.to_datetime(["2026-06-01T12:00:00Z"]),
            "pitch_end_time": pd.to_datetime(["2026-06-01T12:00:10Z"]),
            "fair_before": [0.50], "fair_after": [0.50],
            "at_bat_number": [1], "pitch_number": [1],
            "inning_after": [1], "inning_topbot_after": [1],
            "outs_when_up_after": [0], "score_diff_after": [0],
            "balls_after": [1], "strikes_after": [0],
            "runner_on_first_after": [0], "runner_on_second_after": [0],
            "runner_on_third_after": [0],
        })
        times = pd.to_datetime([
            "2026-06-01T11:59:59Z",  # causal anchor
            "2026-06-01T12:00:11Z",  # starts persistence watch
            "2026-06-01T12:00:12Z",  # later compatible entry fill
            "2026-06-01T12:00:13Z",  # observes reversion
            "2026-06-01T12:00:14Z",  # later compatible exit fill
        ])
        trades = pd.DataFrame({
            "game_pk": [1] * 5, "created_time": times,
            "trade_id": range(5), "yes_price_dollars": [0.50, 0.70, 0.70, 0.50, 0.49],
            "no_price_dollars": [0.50, 0.30, 0.30, 0.50, 0.51],
            "count_fp": [100.0] * 5,
            "taker_outcome_side": ["yes", "no", "no", "yes", "yes"],
            "home_win": [0] * 5,
        })
        examples = build_state_overshoot_candidates(
            trades, updates,
            OvershootConfig(
                minimum_logit_residual=0.20,
                confirmation_seconds=1,
                maximum_entry_latency_seconds=5,
            ),
        )
        self.assertEqual(len(examples), 1)
        row = examples.iloc[0]
        self.assertEqual(row["side"], "no")
        self.assertEqual(row["entry_time"], times[2])
        self.assertEqual(row["exit_time"], times[4])
        self.assertEqual(row["exit_reason"], "reversion")

    def test_rejected_entry_releases_game_occupancy(self):
        frame = pd.DataFrame({
            "game_pk": [1, 1],
            "entry_time": pd.to_datetime([
                "2026-06-01T12:00:00Z", "2026-06-01T12:00:10Z",
            ]),
            "exit_time": pd.to_datetime([
                "2026-06-01T12:01:00Z", "2026-06-01T12:00:20Z",
            ]),
            "absolute_logit_residual": [0.2, 0.2],
            "target_reversion_pnl": [2.0, 2.0], "failure_pnl": [-10.0, -10.0],
            "pnl": [-10.0, 2.0], "fees": [0.5, 0.5], "entry_fee": [0.2, 0.2],
            "exit_reason": ["settlement", "reversion"],
        })
        result = simulate_state_reversion(
            frame, [0.1, 0.95],
            OvershootConfig(
                minimum_logit_residual=0.1,
                minimum_reversion_probability=0.5,
                minimum_expected_pnl=0,
            ),
        )
        self.assertEqual(result.accepted, 1)
        self.assertEqual(result.rejected, 1)
        self.assertEqual(result.pnl, 2.0)

    def test_two_stage_expected_pnl_overrides_static_payoff(self):
        frame = pd.DataFrame({
            "game_pk": [1],
            "entry_time": pd.to_datetime(["2026-06-01T12:00:00Z"]),
            "exit_time": pd.to_datetime(["2026-06-01T12:00:10Z"]),
            "absolute_logit_residual": [0.2],
            "target_reversion_pnl": [10.0], "failure_pnl": [-1.0],
            "pnl": [2.0], "fees": [0.5], "entry_fee": [0.2],
            "exit_reason": ["reversion"],
        })
        result = simulate_state_reversion(
            frame, [0.9],
            OvershootConfig(
                minimum_logit_residual=0.1,
                minimum_reversion_probability=0,
                minimum_expected_pnl=0,
            ),
            expected_pnls=[-0.01],
        )
        self.assertEqual(result.accepted, 0)
        self.assertEqual(result.rejected, 1)


if __name__ == "__main__":
    unittest.main()
