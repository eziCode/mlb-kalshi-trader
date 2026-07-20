from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from settlement_value_strategy.strategy import (
    MISPRICING_FEATURES, MispricingConfig, mispricing_feature_frame,
    market_adjusted_probability, model_signal, signal_economics, simulate_mispricing,
    simulate_away_yes,
)


class MispricingTests(unittest.TestCase):
    def test_identity_probability_transform_preserves_raw_forecast(self):
        actual = market_adjusted_probability(
            [.2, .8], [.7, .3], {"mode": "identity"}
        )
        np.testing.assert_allclose(actual, [.2, .8])

    def test_two_sided_policy_can_trade_yes_then_no_after_sixty_seconds(self):
        start = pd.Timestamp("2026-06-01T12:00:00Z")
        frame = pd.DataFrame({
            "game_pk": [1, 1], "signal_time": [start, start + pd.Timedelta(seconds=70)],
            "next_update_time": [
                start + pd.Timedelta(seconds=5), start + pd.Timedelta(seconds=75),
            ],
            "market_home_price": [.40, .60], "home_win": [1, 1],
        })
        trades = pd.DataFrame({
            "game_pk": [1, 1], "trade_id": [1, 2],
            "created_time": [
                start + pd.Timedelta(seconds=1), start + pd.Timedelta(seconds=71),
            ],
            "yes_price_dollars": [.41, .59],
            "no_price_dollars": [.59, .41], "count_fp": [100, 100],
            "taker_outcome_side": ["yes", "no"],
        })
        result = simulate_mispricing(
            frame, [.80, .20], trades,
            MispricingConfig(
                minimum_expected_pnl=0, minimum_probability_edge=0,
                side_filter="both", maximum_positions_per_game=3,
                minimum_seconds_between_entries=60,
            ),
        )
        self.assertEqual(result.trades, 2)
        self.assertEqual([row["side"] for row in result.records], ["yes", "no"])

    def test_away_execution_requires_eligible_no_model_signal(self):
        config = MispricingConfig(
            minimum_expected_pnl=.5, minimum_probability_edge=.04,
            side_filter="no", execution_contract="away_yes",
        )
        side, _, _, eligible = model_signal(.80, .70, config)
        self.assertEqual(side, "yes")
        self.assertFalse(eligible)

    def test_away_replay_rejects_ineligible_home_yes_signal(self):
        signal_time = pd.Timestamp("2026-06-01T12:00:00Z")
        frame = pd.DataFrame({
            "game_pk": [1], "signal_time": [signal_time],
            "next_update_time": [signal_time + pd.Timedelta(seconds=4)],
            "market_home_price": [.70], "home_win": [0],
        })
        trades = pd.DataFrame({
            "game_pk": [1, 1], "trade_id": [1, 2],
            "created_time": [
                signal_time - pd.Timedelta(seconds=1),
                signal_time + pd.Timedelta(seconds=1),
            ],
            "yes_price_dollars": [.10, .10], "count_fp": [100, 100],
            "taker_outcome_side": ["yes", "yes"],
        })
        result = simulate_away_yes(
            frame, [.80], trades,
            MispricingConfig(
                minimum_expected_pnl=.5, minimum_probability_edge=.04,
                side_filter="no", execution_contract="away_yes",
            ),
        )
        self.assertEqual(result.trades, 0)

    def test_feature_contract_is_event_agnostic(self):
        self.assertFalse(any("event" in name for name in MISPRICING_FEATURES))
        frame = pd.DataFrame({name: [0.0] for name in MISPRICING_FEATURES})
        frame["event_type"] = "double"
        self.assertNotIn("event_type", mispricing_feature_frame(frame))

    def test_signal_economics_selects_undervalued_yes(self):
        yes_ev, no_ev = signal_economics(0.70, 0.40)
        self.assertGreater(yes_ev, 0)
        self.assertLess(no_ev, 0)

    def test_fill_is_strictly_later_and_holds_to_settlement(self):
        signal_time = pd.Timestamp("2026-06-01T12:00:00Z")
        frame = pd.DataFrame({
            "game_pk": [1], "signal_time": [signal_time],
            "next_update_time": [signal_time + pd.Timedelta(seconds=4)],
            "market_home_price": [0.40], "home_win": [1],
        })
        trades = pd.DataFrame({
            "game_pk": [1, 1], "trade_id": [1, 2],
            "created_time": [signal_time, signal_time + pd.Timedelta(seconds=1)],
            "yes_price_dollars": [0.40, 0.41],
            "no_price_dollars": [0.60, 0.59],
            "count_fp": [100.0, 100.0],
            "taker_outcome_side": ["yes", "yes"],
        })
        result = simulate_mispricing(
            frame, [0.80], trades,
            MispricingConfig(
                minimum_expected_pnl=0, minimum_probability_edge=0,
            ),
        )
        self.assertEqual(result.trades, 1)
        self.assertEqual(result.records[0]["fill_time"], trades.created_time.iloc[1])
        self.assertGreater(result.pnl, 0)

    def test_away_view_routes_to_paired_yes_and_respects_game_cap(self):
        signal_time = pd.Timestamp("2026-06-01T12:00:00Z")
        frame = pd.DataFrame({
            "game_pk": [1, 1],
            "signal_time": [signal_time, signal_time + pd.Timedelta(seconds=2)],
            "next_update_time": [
                signal_time + pd.Timedelta(seconds=2),
                signal_time + pd.Timedelta(seconds=4),
            ],
            "market_home_price": [.60, .60], "home_win": [0, 0],
        })
        trades = pd.DataFrame({
            "game_pk": [1, 1, 1], "trade_id": [1, 2, 3],
            "created_time": [
                signal_time - pd.Timedelta(seconds=1),
                signal_time + pd.Timedelta(seconds=1),
                signal_time + pd.Timedelta(seconds=3),
            ],
            "yes_price_dollars": [.35, .36, .36],
            "count_fp": [100.0, 100.0, 100.0],
            "taker_outcome_side": ["yes", "yes", "yes"],
        })
        result = simulate_away_yes(
            frame, [.40, .40], trades,
            MispricingConfig(
                minimum_expected_pnl=0, minimum_probability_edge=0,
                maximum_positions_per_game=1,
                minimum_seconds_between_entries=0,
            ),
        )
        self.assertEqual(result.trades, 1)
        self.assertEqual(result.yes_trades, 1)
        self.assertEqual(result.records[0]["execution_contract"], "away_yes")
        self.assertGreater(result.pnl, 0)


if __name__ == "__main__":
    unittest.main()
