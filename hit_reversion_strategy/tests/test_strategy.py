from __future__ import annotations

import unittest

import pandas as pd

from scripts.paper_trade import should_surface_worker_line
from trade_tape_strategy.core import (
    TradeTapeConfig,
    simulate_trade_tape,
    trade_signal,
)


class TradeTapeStrategyTests(unittest.TestCase):
    def test_main_log_surfaces_readiness_and_trades(self):
        self.assertTrue(should_surface_worker_line("TRADER READY game_pk=1"))
        self.assertTrue(should_surface_worker_line("TRADE SELL YES"))
        self.assertFalse(should_surface_worker_line("Initialized baseline"))

    def test_trade_signal_is_relative_to_yes_price_after_fees(self):
        self.assertEqual(trade_signal(0.60, 0.50, 0.05)[0], "yes")
        self.assertEqual(trade_signal(0.40, 0.50, 0.05)[0], "no")
        self.assertIsNone(trade_signal(0.52, 0.50, 0.05)[0])

    def test_trade_signal_rejects_gross_edge_consumed_by_fees(self):
        side, net_edge = trade_signal(0.56, 0.50, 0.05)
        self.assertIsNone(side)
        self.assertAlmostEqual(net_edge, 0.025)

    def test_position_has_no_time_based_exit(self):
        trades, updates = self._frames(include_reversion=False)
        result = simulate_trade_tape(
            trades, updates, TradeTapeConfig(minimum_edge=0.05)
        )
        self.assertEqual(result.trades, 1)
        self.assertEqual(result.reversion_exits, 0)
        self.assertEqual(result.settlements, 1)
        self.assertEqual(result.records[0].exit_reason, "settlement")
        self.assertEqual(result.records[0].trigger_at_bat, 1)
        self.assertEqual(
            result.records[0].trigger_event_time,
            pd.Timestamp("2026-07-01T12:00:01Z"),
        )

    def test_hit_without_meaningful_directional_fair_move_is_rejected(self):
        trades, updates = self._frames(include_reversion=False)
        updates["fair_after"] = updates["fair_before"]
        result = simulate_trade_tape(trades, updates, TradeTapeConfig())
        self.assertEqual(result.trades, 0)
        self.assertEqual(result.rejected_fair_updates, 1)

    def test_next_completed_plate_appearance_invalidates_candidate(self):
        trades, updates = self._frames(include_reversion=False)
        next_pa = updates.iloc[0].copy()
        next_pa["pitch_start_time"] = pd.Timestamp("2026-07-01T12:00:01.5Z")
        next_pa["pitch_end_time"] = pd.Timestamp("2026-07-01T12:00:02Z")
        next_pa["is_hit"] = False
        next_pa["completed_event"] = "field_out"
        next_pa["completed_event_batting_home"] = True
        next_pa["at_bat_number"] = 2
        next_pa["pitch_number"] = 1
        next_pa["fair_before"] = 0.50
        next_pa["fair_after"] = 0.48
        next_pa["outs_when_up_after"] = 1
        updates = pd.concat([updates, next_pa.to_frame().T], ignore_index=True)
        result = simulate_trade_tape(
            trades, updates, TradeTapeConfig(minimum_edge=0.05)
        )
        self.assertEqual(result.trades, 0)
        self.assertEqual(result.invalidated_candidates, 1)

    def test_next_pitch_invalidates_candidate_before_entry(self):
        trades, updates = self._frames(include_reversion=False)
        later_pitch = updates.iloc[0].copy()
        later_pitch["pitch_start_time"] = pd.Timestamp(
            "2026-07-01T12:00:01.5Z"
        )
        later_pitch["pitch_end_time"] = pd.Timestamp(
            "2026-07-01T12:00:03.25Z"
        )
        later_pitch["is_hit"] = False
        later_pitch["completed_event"] = None
        later_pitch["at_bat_number"] = 2
        later_pitch["pitch_number"] = 1
        updates = pd.concat(
            [updates, later_pitch.to_frame().T], ignore_index=True
        )

        result = simulate_trade_tape(
            trades,
            updates,
            TradeTapeConfig(minimum_edge=0.05),
        )
        self.assertEqual(result.trades, 0)
        self.assertEqual(result.invalidated_candidates, 1)

    def test_event_candidate_expires_before_late_fill(self):
        trades, updates = self._frames(include_reversion=False)
        trades.loc[
            trades["created_time"] > pd.Timestamp("2026-07-01T12:00:01Z"),
            "created_time",
        ] += pd.Timedelta(seconds=20)

        result = simulate_trade_tape(
            trades,
            updates,
            TradeTapeConfig(
                minimum_edge=0.05,
                maximum_event_to_entry_seconds=10.0,
            ),
        )
        self.assertEqual(result.trades, 0)
        self.assertEqual(result.expired_candidates, 1)

    def test_reversion_requires_a_later_compatible_trade(self):
        trades, updates = self._frames(include_reversion=True)
        result = simulate_trade_tape(
            trades, updates, TradeTapeConfig(minimum_edge=0.05)
        )
        self.assertEqual(result.trades, 1)
        self.assertEqual(result.reversion_exits, 1)
        self.assertEqual(result.settlements, 0)
        self.assertEqual(result.records[0].exit_reason, "reversion")

    def test_favorable_velocity_delays_exit_until_trailing_giveback(self):
        trades, updates = self._frames(include_reversion=False)
        momentum = pd.DataFrame({
            "game_pk": 1,
            "trade_id": ["m1", "m2", "m3", "m4", "m5"],
            "created_time": pd.to_datetime([
                "2026-07-01T12:00:03.8Z",
                "2026-07-01T12:00:04.2Z",
                "2026-07-01T12:00:04.6Z",
                "2026-07-01T12:00:05.0Z",
                "2026-07-01T12:00:05.1Z",
            ], utc=True),
            "yes_price_dollars": [0.47, 0.50, 0.54, 0.52, 0.52],
            "no_price_dollars": [0.53, 0.50, 0.46, 0.48, 0.48],
            "count_fp": 100.0,
            "taker_outcome_side": ["no", "no", "no", "no", "no"],
            "home_win": 1,
        })
        trades = pd.concat([trades, momentum], ignore_index=True)
        config = TradeTapeConfig(
            minimum_edge=0.05,
            momentum_exit_enabled=True,
            momentum_window_seconds=2.0,
            minimum_favorable_velocity=0.01,
            momentum_trailing_giveback=0.01,
            momentum_max_hold_seconds=2.0,
            minimum_momentum_trades=3,
        )
        result = simulate_trade_tape(trades, updates, config)
        self.assertEqual(result.momentum_exits, 1)
        self.assertEqual(result.records[0].exit_reason, "momentum_reversion")
        self.assertEqual(result.records[0].exit_price, 0.52)

    @staticmethod
    def _frames(include_reversion: bool):
        times = [
            "2026-07-01T11:59:59Z",  # safe pre-pitch anchor
            "2026-07-01T12:00:01.100Z",
            "2026-07-01T12:00:03.200Z",  # confirms persistence
            "2026-07-01T12:00:03.300Z",  # later compatible entry
            "2026-07-01T12:05:00Z",      # no timeout despite elapsed time
        ]
        prices = [0.40, 0.40, 0.40, 0.40, 0.41]
        taker = ["yes", "yes", "yes", "yes", "no"]
        if include_reversion:
            times += [
                "2026-07-01T12:05:01Z",  # observes reversion
                "2026-07-01T12:05:02Z",  # later compatible exit
            ]
            prices += [0.51, 0.51]
            taker += ["no", "no"]
        trades = pd.DataFrame({
            "game_pk": 1,
            "trade_id": [str(index) for index in range(len(times))],
            "created_time": pd.to_datetime(times, utc=True, format="mixed"),
            "yes_price_dollars": prices,
            "no_price_dollars": [1 - price for price in prices],
            "count_fp": 100.0,
            "taker_outcome_side": taker,
            "home_win": 1,
        })
        updates = pd.DataFrame({
            "game_pk": [1],
            "pitch_start_time": pd.to_datetime(
                ["2026-07-01T12:00:00Z"], utc=True
            ),
            "pitch_end_time": pd.to_datetime(
                ["2026-07-01T12:00:01Z"], utc=True
            ),
            "is_hit": [True],
            "completed_event": ["single"],
            "completed_event_batting_home": [True],
            "at_bat_number": [1],
            "pitch_number": [3],
            "fair_before": [0.40],
            "fair_after": [0.50],
            "score_diff_after": [0],
            "outs_when_up_after": [0],
            "runner_on_first_after": [1],
            "runner_on_second_after": [0],
            "runner_on_third_after": [0],
        })
        return trades, updates


if __name__ == "__main__":
    unittest.main()
