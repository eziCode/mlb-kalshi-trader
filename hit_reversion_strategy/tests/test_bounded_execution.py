from dataclasses import replace
import asyncio
import os
import unittest
from unittest.mock import patch

import pandas as pd
import test_strategy as fixtures
from scripts import paper_trade
from scripts.backtest import apply_live_paired_execution_prices
from trade_tape_strategy.core import TradeTapeConfig, simulate_trade_tape, position_contracts


class BoundedExecutionTests(unittest.TestCase):
    def test_paired_market_print_at_identical_time_is_not_observable(self):
        home = pd.DataFrame([{"game_pk": 1, "trade_id": "h", "created_time": pd.Timestamp("2026-09-01T12:00:00Z"),
                              "yes_price_dollars": .5, "no_price_dollars": .5, "count_fp": 10,
                              "taker_outcome_side": "yes"}])
        away = home.copy()
        away["trade_id"] = "a"
        paired = apply_live_paired_execution_prices(home, away)
        self.assertTrue(paired.no_price_dollars.isna().all())
        self.assertTrue(paired.home_market_observation.all())

    def test_paper_override_cannot_enable_disabled_live_policy(self):
        with patch.object(paper_trade, "GAME_PK", "1"), patch.object(paper_trade, "MARKET_TICKER", "TEST"), \
             patch.object(paper_trade, "LIVE_MODE", True), patch.dict(os.environ, {"ALLOW_UNVALIDATED_HYBRID": "1"}), \
             patch.object(paper_trade, "LiveExecutor") as executor:
            with self.assertRaisesRegex(RuntimeError, "disabled for real-money"):
                asyncio.run(paper_trade.main())
            executor.assert_not_called()

    def frames(self):
        trades, updates = fixtures.TradeTapeStrategyTests._frames(False)
        trades = trades.iloc[:3].copy()
        trades.loc[2, "created_time"] = pd.Timestamp("2026-07-01T12:00:01.850Z")
        return trades, updates

    def config(self, **overrides):
        return replace(TradeTapeConfig(
            confirmation_seconds=0, execution_window_seconds=.25,
            entry_submission_latency_seconds=.68,
            exit_submission_latency_seconds=.68,
            position_sizing="fixed_budget", order_budget=2.5,
            execution_price_penalty=.01,
        ), **overrides)

    def test_order_does_not_wait_for_a_late_print(self):
        trades, updates = self.frames()
        trades.loc[2, "created_time"] += pd.Timedelta(seconds=1)
        result = simulate_trade_tape(trades, updates, self.config())
        self.assertEqual(result.trades, 0)
        self.assertEqual(result.expired_entry_orders, 1)

    def test_entry_price_and_quantity_are_frozen_at_submission(self):
        trades, updates = self.frames()
        trades.loc[2, ["yes_price_dollars", "no_price_dollars"]] = [.38, .62]
        config = self.config()
        result = simulate_trade_tape(trades, updates, config)
        self.assertEqual(result.trades, 1)
        self.assertAlmostEqual(result.records[0].entry_price, .39)
        self.assertEqual(result.records[0].contracts, position_contracts(.41, config))

    def test_later_price_cannot_raise_the_order_limit(self):
        trades, updates = self.frames()
        trades.loc[2, ["yes_price_dollars", "no_price_dollars"]] = [.42, .58]
        self.assertEqual(simulate_trade_tape(trades, updates, self.config()).trades, 0)

    def test_inflight_entry_survives_adverse_state_and_next_pitch(self):
        trades, updates = self.frames()
        later = updates.iloc[0].copy()
        later["pitch_start_time"] += pd.Timedelta(seconds=1.2)
        later["pitch_end_time"] += pd.Timedelta(seconds=.5)
        later["pitch_number"] = 4
        later["completed_event"] = None
        later["fair_after"] = .2
        result = simulate_trade_tape(trades, pd.concat([updates, later.to_frame().T]), self.config())
        self.assertEqual(result.trades, 1)

    def test_model_is_not_rescored_using_a_future_fill(self):
        class Scorer:
            def accepts(self, row):
                return row["entry_lag_seconds"] < .5, 1.0
        trades, updates = self.frames()
        self.assertEqual(simulate_trade_tape(trades, updates, self.config(), Scorer()).trades, 1)

    def test_volume_participation_caps_partial_fill(self):
        trades, updates = self.frames()
        trades.loc[2, "count_fp"] = 5
        result = simulate_trade_tape(trades, updates, self.config(maximum_volume_participation=.1))
        self.assertEqual(result.records[0].contracts, .5)

    def test_state_at_exact_decision_time_is_not_observable(self):
        trades, updates = self.frames()
        updates.loc[0, "event_available_time"] = trades.loc[1, "created_time"]
        self.assertEqual(simulate_trade_tape(trades, updates, self.config()).trades, 0)

    def test_inflight_exit_is_not_cancelled_by_future_target(self):
        trades, updates = self.frames()
        for index, timestamp in ((3, "2026-07-01T12:00:05.100Z"), (4, "2026-07-01T12:00:05.850Z")):
            row = trades.iloc[-1].copy()
            row["created_time"] = pd.Timestamp(timestamp)
            row["trade_id"] = str(index)
            row["yes_price_dollars"], row["no_price_dollars"] = .51, .49
            row["taker_outcome_side"] = "no"
            trades.loc[index] = row
        later = updates.iloc[0].copy()
        later["pitch_start_time"] = pd.Timestamp("2026-07-01T12:00:05.200Z")
        later["pitch_end_time"] = pd.Timestamp("2026-07-01T12:00:05.300Z")
        later["completed_event"] = None
        later["fair_after"] = .8
        result = simulate_trade_tape(trades, pd.concat([updates, later.to_frame().T]), self.config())
        self.assertEqual(result.reversion_exits, 1)
        self.assertAlmostEqual(result.records[0].exit_price, .50)

    def test_invalid_execution_assumptions_fail_closed(self):
        trades, updates = self.frames()
        for changes in ({"execution_window_seconds": float("nan")}, {"maximum_volume_participation": 2},
                        {"execution_price_penalty": -.01}, {"require_post_signal_trade": False}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                simulate_trade_tape(trades, updates, self.config(**changes))
