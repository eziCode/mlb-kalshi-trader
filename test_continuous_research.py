"""Causality and portfolio invariants for continuous research."""
import unittest
from dataclasses import replace
from datetime import date
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

from research.continuous import (
    Execution, Policy, NS, FEATURES, fill_evidence, prior_values, simulate,
    game_frame, causal_states, metrics,
    check,
)


def decisions(rows):
    result = []
    for changes in rows:
        row = dict(game_pk=1, game_date=date(2026, 9, 16), time=100 * NS, market=0,
                   ticker="HOME", outcome=1, mid=.5, ask=.5, bid=.5, spread=0.,
                   fair_gap=.2, entry_allowed=True, buy_limit=.51, sell_limit=.49,
                   buy_time=0, buy_price=np.nan, sell_time=0, sell_price=np.nan)
        row.update(changes)
        result.append(row)
    return pd.DataFrame(result).sort_values(["time", "game_pk", "market"]).reset_index(drop=True)


class ContinuousResearchTests(unittest.TestCase):
    def test_final_check_rejects_changed_development_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "manifest.json").write_text(json.dumps({"hashes": {"research/continuous.py": "changed"}}))
            with self.assertRaisesRegex(RuntimeError, "Inputs changed"):
                check(SimpleNamespace(output_dir=path, workers=1))

    def test_features_do_not_change_when_future_prices_or_winner_change(self):
        tape = pd.DataFrame({"game_pk": [1] * 10, "game_date": [date(2026, 9, 16)] * 10,
            "market_ticker": ["HOME"] * 10, "home_win": [1] * 10,
            "created_time": pd.to_datetime([101, 102, 111, 112, 121, 122, 131, 132, 151, 152], unit="s", utc=True),
            "trade_id": list(map(str, range(10))), "yes_price_dollars": [.51, .49] * 5,
            "count_fp": [100.] * 10, "taker_outcome_side": ["yes", "no"] * 5})
        state = pd.DataFrame({"game_pk": [1] * 3, "at_bat_number": [1, 1, 1], "pitch_number": [1, 2, 3],
            "pitch_start_time": pd.to_datetime([100, 120, 140], unit="s", utc=True),
            "pitch_end_time": pd.to_datetime([102, 122, 142], unit="s", utc=True),
            "completed_event": [None, None, "field_out"], "atomic_play_input": [True] * 3,
            "fair_before": [.5, .51, .52], "fair_after": [.51, .52, .53],
            "inning_after": [1.] * 3, "score_diff_after": [0.] * 3, "outs_when_up_after": [0] * 3})
        original = game_frame(tape, tape.iloc[:0], state, Execution(), False)
        altered_tape = tape.copy()
        altered_tape.loc[altered_tape.created_time > pd.Timestamp(125, unit="s", tz="UTC"), "yes_price_dollars"] = .8
        altered_tape["home_win"] = 0
        altered_state = state.copy()
        altered_state.loc[2, "fair_after"] = .99
        changed = game_frame(altered_tape, altered_tape.iloc[:0], altered_state, Execution(), False)
        mask = original.time < 125 * NS
        pd.testing.assert_frame_equal(original.loc[mask, FEATURES], changed.loc[mask, FEATURES])

    def test_exact_time_and_stale_observations_are_not_features(self):
        result = prior_values(np.array([10, 20]) * NS, np.array([.4, .8]),
                              np.array([10, 20, 21, 35]) * NS, 10)
        np.testing.assert_allclose(result, [np.nan, .4, .8, np.nan], equal_nan=True)

    def test_bounded_same_market_same_side_evidence_and_frozen_limit(self):
        execution = Execution()
        sides = {"yes": (np.array([100.68, 100.70, 100.75, 100.95]) * NS,
                         np.array([.40, .51, .50, .40]), np.array([100, 100, 9, 100])),
                 "no": (np.array([100.8]) * NS, np.array([.4]), np.array([100]))}
        time, _ = fill_evidence(sides, np.array([100 * NS]), np.array([.51]), execution, True)
        self.assertEqual(time[0], 0)  # equal arrival, too expensive, too small, too late
        sides["yes"][2][2] = 10
        time, price = fill_evidence(sides, np.array([100 * NS]), np.array([.51]), execution, True)
        self.assertEqual(time[0], int(100.75 * NS))
        self.assertAlmostEqual(price[0], .51)

    def test_pending_buy_survives_adverse_later_signal(self):
        frame = decisions([dict(buy_time=int(100.8 * NS), buy_price=.51),
                           dict(time=105 * NS, fair_gap=-.4)])
        trades, counters = simulate(frame, Policy("state_value", 60, 0), Execution())
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0].exit_reason, "settlement")
        self.assertEqual(counters["entry_orders"], 1)

    def test_do_not_choose_market_using_future_fill(self):
        frame = decisions([dict(fair_gap=.3), dict(market=1, ticker="AWAY", fair_gap=.2,
                                                  buy_time=int(100.8 * NS), buy_price=.51)])
        trades, counters = simulate(frame, Policy("state_value", 60, 0), Execution())
        self.assertEqual(len(trades), 0)
        self.assertEqual(counters["expired_entries"], 1)

    def test_pending_orders_reserve_cash_before_fill(self):
        frame = decisions([dict(), dict(game_pk=2, buy_time=int(100.8 * NS), buy_price=.51)])
        trades, counters = simulate(frame, Policy("state_value", 60, 0), Execution(), starting_cash=.6)
        self.assertEqual(len(trades), 0)
        self.assertEqual(counters["cash_rejected_orders"], 1)
        self.assertEqual(counters["expired_entries"], 1)

    def test_cash_rejection_can_retry_after_pending_expiry(self):
        frame = decisions([dict(), dict(game_pk=2),
                           dict(game_pk=2, time=105 * NS, buy_time=int(105.8 * NS), buy_price=.51)])
        trades, counters = simulate(frame, Policy("state_value", 60, 0), Execution(), starting_cash=.6)
        self.assertEqual(trades.game_pk.tolist(), [2])
        self.assertEqual(counters["cash_rejected_orders"], 1)

    def test_exit_needs_own_market_and_fresh_delay(self):
        frame = decisions([dict(buy_time=int(100.8 * NS), buy_price=.51),
            dict(time=165 * NS, entry_allowed=False),
            dict(time=165 * NS, market=1, sell_time=int(165.8 * NS), sell_price=.8, entry_allowed=False),
            dict(time=170 * NS, sell_time=int(170.8 * NS), sell_price=.6, entry_allowed=False)])
        trades, counters = simulate(frame, Policy("state_value", 60, 0), Execution())
        self.assertEqual(counters["expired_exits"], 1)
        self.assertEqual(counters["exit_orders"], 2)
        self.assertEqual(trades.iloc[0].exit_price, .6)
        self.assertEqual(trades.iloc[0].exit_time.value, int(170.8 * NS))
        self.assertAlmostEqual(trades.iloc[0].pnl, .6 - .51 - trades.iloc[0].fees)

    def test_no_recycling_settlement_cash(self):
        frame = decisions([dict(buy_time=int(100.8 * NS), buy_price=.51),
                           dict(game_pk=2, time=1000 * NS, buy_time=int(1000.8 * NS), buy_price=.51)])
        trades, counters = simulate(frame, Policy("state_value", 60, 0), Execution(), starting_cash=.6)
        self.assertEqual(len(trades), 1)
        self.assertEqual(counters["cash_rejected_orders"], 1)
        self.assertFalse(counters["settlement_cash_reused"])

    def test_equal_time_exit_proceeds_cannot_fund_entry(self):
        # Inject a synchronized sale to directly test event ordering.
        frame = decisions([dict(buy_time=int(100.8 * NS), buy_price=.51),
            dict(time=165 * NS, sell_time=170 * NS, sell_price=.8, entry_allowed=False),
            dict(game_pk=2, time=170 * NS, buy_time=int(170.8 * NS), buy_price=.51)])
        trades, counters = simulate(frame, Policy("state_value", 60, 0), Execution(), starting_cash=.6)
        self.assertEqual(trades.game_pk.tolist(), [1])
        self.assertEqual(counters["cash_rejected_orders"], 1)

    def test_zero_trade_metrics(self):
        frame = decisions([dict(fair_gap=-1)])
        trades, counters = simulate(frame, Policy("state_value", 60, 0), Execution())
        summary = metrics(frame, trades, counters, date(2026, 9, 16), date(2026, 9, 26))
        self.assertEqual(summary["entries_per_game"], 0)
        self.assertFalse(summary["every_game_traded"])

    def test_features_exclude_outcomes_and_evidence(self):
        self.assertFalse(set(FEATURES) & {"outcome", "game_pk", "game_date", "buy_time", "sell_time", "buy_price", "sell_price"})

    def test_invalid_execution_rejected(self):
        for changes in ({"latency": -1}, {"latency": 5}, {"participation": 0}, {"penalty": np.nan}):
            with self.assertRaises(ValueError):
                replace(Execution(), **changes)


if __name__ == "__main__":
    unittest.main()
