import unittest

import numpy as np
import pandas as pd

from research.continuous import NS
from research.settlement_study import simulate


def row(game=1, when=1., filled=2., price=.5, limit=.51, fair=.6):
    return dict(game_pk=game, time=int(when * NS), market=0, ticker=f"M{game}",
                buy_limit=limit, buy_time=int(filled * NS), buy_price=price,
                mid=.5, fair_gap=fair - .5, entry_allowed=True)


class SettlementStudyTests(unittest.TestCase):
    def test_cash_recycles_only_after_actual_settlement_and_delay(self):
        frame = pd.DataFrame([row(), row(2, 5., 6.), row(2, 71., 72.)])
        terminal = {"M1": {"time": 10 * NS, "value": 1.}, "M2": {"time": 100 * NS, "value": 1.}}
        trades, counters = simulate(frame, 0., terminal, starting_cash=.6)
        self.assertEqual(len(trades), 2)
        self.assertEqual(counters["cash_rejected_orders"], 1)
        self.assertEqual(trades.iloc[1].entry_decision_time.value, 71 * NS)
        self.assertAlmostEqual(counters["ending_equity"], .6 + trades.pnl.sum())

    def test_missing_settlement_keeps_inventory_and_cost(self):
        frame = pd.DataFrame([row(), row(2, 100., 101.)])
        trades, counters = simulate(frame, 0., {}, starting_cash=.6)
        self.assertEqual(len(trades), 1)
        self.assertEqual(counters["unresolved_positions"], 1)
        self.assertEqual(trades.iloc[0].exit_reason, "unresolved_zero_mark")
        self.assertLess(trades.iloc[0].pnl, -.5)

    def test_future_fill_failure_does_not_change_order_submission(self):
        missed = pd.DataFrame([row(filled=0, price=np.nan)])
        filled = pd.DataFrame([row()])
        a, ca = simulate(missed, 0., {})
        b, cb = simulate(filled, 0., {})
        self.assertEqual(ca["entry_orders"], cb["entry_orders"])
        self.assertEqual(ca["expired_entries"], 1)
        self.assertEqual((len(a), len(b)), (0, 1))

    def test_no_new_entry_after_same_game_settles(self):
        frame = pd.DataFrame([row(), row(when=100., filled=101.)])
        trades, counters = simulate(frame, 0., {"M1": {"time": 10 * NS, "value": 1.}})
        self.assertEqual(counters["entry_orders"], 1)
        self.assertEqual(len(trades), 1)

    def test_terminal_value_does_not_determine_entry(self):
        frame = pd.DataFrame([row()])
        win, wc = simulate(frame, 0., {"M1": {"time": 10 * NS, "value": 1.}})
        loss, lc = simulate(frame, 0., {"M1": {"time": 10 * NS, "value": 0.}})
        self.assertEqual(wc["entry_orders"], lc["entry_orders"])
        self.assertEqual(win.iloc[0].entry_price, loss.iloc[0].entry_price)
        self.assertAlmostEqual(win.pnl.sum() - loss.pnl.sum(), 1.)

    def test_executable_limit_and_fee_must_fit_estimated_edge(self):
        trades, counter = simulate(pd.DataFrame([row(fair=.52)]), 0., {})
        self.assertEqual(counter["entry_orders"], 0)
        self.assertTrue(trades.empty)

    def test_fill_after_settlement_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "after exchange settlement"):
            simulate(pd.DataFrame([row()]), 0., {"M1": {"time": NS, "value": 1.}})


if __name__ == "__main__":
    unittest.main()
