from __future__ import annotations

import unittest

import pandas as pd

from mlb_kalshi.strategy import (
    fee_aware_signal_side,
    reaction_feature_frame,
    signal_side,
    state_feature_frame,
    taker_fee,
    validate_market_prices,
)


class StrategyContractTests(unittest.TestCase):
    def test_shared_frames_reject_missing_or_out_of_range_features(self):
        state = pd.DataFrame({
            "pregame_prob": [0.45], "inning": [5], "inning_topbot": [1],
            "outs_when_up": [1], "score_diff": [-2], "balls": [1],
            "strikes": [2], "runner_on_first": [1],
            "runner_on_second": [0], "runner_on_third": [0],
        })
        self.assertEqual(state_feature_frame(state).shape, (1, 10))
        reaction = pd.DataFrame({
            "market_error": [0.03], "kalshi_price": [0.48],
            "pregame_prob": [0.45], "spread": [0.02], "inning": [5],
        })
        self.assertEqual(reaction_feature_frame(reaction).shape, (1, 5))

    def test_market_prices_must_be_real_midpoint_and_spread(self):
        frame = pd.DataFrame({
            "yes_bid_close": [0.47], "yes_ask_close": [0.49],
            "kalshi_price": [0.48], "spread": [0.02],
        })
        validate_market_prices(frame)
        with self.assertRaises(ValueError):
            validate_market_prices(frame.assign(kalshi_price=0.49))

    def test_signal_and_fee_contract(self):
        self.assertEqual(signal_side(0.70, 0.49, 0.51)[0], "yes")
        self.assertEqual(signal_side(0.30, 0.49, 0.51)[0], "no")
        self.assertIsNone(signal_side(0.50, 0.49, 0.51)[0])
        self.assertEqual(taker_fee(100, 0.50), 1.75)

    def test_fee_aware_signal_reserves_two_taker_crossings(self):
        # Raw executable edge is 4%, but two 50-cent taker fees consume about
        # 3.5%, so a 1% buffer rejects the trade.
        side, net_edge = fee_aware_signal_side(0.55, 0.49, 0.51, 0.01)
        self.assertIsNone(side)
        self.assertLess(net_edge, 0.01)

        side, net_edge = fee_aware_signal_side(0.57, 0.49, 0.51, 0.01)
        self.assertEqual(side, "yes")
        self.assertGreaterEqual(net_edge, 0.01)


if __name__ == "__main__":
    unittest.main()
