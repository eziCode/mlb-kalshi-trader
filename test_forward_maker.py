import unittest

from research.forward_maker import ForwardMakerGame
from research.maker_portfolio import CashAccount, PortfolioClock
from test_forward_value import ConstantModel, observation
from test_passive_research import snapshot, trade, delta


class ForwardMakerTests(unittest.TestCase):
    def engine(self, home_bias=.8):
        account = CashAccount.funded(100.)
        game = {"market_ticker": "M", "away_market_ticker": "P", "game_pk": 1}
        correction = {"coefficients": [0, 0, 0, home_bias, 0], "scales": [1]*5}
        e = ForwardMakerGame(game, account, ConstantModel(), .5, correction)
        e.observe(0., snapshot())
        peer = snapshot()
        peer["msg"]["market_ticker"] = "P"
        e.observe(0., peer)
        pre = trade("pre", "yes", .5, 10)
        pre["msg"]["exchange_elapsed_seconds"] = 9.
        e.observe(9., pre)
        e.baseball(12., {"game_pk": 1, "observation": observation()}, 0.)
        e.advance_clock(15.1)
        return e, account

    def test_observed_queue_must_trade_before_our_order_fills(self):
        e, _ = self.engine()
        e.observe(16., trade("a", "no", .45, 5))
        self.assertFalse(e.fills)
        e.observe(16.1, trade("b", "no", .45, 1))
        self.assertEqual(len(e.fills), 1)
        self.assertAlmostEqual(e.fills[0]["fee"], .0044)
        self.assertAlmostEqual(e.position, 1.)

    def test_book_cancellations_do_not_grant_queue_priority(self):
        e, _ = self.engine()
        e.observe(16., delta("yes", .45, -4))
        e.observe(16.1, trade("a", "no", .45, 1))
        self.assertFalse(e.fills)

    def test_adverse_trade_through_fills_at_our_resting_price(self):
        e, account = self.engine()
        e.observe(16., trade("a", "no", .44, 1))
        self.assertEqual(e.fills[0]["price"], .45)
        self.assertTrue(e.fills[0]["traded_through"])
        PortfolioClock([e], account).summary()

    def test_actual_away_market_supplies_away_fill_evidence(self):
        e, account = self.engine(-.8)
        e.observe(16., trade("wrong_market", "no", .44, 1))
        self.assertFalse(e.fills)
        event = trade("away", "no", .44, 1)
        event["msg"]["market_ticker"] = "P"
        e.observe(16.1, event)
        self.assertEqual(e.position, -1.)
        self.assertEqual(e.fills[0]["ticker"], "P")
        PortfolioClock([e], account).summary()

    def test_post_only_rejection_uses_arrival_book(self):
        e, _ = self.engine()
        e.observe(15.2, delta("no", .56, 1))
        e.advance_clock(15.7)
        self.assertFalse(e.orders)
        self.assertEqual(e.counters["maker_post_only_rejections"], 1)
        self.assertAlmostEqual(e.cash, 100.)

    def test_partial_fill_remains_exposed_during_cancellation(self):
        e, _ = self.engine()
        e.observe(16., trade("a", "no", .45, 5.4))
        e.advance_clock(20.1)
        e.observe(20.2, trade("b", "no", .44, .6))
        self.assertTrue(e.fills[-1]["cancel_pending"])
        self.assertAlmostEqual(e.position, 1.)
        self.assertEqual(e.summary()["entry_orders_filled"], 1)
        e.advance_clock(40.)
        self.assertEqual(e.counters["maker_orders"], 1)

    def test_old_exchange_trade_cannot_fill_after_late_receipt(self):
        e, _ = self.engine()
        event = trade("old", "no", .44, 1)
        event["msg"]["exchange_elapsed_seconds"] = 15.
        e.observe(16., event)
        self.assertFalse(e.fills)

    def test_incomplete_new_state_cancels_quote_without_erasing_race(self):
        e, _ = self.engine()
        data = observation()
        data["currentPlay"]["playEvents"][0]["details"] = {"isInPlay": True}
        e.baseball(19., {"game_pk": 1, "observation": data}, 0.)
        e.advance_clock(20.1)
        e.observe(20.2, trade("a", "no", .44, 1))
        self.assertTrue(e.fills[0]["cancel_pending"])


if __name__ == "__main__":
    unittest.main()
