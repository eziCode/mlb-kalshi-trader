import unittest

import numpy as np

from research.forward_value import ForwardValueGame, observed_state
from research.maker_portfolio import CashAccount, PortfolioClock
from test_passive_research import snapshot, delta, trade


class ConstantModel:
    def predict_proba(self, frame):
        return np.tile([.5, .5], (len(frame), 1))


def observation():
    return {"gameData": {"status": {"abstractGameState": "Live"},
             "teams": {"home": {"id": 1}, "away": {"id": 2}}},
        "linescore": {"currentInning": 1, "isTopInning": True, "outs": 0, "balls": 1, "strikes": 0,
            "teams": {"home": {"runs": 0}, "away": {"runs": 0}}, "offense": {"team": {"id": 2}}},
        "currentPlay": {"atBatIndex": 0, "playEvents": [{"isPitch": True, "pitchNumber": 1, "index": 0,
            "startTime": "1970-01-01T00:00:10Z", "endTime": "1970-01-01T00:00:11Z"}]}}


class ForwardValueTests(unittest.TestCase):
    def engine(self, home_bias=.8):
        account = CashAccount.funded(100.)
        game = {"market_ticker": "M", "away_market_ticker": "P", "game_pk": 1}
        correction = {"coefficients": [0, 0, 0, home_bias, 0], "scales": [1]*5}
        e = ForwardValueGame(game, account, ConstantModel(), .5, correction)
        e.observe(0., snapshot())
        peer = snapshot()
        peer["msg"]["market_ticker"] = "P"
        e.observe(0., peer)
        return e, account

    def live(self, e):
        event = trade("pre", "yes", .5, 10)
        event["msg"]["exchange_elapsed_seconds"] = 9.
        e.observe(9., event)
        e.baseball(12., {"game_pk": 1, "observation": observation()}, 0.)
        e.advance_clock(15.1)

    def test_no_entry_without_pregame_anchor(self):
        e, _ = self.engine()
        e.baseball(12., {"game_pk": 1, "observation": observation()}, 0.)
        e.advance_clock(20.)
        self.assertEqual(e.counters["value_orders"], 0)

    def test_first_pitch_does_not_accept_later_trade_as_pregame(self):
        e, _ = self.engine()
        event = trade("late", "yes", .5, 10)
        event["msg"]["exchange_elapsed_seconds"] = 11.
        e.observe(11., event)
        e.baseball(12., {"game_pk": 1, "observation": observation()}, 0.)
        self.assertIsNone(e.anchor_offset)

    def test_ioc_waits_for_arrival_and_uses_frozen_limit(self):
        e, _ = self.engine()
        self.live(e)
        self.assertEqual(len(e.fills), 0)
        e.observe(15.2, delta("no", .53, -5))
        e.observe(15.3, delta("no", .40, 5))
        e.advance_clock(15.7)
        self.assertEqual(len(e.fills), 0)
        self.assertEqual(e.counters["value_misses"], 1)
        self.assertAlmostEqual(e.cash, 100.)

    def test_adverse_state_does_not_erase_submitted_ioc(self):
        e, _ = self.engine()
        self.live(e)
        changed = observation()
        changed["gameData"]["status"]["abstractGameState"] = "Final"
        e.baseball(15.2, {"game_pk": 1, "observation": changed}, 0.)
        e.advance_clock(15.7)
        self.assertEqual(e.summary()["entry_orders_filled"], 1)
        self.assertAlmostEqual(e.fills[0]["price"], .48)

    def test_partial_ioc_keeps_inventory_and_releases_unused_cash(self):
        e, account = self.engine()
        self.live(e)
        e.observe(15.2, delta("no", .53, -4.6))
        e.advance_clock(15.7)
        self.assertAlmostEqual(e.position, .4)
        self.assertEqual(e.reserved, 0.)
        self.assertEqual(e.counters["value_partial_fills"], 1)
        clock = PortfolioClock([e], account)
        self.assertLess(clock.summary()["liquidation_marked_pnl"], 0.)
        e.advance_clock(100.)
        self.assertEqual(e.counters["value_orders"], 1)

    def test_away_inventory_is_marked_on_away_market(self):
        e, account = self.engine(-.8)
        self.live(e)
        e.advance_clock(15.7)
        self.assertEqual(e.position, -1.)
        event = delta("yes", .45, -5)
        event["msg"]["market_ticker"] = "P"
        e.observe(16., event)
        event = delta("yes", .30, 5)
        event["msg"]["market_ticker"] = "P"
        e.observe(16.1, event)
        self.assertLess(e.summary()["inventory_liquidation_mark"], .30)
        PortfolioClock([e], account).summary()  # Account reconciliation must pass.

    def test_inconsistent_batting_identity_disables_state(self):
        data = observation()
        data["linescore"]["offense"]["team"]["id"] = 1
        self.assertIsNone(observed_state(data))

    def test_incomplete_ball_in_play_disables_new_signal(self):
        data = observation()
        data["currentPlay"]["playEvents"][0]["details"] = {"isInPlay": True}
        self.assertIsNone(observed_state(data))

    def test_new_foul_pitch_refreshes_state_but_heartbeat_does_not(self):
        e, _ = self.engine()
        self.live(e)
        prior = e.state_seen
        e.baseball(20., {"game_pk": 1, "observation": None}, 0.)
        self.assertEqual(e.state_seen, prior)
        data = observation()
        data["currentPlay"]["playEvents"][0]["endTime"] = "1970-01-01T00:00:21Z"
        e.baseball(22., {"game_pk": 1, "observation": data}, 0.)
        self.assertEqual(e.state_seen, 22.)


if __name__ == "__main__":
    unittest.main()
