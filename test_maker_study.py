import unittest
from dataclasses import replace

from research.maker_study import MakerStudyReplay, StudyConfig, candidates
from test_passive_research import snapshot, trade, delta


def baseball(live=True, phase="Top", outs=0, end="1970-01-01T00:00:00Z", inning=1):
    return {"game_pk": 1, "observation": {
        "gameData": {"status": {"abstractGameState": "Live" if live else "Preview"}},
        "linescore": {"currentInning": inning, "inningState": phase},
        "currentPlay": {"count": {"outs": outs}, "about": {"endTime": end},
                        "playEvents": [{"isPitch": True}]}}}


class MakerStudyTests(unittest.TestCase):
    def engine(self, **changes):
        e = MakerStudyReplay("M", "PEER", 1, replace(StudyConfig(), **changes))
        e.baseball(0., baseball(), 0.)
        e.observe(0., snapshot())
        return e

    def test_twelve_candidates_declared(self):
        self.assertEqual(len(candidates()), 12)
        self.assertTrue(all(c.maker_fee_rate == .0175 for c in candidates().values()))

    def test_no_opening_orders_before_live_observation(self):
        e = MakerStudyReplay("M", "PEER", 1)
        e.observe(0., snapshot())
        self.assertEqual(len(e.orders), 0)
        e.baseball(.1, baseball(False), 0.)
        e.observe(2., {"type": "heartbeat"})
        self.assertEqual(len(e.orders), 0)
        e.baseball(2.1, baseball(), 0.)
        e.observe(3.1, {"type": "heartbeat"})
        self.assertEqual(len(e.orders), 2)

    def test_mlb_warmup_live_label_does_not_allow_pregame_quotes(self):
        e = MakerStudyReplay("M", "PEER", 1)
        data = baseball()
        data["observation"]["gameData"]["status"]["detailedState"] = "Warmup"
        data["observation"]["currentPlay"]["playEvents"] = [{"isPitch": False}]
        e.baseball(0., data, 0.)
        e.observe(0., snapshot())
        self.assertFalse(e.orders)
        self.assertFalse(e.game_live)

    def test_break_clock_uses_source_end_and_does_not_reset_on_poll(self):
        e = MakerStudyReplay("M", "PEER", 1, StudyConfig(entry_regime="break"))
        e.baseball(10., baseball(phase="Middle", outs=3), 0.)
        self.assertTrue(e.opening_allowed())
        e.baseball(40., baseball(phase="Middle", outs=3), 0.)
        self.assertEqual(e.break_since, 0.)
        self.assertTrue(e.opening_allowed())
        e.baseball(46., {"game_pk": 1, "observation": None}, 0.)
        self.assertFalse(e.opening_allowed())

    def test_missing_break_timestamp_disables_break_entry(self):
        e = MakerStudyReplay("M", "PEER", 1, StudyConfig(entry_regime="break"))
        e.baseball(10., baseball(phase="Middle", outs=3, end=None), 0.)
        self.assertFalse(e.opening_allowed())

    def test_stale_baseball_feed_cancels_opening_quotes(self):
        e = self.engine(maximum_feed_age=2.)
        e.observe(3.1, {"type": "heartbeat"})
        self.assertTrue(all(o["cancelling"] for o in e.orders.values()))

    def test_forced_exit_waits_for_cancellation_and_submission(self):
        e = self.engine(maximum_inventory_seconds=2.)
        e.observe(1., trade("a", "no", .45, 6))
        e.observe(5., {"type": "heartbeat"})
        exits = [f for f in e.fills if f["liquidity_role"] == "taker"]
        self.assertEqual(len(exits), 1)
        self.assertAlmostEqual(exits[0]["received_seconds"], 4.68)
        self.assertAlmostEqual(exits[0]["price"], .44)
        self.assertEqual(e.position, 0)

    def test_pending_cancel_fill_closes_inventory_without_duplicate_exit(self):
        e = self.engine(maximum_inventory_seconds=2.)
        e.observe(1., trade("a", "no", .45, 6))
        e.observe(3.2, trade("b", "yes", .47, 6))
        e.observe(5., {"type": "heartbeat"})
        self.assertEqual(e.counters["liquidation_orders"], 0)
        self.assertTrue(e.fills[-1]["cancel_pending"])
        self.assertEqual(e.position, 0)

    def test_liquidation_limit_is_frozen_before_adverse_update(self):
        e = self.engine(maximum_inventory_seconds=2.)
        e.observe(1., trade("a", "no", .45, 6))
        e.observe(4.1, delta("yes", .45, -5))
        e.observe(4.2, delta("yes", .30, 5))
        e.observe(4.8, {"type": "heartbeat"})
        self.assertEqual(e.counters["liquidation_misses"], 1)
        self.assertEqual(e.position, 1)

    def test_partial_exit_cannot_reuse_unchanged_visible_depth(self):
        e = self.engine(maximum_inventory_seconds=2.)
        e.observe(1., trade("a", "no", .45, 6))
        e.observe(4.1, delta("yes", .45, -4.6))
        e.observe(5.9, {"type": "heartbeat"})
        self.assertAlmostEqual(e.position, .6)
        self.assertEqual(e.counters["liquidation_misses"], 1)
        e.observe(6.1, delta("yes", .45, .6))
        e.observe(6.8, {"type": "heartbeat"})
        self.assertAlmostEqual(e.position, 0.)

    def test_markouts_do_not_change_entry_and_include_costs(self):
        e = self.engine(maximum_inventory_seconds=100.)
        e.observe(1., trade("a", "no", .45, 6))
        e.observe(6.1, {"type": "heartbeat"})
        marks = [m for m in e.markouts if m["horizon_seconds"] == 5]
        self.assertEqual(len(marks), 1)
        self.assertAlmostEqual(marks[0]["observed_seconds"], 6.)
        self.assertLess(marks[0]["liquidation_pnl_per_contract"], 0)
        self.assertEqual(e.summary()["entry_orders_filled"], 1)

    def test_partial_entry_is_one_entry_order(self):
        e = self.engine()
        e.observe(1., trade("a", "no", .45, 5.4))
        e.observe(1.2, trade("b", "no", .44, .6))
        self.assertEqual(e.summary()["maker_fill_events"], 2)
        self.assertEqual(e.summary()["entry_orders_filled"], 1)

    def test_peer_filter_requires_an_observed_peer_book(self):
        e = self.engine(price_filter="peer")
        self.assertEqual(len(e.orders), 0)
        peer = snapshot(.53, .45)
        peer["msg"]["market_ticker"] = "PEER"
        e.observe(.1, peer)
        e.observe(1.1, {"type": "heartbeat"})
        self.assertEqual(len(e.orders), 2)


if __name__ == "__main__":
    unittest.main()
