import unittest
from dataclasses import replace
import gzip
import json
from pathlib import Path
import tempfile

from research.passive import MakerConfig, PassiveReplay, replay, fee


def snapshot(yes=.45, no=.53, size=5):
    return {"type": "orderbook_snapshot", "msg": {"market_ticker": "M",
        "yes_dollars_fp": [[str(yes), str(size)]], "no_dollars_fp": [[str(no), str(size)]]}}


def trade(identity, taker, yes, size):
    return {"type": "trade", "msg": {"market_ticker": "M", "trade_id": identity,
        "taker_outcome_side": taker, "yes_price_dollars": str(yes),
        "no_price_dollars": str(round(1 - yes, 4)), "count_fp": str(size)}}


def delta(side, price, size):
    return {"type": "orderbook_delta", "msg": {"market_ticker": "M", "side": side,
        "price_dollars": str(price), "delta_fp": str(size)}}


class PassiveResearchTests(unittest.TestCase):
    def engine(self, **changes):
        # Most execution invariants isolate queue behavior from fees.
        changes = {"maker_fee_rate": 0., **changes}
        engine = PassiveReplay("M", replace(MakerConfig(), **changes))
        engine.observe(0., snapshot())
        return engine

    def test_touch_does_not_fill_and_queue_requires_volume(self):
        e = self.engine()
        e.observe(1., trade("a", "no", .45, 4))
        self.assertEqual(len(e.fills), 0)
        e.observe(2., trade("b", "no", .45, 2))
        self.assertEqual(len(e.fills), 1)
        self.assertEqual(e.position, 1)
        self.assertEqual(e.fills[0]["price"], .45)

    def test_book_cancellations_do_not_improve_queue(self):
        e = self.engine()
        e.observe(1., delta("yes", .45, -4))
        e.observe(1.1, trade("a", "no", .45, 2))
        self.assertEqual(len(e.fills), 0)

    def test_future_book_change_cannot_reprice_submitted_order(self):
        e = self.engine()
        e.observe(.2, delta("yes", .44, 100))
        e.observe(1., trade("a", "no", .44, 1))
        self.assertEqual(e.fills[0]["price"], .45)
        self.assertTrue(e.fills[0]["traded_through"])

    def test_opposite_taker_direction_cannot_fill_order(self):
        e = self.engine()
        e.observe(1., trade("a", "yes", .45, 100))
        self.assertEqual(len(e.fills), 0)  # no contract trades at .55, above our .53

    def test_duplicate_trade_not_counted_twice(self):
        e = self.engine()
        e.observe(1., trade("a", "no", .45, 4))
        e.observe(1.1, trade("a", "no", .45, 4))
        self.assertEqual(len(e.fills), 0)

    def test_post_only_rejected_if_book_crossed_during_submission(self):
        e = self.engine()
        e.observe(.3, delta("no", .56, 5))
        e.observe(.8, {"type": "heartbeat"})
        self.assertEqual(e.counters["post_only_rejections"], 1)
        self.assertFalse(any(o["side"] == "yes" for o in e.orders.values()))

    def test_quote_remains_fillable_during_cancellation(self):
        e = self.engine()
        e.observe(1., delta("yes", .46, 5))  # spread contracts; request cancellation
        self.assertTrue(all(o["cancelling"] for o in e.orders.values()))
        e.observe(1.3, trade("a", "no", .44, 1))
        self.assertEqual(len(e.fills), 1)
        self.assertTrue(e.fills[0]["cancel_pending"])

    def test_confirmed_cancellation_removes_exposure(self):
        e = self.engine()
        e.observe(1., delta("yes", .46, 5))
        e.observe(1.8, trade("a", "no", .44, 1))
        self.assertEqual(len(e.fills), 0)
        self.assertEqual(e.reserved, 0)

    def test_paired_fills_reconcile_cash_and_realized_profit(self):
        e = self.engine()
        e.observe(1., trade("a", "no", .45, 6))
        e.observe(1.2, trade("b", "yes", .47, 6))
        result = e.summary()
        self.assertEqual(e.position, 0)
        self.assertAlmostEqual(result["realized_pnl"], .02)
        self.assertAlmostEqual(result["liquidation_marked_pnl"], .02)
        self.assertAlmostEqual(e.cash + e.reserved, 100.02)

    def test_mlb_default_charges_maker_fees_on_both_legs(self):
        self.assertEqual(MakerConfig().maker_fee_rate, .0175)
        e = self.engine(maker_fee_rate=MakerConfig().maker_fee_rate)
        e.observe(1., trade("a", "no", .45, 6))
        e.observe(1.2, trade("b", "yes", .47, 6))
        result = e.summary()
        self.assertAlmostEqual(result["fees"], .0088)
        self.assertAlmostEqual(result["realized_pnl"], .0112)
        self.assertAlmostEqual(e.cash + e.reserved, 100.0112)

    def test_fractional_fee_rounds_combined_cost_to_centicent(self):
        self.assertAlmostEqual(.01 * .3333 + fee(.01, .3333, .0175), .0034)

    def test_unpaired_inventory_is_marked_after_taker_cost(self):
        e = self.engine()
        e.observe(1., trade("a", "no", .45, 6))
        result = e.summary()
        self.assertEqual(result["realized_pnl"], 0)
        self.assertEqual(result["inventory"], 1)
        self.assertLess(result["liquidation_marked_pnl"], 0)

    def test_partial_fill_preserves_remaining_order_and_inventory(self):
        e = self.engine()
        e.observe(1., trade("a", "no", .45, 5.4))
        self.assertAlmostEqual(e.position, .4)
        # The remainder is cancelling after the decision, but still exposed.
        e.observe(1.2, trade("b", "no", .44, 1))
        self.assertAlmostEqual(e.position, 1)
        self.assertAlmostEqual(sum(f["quantity"] for f in e.fills), 1)

    def test_insufficient_cash_cannot_post_both_sides(self):
        e = self.engine(starting_cash=.6)
        self.assertEqual(len(e.orders), 1)
        self.assertEqual(e.counters["cash_rejections"], 1)
        self.assertGreaterEqual(e.cash, 0)

    def test_capture_gap_does_not_pretend_orders_were_cancelled(self):
        message = snapshot()
        message.update(sid=1, seq=1)
        records = [dict(type="connection_start", recorded_monotonic_ns=0),
                   dict(type="kalshi_message", recorded_monotonic_ns=100, message=message),
                   dict(type="connection_gap", recorded_monotonic_ns=1_000_000_000)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl.gz"
            with gzip.open(path, "wt") as stream:
                stream.write("\n".join(json.dumps(r) for r in records))
            result, _ = replay(path, "M")
        self.assertFalse(result["full_replay_valid"])
        self.assertEqual(result["outstanding_orders"], 2)
        self.assertGreater(result["pending_reserved_cash"], 0)

    def test_missing_market_is_not_a_valid_zero_fill_test(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl.gz"
            with gzip.open(path, "wt") as stream:
                stream.write(json.dumps(dict(type="connection_start", recorded_monotonic_ns=0)))
            result, _ = replay(path, "M")
        self.assertFalse(result["full_replay_valid"])
        self.assertIn("No snapshot", result["continuity_error"])

    def test_negative_book_depth_is_rejected(self):
        e = self.engine()
        with self.assertRaises(ValueError):
            e.observe(1., delta("yes", .45, -100))

    def test_late_receipt_of_old_trade_cannot_fill_new_order(self):
        e = self.engine()
        message = trade("a", "no", .44, 100)
        message["msg"]["exchange_elapsed_seconds"] = .2
        e.observe(1., message)
        self.assertEqual(len(e.fills), 0)

    def test_quote_expiry_timer_runs_during_silent_stream(self):
        e = self.engine(lifetime_seconds=2.)
        # Old orders expire/cancel at 2.68, replacement orders arrive at 3.68.
        e.observe(3.3, trade("a", "no", .44, 100))
        self.assertEqual(len(e.fills), 0)
        self.assertEqual(e.counters["cancellations"], 2)


if __name__ == "__main__":
    unittest.main()
