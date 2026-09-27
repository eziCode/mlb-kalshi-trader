import contextlib
import copy
from dataclasses import replace
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest

from research.fill_quality import analyze_games, clustered_interval
from research.maker_portfolio import CashAccount, PortfolioClock
from research.selective_maker import AnchoredForecast, SelectiveConfig, SelectiveGame
from research.selective_study import evaluate, freeze, reconcile_ledger, settlement_mark, validate_capture_role
from research.sportsbook import OddsTape
from test_forward_value import ConstantModel, observation
from test_passive_research import snapshot, delta, trade
from test_sportsbook import BASE, SLATE, iso, odds_row


GAME = {"game_pk": 1, "market_ticker": "M", "away_market_ticker": "P",
        "home_code": "WSH", "away_code": "NYM", "scheduled_time": iso(BASE + 300)}


class SelectiveMakerTests(unittest.TestCase):
    def engine(self, **kwargs):
        account = CashAccount.funded(100.)
        engine = SelectiveGame(GAME, replace(SelectiveConfig(), **kwargs), account)
        engine.baseball(12., {"game_pk": 1, "observation": observation()}, 0.)
        peer = snapshot(.47, .49)
        peer["msg"]["market_ticker"] = "P"
        engine.observe(12., peer)
        engine.observe(12., snapshot(.49, .47))  # 4c spread, paired mid agrees.
        return engine, account

    def test_three_cent_spread_clears_two_maker_fees(self):
        engine, _ = self.engine()
        self.assertEqual(len(engine.orders), 2)
        for order in engine.orders.values():
            self.assertGreater(order["features"]["estimated_roundtrip_edge"], .005)
        self.assertEqual(engine.summary()["opening_order_attempts"], 2)

    def test_queue_priority_and_partial_fills_count_one_opening_order(self):
        e, account = self.engine()
        e.observe(13., trade("a", "no", .49, 5.4))
        e.observe(13.1, trade("b", "no", .48, .6))
        self.assertAlmostEqual(e.position, 1.)
        self.assertEqual(e.summary()["entry_orders_filled"], 1)
        self.assertEqual(e.summary()["order_attempts"][0]["filled_quantity"], 1.)
        PortfolioClock([e], account).summary()
        reconcile_ledger({**e.summary(), "fills": e.fills})

    def test_stale_quotes_cancel_but_race_fill_survives(self):
        e, _ = self.engine(maximum_book_age=1.)
        e.advance_clock(14.1)
        self.assertTrue(all(o["cancelling"] for o in e.orders.values()))
        e.observe(14.2, trade("adverse", "no", .48, 1.))
        self.assertTrue(e.fills[0]["cancel_pending"])
        self.assertEqual(e.fills[0]["price"], .49)

    def test_fast_move_blocks_replacement_but_does_not_erase_old_order(self):
        e, _ = self.engine()
        e.observe(12.8, delta("yes", .49, -5))
        e.observe(12.9, delta("yes", .30, 5))
        e.advance_clock(13.1)
        self.assertTrue(all(o["cancelling"] for o in e.orders.values()))
        e.observe(13.2, trade("race", "no", .29, 1.))
        self.assertEqual(e.fills[0]["price"], .49)
        self.assertTrue(e.fills[0]["traded_through"])

    def test_missing_sportsbook_is_not_replaced_by_market_prices(self):
        e, _ = self.engine(reference_mode="sportsbook_anchor")
        self.assertFalse(e.orders)
        self.assertGreater(e.rejections["reference_unavailable"], 0)

    def test_queue_cap_blocks_openings(self):
        e, _ = self.engine(maximum_queue=4.)
        self.assertFalse(e.orders)
        self.assertGreater(e.rejections["queue_too_deep"], 0)

    def test_reducing_quotes_survive_opening_filter_failure(self):
        e, _ = self.engine()
        e.observe(13., trade("a", "no", .49, 6))
        e.observe(13.1, delta("yes", .49, 100))
        e.advance_clock(14.1)
        self.assertTrue(any(o["side"] == "no" and not o["cancelling"] for o in e.orders.values()))

    def test_reentry_requires_flat_inventory_and_cooldown(self):
        e, _ = self.engine()
        e.observe(13., trade("buy", "no", .49, 6))
        e.observe(13.2, trade("close", "yes", .53, 6))
        self.assertEqual(e.position, 0.)
        e.advance_clock(17.9)
        self.assertFalse(e.orders)
        e.advance_clock(19.1)
        self.assertEqual(len(e.orders), 2)
        self.assertEqual(e.summary()["entry_orders_filled"], 1)

    def test_entry_cap_applies_to_filled_orders(self):
        e, _ = self.engine(maximum_entry_orders=1)
        e.observe(13., trade("buy", "no", .49, 6))
        e.observe(13.2, trade("close", "yes", .53, 6))
        e.advance_clock(20.)
        self.assertFalse(e.orders)

    def test_depth_mark_reports_unexecutable_remainder(self):
        e, _ = self.engine(maximum_inventory_seconds=100.)
        e.observe(13., trade("buy", "no", .49, 6))
        e.observe(14., delta("yes", .49, -4.8))
        e.advance_clock(18.1)
        mark = e.markouts[0]
        self.assertTrue(mark["depth_checked"])
        self.assertAlmostEqual(mark["unfilled_mark_quantity"], .8)
        self.assertLess(mark["liquidation_pnl_per_contract"], -.39)

    def test_missing_horizon_is_not_zero_return(self):
        e, _ = self.engine()
        e.observe(13., trade("buy", "no", .49, 6))
        result = analyze_games([{**e.summary(), "fills": e.fills, "fill_markouts": e.markouts, "replay_valid": True}])
        horizon = result["all_openings"]["horizons"]["30"]
        self.assertEqual(horizon["missing_or_invalid_observations"], 1)
        self.assertIsNone(horizon["mid_change_per_contract"]["weighted_mean"])
        self.assertIsNone(clustered_interval({1: [100, 10000]}))

    def test_independent_ledger_reconciles_partial_exits(self):
        e, _ = self.engine(maximum_inventory_seconds=2.)
        e.observe(13., trade("buy", "no", .49, 6))
        e.observe(14., delta("yes", .49, -4.6))
        e.advance_clock(18.)
        game = {**e.summary(), "fills": e.fills}
        reconcile_ledger(game)
        self.assertGreater(e.position, 0)
        corrupted = copy.deepcopy(game)
        corrupted["fills"][0]["opening_quantity"] = 2
        with self.assertRaises(ValueError):
            reconcile_ledger(corrupted)

    def test_no_position_payout_uses_complement_of_same_market(self):
        e, _ = self.engine(maximum_inventory_seconds=100.)
        e.observe(13., trade("no_buy", "yes", .53, 6))
        self.assertEqual(e.position, -1.)
        game = {**e.summary(), "fills": e.fills}
        terminal = {"M": {"value": 0., "exchange_time": 100., "observed_monotonic_ns": 101_000_000_000,
                           "observed_at": iso(101.)}}
        result = settlement_mark(game, terminal, 0., 0)
        self.assertTrue(result["position_settled"])
        self.assertEqual(result["inventory_value"], 1.)
        self.assertAlmostEqual(result["net_shadow_pnl"], 1 - .47 - e.fills[0]["fee"])

    def test_later_book_does_not_change_earlier_orders(self):
        e, _ = self.engine()
        before = copy.deepcopy(e.summary()["order_attempts"])
        e.observe(12.5, delta("yes", .49, 20))
        after = e.summary()["order_attempts"]
        self.assertEqual([a["features"] for a in before], [a["features"] for a in after])


class AnchorTests(unittest.TestCase):
    def event(self):
        event = observation()
        event["currentPlay"]["playEvents"][0]["startTime"] = iso(BASE + 10)
        return event

    def test_anchor_requires_consensus_received_before_actual_first_pitch(self):
        early = odds_row(home_price=1.5, away_price=2.8)
        late = odds_row(BASE + 11)
        forecaster = AnchoredForecast(1, ConstantModel(), .5, OddsTape([early, late], SLATE))
        forecaster.observe(12., self.event(), BASE)
        self.assertAlmostEqual(forecaster.probability(12), (1 / 1.5) / (1 / 1.5 + 1 / 2.8))
        missing = AnchoredForecast(1, ConstantModel(), .5, OddsTape([late], SLATE))
        missing.observe(12., self.event(), BASE)
        self.assertIsNone(missing.probability(12))

    def test_late_available_odds_do_not_retroactively_create_anchor(self):
        row = odds_row()
        row["recorded_at"] = iso(BASE + 11)
        forecaster = AnchoredForecast(1, ConstantModel(), .5, OddsTape([row], SLATE))
        forecaster.observe(12., self.event(), BASE)
        self.assertIsNone(forecaster.anchor)


class SelectiveRunnerTests(unittest.TestCase):
    def fixture(self, root):
        capture, slate = root / "capture.gz", root / "slate.json"
        obs = observation()
        peer = snapshot(.47, .49)
        peer["msg"]["market_ticker"] = "P"
        fill = trade("x", "no", .49, 6)
        fill["msg"]["ts_ms"] = 13000
        rows = [(0., {"type": "capture_start"}), (12., {"type": "mlb_observation", "game_pk": 1, "observation": obs}),
                (12., {"type": "kalshi_message", "message": peer}),
                (12., {"type": "kalshi_message", "message": snapshot(.49, .47)}),
                (13., {"type": "kalshi_message", "message": fill}),
                (14., {"type": "connection_end"}), (100., {"type": "capture_end"})]
        with gzip.open(capture, "wt") as stream:
            for when, row in rows:
                stream.write(json.dumps({**row, "recorded_at": iso(when), "recorded_monotonic_ns": int(when * 1e9)}) + "\n")
        slate.write_text(json.dumps({"games": [GAME], "scheduled_game_count": 2, "unmapped_game_pks": [2]}))
        return capture, slate

    def test_replay_retains_open_inventory_and_zero_trade_candidates(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            capture, slate = self.fixture(root)
            protocol = freeze(root / "frozen")
            result = evaluate(capture, slate, protocol, root / "out", latencies=(.68,))
            live = result["candidates"]["selective_live_latency_0.68"]
            self.assertEqual(live["entry_orders_filled"], 1)
            self.assertEqual(live["scheduled_game_coverage"], .5)
            self.assertEqual(live["unsettled_quantity"], 1.)
            self.assertFalse(live["complete_policy_evaluation"])
            self.assertEqual(result["observed_seconds"], 14.)
            self.assertEqual(result["candidates"]["sportsbook_anchor_latency_0.68"]["entry_orders_filled"], 0)
            self.assertTrue((root / "out/fill_quality/per_game.csv").exists())

    def test_protocol_tampering_fails_before_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            capture, slate = self.fixture(root)
            protocol = freeze(root / "frozen")
            spec = json.loads(protocol.read_text())
            spec["policies"]["selective_live"]["maximum_queue"] = 999
            protocol.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                evaluate(capture, slate, protocol, root / "out", latencies=(.68,))

    def test_same_day_development_cannot_be_labeled_validation(self):
        spec = {"frozen_at": iso(BASE)}
        with self.assertRaisesRegex(ValueError, "later date"):
            validate_capture_role(spec, {"recorded_at": iso(BASE + 1)}, {"games": [GAME]}, "validation")
        with self.assertRaisesRegex(ValueError, "predates"):
            validate_capture_role(spec, {"recorded_at": iso(BASE - 1)}, {"games": [GAME]}, "validation")
        tomorrow = {"games": [{**GAME, "scheduled_time": iso(BASE + 86400)}]}
        validate_capture_role(spec, {"recorded_at": iso(BASE + 86300)}, tomorrow, "validation")


if __name__ == "__main__":
    unittest.main()
