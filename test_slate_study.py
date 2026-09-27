import contextlib
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
import zlib

from research.maker_portfolio import CashAccount, PortfolioClock, PortfolioGame
from research.maker_study import StudyConfig
from research.slate_study import evaluate, snapshot as freeze
from test_maker_study import baseball
from test_passive_research import snapshot, trade


class PortfolioTests(unittest.TestCase):
    def setUp(self):
        self.account = CashAccount.funded(1.1)
        self.games = [PortfolioGame(f"M{i}", f"P{i}", i, StudyConfig(), self.account) for i in (1, 2)]
        self.clock = PortfolioClock(self.games, self.account)

    def start(self, engine):
        feed = baseball()
        feed["game_pk"] = engine.game_pk
        engine.baseball(0., feed, 0.)
        book = snapshot()
        book["msg"]["market_ticker"] = engine.ticker
        engine.observe(0., book)

    def test_second_game_cannot_spend_reserved_cash(self):
        for engine in self.games:
            self.start(engine)
        self.assertEqual(len(self.games[0].orders), 2)
        self.assertEqual(len(self.games[1].orders), 0)
        self.assertGreater(self.games[1].counters["cash_rejections"], 0)
        self.assertAlmostEqual(self.clock.summary()["equity"], 1.1)

    def test_later_cash_release_cannot_finance_earlier_decision(self):
        for engine in self.games:
            self.start(engine)
        first, second = self.games
        # Release the first game's reserves only at t=2.5. The second game's
        # t=1 and t=2 decisions must remain cash constrained even when the
        # next observation is not received until t=4.
        first.game_live = False
        for order in first.orders.values():
            order["cancelling"] = True
        for order_id in first.orders:
            first.schedule(2.5, "cancel", order_id)
        self.clock.advance_before(4.)
        self.assertEqual({o["submitted_at"] for o in second.orders.values()}, {3.})
        self.assertAlmostEqual(self.clock.summary()["equity"], 1.1)

    def test_shared_pnl_reconciles_after_fill_and_forced_exit(self):
        engine = self.games[0]
        self.start(engine)
        self.clock.advance_before(1.)
        event = trade("fill", "no", .45, 6)
        event["msg"]["market_ticker"] = engine.ticker
        engine.observe(1., event)
        self.clock.advance_before(35.)
        result = self.clock.summary()
        self.assertEqual(engine.position, 0.)
        self.assertLess(result["liquidation_marked_pnl"], 0.)
        self.assertAlmostEqual(result["liquidation_marked_pnl"], engine.realized)
        self.assertIsNone(engine.summary()["cash"])


class SnapshotTests(unittest.TestCase):
    def test_active_gzip_preserves_complete_lines_without_faking_end(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            compressor = zlib.compressobj(wbits=16 + zlib.MAX_WBITS)
            raw = b'{"type":"capture_start"}\n{"type":"unfinished"'
            source = root / "active.gz"
            source.write_bytes(compressor.compress(raw) + compressor.flush(zlib.Z_SYNC_FLUSH))
            target, metadata = freeze(source, root / "frozen")
            self.assertEqual(gzip.decompress(target.read_bytes()), b'{"type":"capture_start"}\n')
            self.assertEqual(metadata["complete_records"], 1)
            self.assertFalse(metadata["capture_complete"])
            self.assertGreater(metadata["partial_trailing_json_bytes_omitted"], 0)

    def test_completed_gzip_without_capture_end_is_still_incomplete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "capture.gz"
            source.write_bytes(gzip.compress(b'{"type":"capture_start"}\n'))
            _, result = freeze(source, root / "frozen")
            self.assertFalse(result["capture_complete"])
            with self.assertRaisesRegex(ValueError, "empty output"):
                freeze(source, root / "frozen")

    def test_completed_capture_end_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "capture.gz"
            source.write_bytes(gzip.compress(b'{"type":"capture_start"}\n{"type":"capture_end"}\n'))
            _, result = freeze(source, root / "frozen")
            self.assertTrue(result["capture_complete"])


class EvaluationTests(unittest.TestCase):
    def evaluate_rows(self, root, events):
        capture, slate = root / "capture.gz", root / "slate.json"
        with gzip.open(capture, "wt") as stream:
            for seconds, row in events:
                row = {"recorded_monotonic_ns": int(seconds * 1e9),
                       "recorded_at": "1970-01-01T00:00:00+00:00" if seconds == 0 else None, **row}
                stream.write(json.dumps(row) + "\n")
        slate.write_text(json.dumps({"games": [{"game_pk": 1, "market_ticker": "M", "away_market_ticker": "P"}]}))
        with contextlib.redirect_stdout(io.StringIO()):
            result = evaluate(capture, slate, root / "results", shared_cash=1.1, policy="join_none_live")
        return result["join_none_live"]

    def events(self):
        book = snapshot()
        book.update(sid=1, seq=1)
        event = trade("a", "no", .45, 6)
        event["msg"]["ts_ms"] = 1000
        return [(0., {"type": "capture_start"}),
                (.01, {"type": "mlb_observation", **baseball()}),
                (.02, {"type": "kalshi_message", "message": book}),
                (1., {"type": "kalshi_message", "message": event})]

    def test_unfinished_prefix_reports_inventory_and_completion_separately(self):
        with tempfile.TemporaryDirectory() as temp:
            result = self.evaluate_rows(Path(temp), self.events())
            self.assertTrue(result["all_replays_valid"])
            self.assertFalse(result["capture_complete"])
            self.assertEqual(result["entry_orders_filled"], 1)
            self.assertEqual(result["games_with_open_inventory"], 1)
            self.assertEqual(result["games_observed_final"], 0)
            self.assertLess(result["portfolio"]["liquidation_marked_pnl"], 0)

    def test_gap_preserves_exposure_but_invalidates_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            result = self.evaluate_rows(Path(temp), self.events() + [(2., {"type": "connection_gap"})])
            self.assertFalse(result["all_replays_valid"])
            self.assertEqual(result["games_with_open_inventory"], 1)
            self.assertEqual(result["entry_orders_filled"], 1)
            self.assertIn("indeterminate", result["continuity_error"])

    def test_capture_end_is_not_game_settlement(self):
        with tempfile.TemporaryDirectory() as temp:
            result = self.evaluate_rows(Path(temp), self.events() + [(2., {"type": "capture_end"})])
            self.assertTrue(result["capture_complete"])
            self.assertEqual(result["games_observed_final"], 0)
            self.assertEqual(result["games_with_open_inventory"], 1)

    def test_disconnect_cannot_extend_time_to_force_an_inventory_exit(self):
        with tempfile.TemporaryDirectory() as temp:
            result = self.evaluate_rows(Path(temp), self.events() + [
                (2., {"type": "connection_end"}), (40., {"type": "capture_end"})])
            self.assertTrue(result["capture_complete"])
            self.assertEqual(result["games_with_open_inventory"], 1)
            self.assertEqual(result["taker_fill_events"], 0)


if __name__ == "__main__":
    unittest.main()
