from datetime import date, datetime
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from data.processing_scripts.build_shared_data import score_causal_updates, map_games_to_markets
from settlement_value_strategy.play_eligibility import state_from_play
from research.validation import constrain_cash, summarize
from research.record import BookSequence
import combined_live

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("kalshi_downloader", ROOT / "data/download_scripts/download_live_kalshi_market_logs.py")
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)
statcast_spec = importlib.util.spec_from_file_location("statcast_downloader", ROOT / "data/download_scripts/download_mlb_statcast.py")
statcast_downloader = importlib.util.module_from_spec(statcast_spec)
statcast_stub = types.ModuleType("pybaseball")
statcast_stub.statcast = None
with patch.dict("sys.modules", {"pybaseball": statcast_stub}):
    statcast_spec.loader.exec_module(statcast_downloader)


class ResearchIntegrityTests(unittest.TestCase):
    def test_incremental_statcast_download_preserves_previous_games(self):
        old = pd.DataFrame({"game_pk": [1], "at_bat_number": [1], "pitch_number": [1], "game_date": ["2026-08-01"]})
        new = pd.DataFrame({"game_pk": [2], "at_bat_number": [1], "pitch_number": [1], "game_date": ["2026-09-01"]})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "2026.parquet"
            old.to_parquet(path)
            with patch.object(statcast_downloader, "OUTPUT_DIR", Path(folder)), \
                 patch.object(statcast_downloader, "statcast", return_value=new):
                statcast_downloader.download_season(2026, datetime(2026, 9, 1), datetime(2026, 9, 1))
            self.assertEqual(set(pd.read_parquet(path).game_pk), {1, 2})

    def test_failed_statcast_window_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "2026.parquet"
            path.write_bytes(b"existing corpus")
            with patch.object(statcast_downloader, "OUTPUT_DIR", Path(folder)), \
                 patch.object(statcast_downloader, "statcast", side_effect=RuntimeError("upstream unavailable")), \
                 self.assertRaisesRegex(RuntimeError, "Incomplete Statcast download"):
                statcast_downloader.download_season(2026, datetime(2026, 9, 1), datetime(2026, 9, 1))
            self.assertEqual(path.read_bytes(), b"existing corpus")

    def test_orderbook_gap_requires_new_snapshot(self):
        stream = BookSequence()
        stream.observe({"type": "orderbook_snapshot", "sid": 1, "seq": 2, "msg": {"market_ticker": "T"}})
        stream.observe({"type": "orderbook_delta", "sid": 1, "seq": 3, "msg": {"market_ticker": "T"}})
        with self.assertRaisesRegex(ValueError, "sequence gap"):
            stream.observe({"type": "orderbook_delta", "sid": 1, "seq": 5, "msg": {"market_ticker": "T"}})
        self.assertEqual(stream.books, set())
        with self.assertRaisesRegex(ValueError, "without a snapshot"):
            BookSequence().observe({"type": "orderbook_delta", "sid": 1, "seq": 2, "msg": {"market_ticker": "T"}})

    def test_combined_launcher_starts_nothing_when_policies_are_disabled(self):
        with patch.object(combined_live, "settlement_enabled", return_value=False), \
             patch.object(combined_live, "hit_reversion_enabled", return_value=False), \
             patch.object(combined_live.subprocess, "Popen") as start:
            with self.assertRaisesRegex(RuntimeError, "Both live policies are disabled"):
                combined_live.run(None)
            start.assert_not_called()

    def test_cash_is_not_borrowed_or_released_at_settlement_without_timestamp(self):
        records = pd.DataFrame([
            {"game_pk": 1, "side": "yes", "entry_time": "2026-09-01T12:00:00Z", "exit_time": None,
             "contracts": 10, "entry_price": .5, "pnl": 5., "fees": 0.},
            {"game_pk": 2, "side": "yes", "entry_time": "2026-09-01T13:00:00Z", "exit_time": None,
             "contracts": 10, "entry_price": .5, "pnl": 5., "fees": 0.},
        ])
        selected, report = constrain_cash(records, 5., lambda n, p: 0.)
        self.assertEqual(len(selected), 1)
        self.assertEqual(report["cash_rejected_orders"], 1)
        self.assertEqual(report["ending_equity"], 10.)

    def test_same_time_sale_proceeds_cannot_fund_a_new_entry(self):
        records = pd.DataFrame([
            {"game_pk": 1, "side": "yes", "entry_time": "2026-09-01T12:00:00Z", "exit_time": "2026-09-01T13:00:00Z",
             "contracts": 10, "entry_price": .5, "pnl": 1., "fees": 0.},
            {"game_pk": 2, "side": "yes", "entry_time": "2026-09-01T13:00:00Z", "exit_time": None,
             "contracts": 10, "entry_price": .5, "pnl": -5., "fees": 0.},
            {"game_pk": 3, "side": "yes", "entry_time": "2026-09-01T13:00:01Z", "exit_time": None,
             "contracts": 10, "entry_price": .5, "pnl": -5., "fees": 0.},
        ])
        selected, report = constrain_cash(records, 5., lambda n, p: 0.)
        self.assertEqual(list(selected.game_pk), [1, 3])
        self.assertEqual(report["ending_equity"], 1.)

    def test_zero_trade_report_has_zero_interval_and_no_positive_evidence(self):
        empty = pd.DataFrame(columns=["entry_time", "exit_time", "game_pk", "side", "pnl", "fees", "contracts", "entry_price"])
        selected, _ = constrain_cash(empty, 100., lambda n, p: 0.)
        report = summarize(selected, {1: date(2026, 9, 1)}, date(2026, 9, 1), date(2026, 9, 3))
        self.assertEqual(report["calendar_days"], 3)
        self.assertEqual(report["day_block_bootstrap_pnl_95pct"], [0., 0.])
        self.assertFalse(report["positive_lower_bound"])

    def test_missing_endpoint_cannot_be_cached_as_complete_history(self):
        with patch.object(downloader, "api_get", return_value=None):
            with self.assertRaises(RuntimeError):
                downloader.fetch_all_pages("/historical/trades", "trades", {})

    def test_open_market_cache_is_refetched_after_settlement(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder)
            path = downloader.trade_cache_path(cache, "TEST", False)
            path.write_text(json.dumps({"market_status": "active", "complete": True, "trades": [{"trade_id": "old"}]}))
            with patch.object(downloader, "fetch_all_pages", return_value=[{"trade_id": "new"}]) as fetch:
                trades = downloader.fetch_market_trades(
                    {"ticker": "TEST", "status": "settled"}, date(2026, 9, 1), None, cache,
                    include_block_trades=False, refresh=False, verbose=False,
                )
            self.assertTrue(fetch.called)
            self.assertEqual(trades[0]["trade_id"], "new")
            self.assertTrue(json.loads(path.read_text())["complete"])

    def test_last_runner_movement_replaces_earlier_base(self):
        play = {
            "about": {"inning": 4, "isTopInning": False},
            "count": {"outs": 1}, "result": {"homeScore": 3, "awayScore": 2},
            "runners": [
                {"details": {"runner": {"id": 1}, "playIndex": 1}, "movement": {"end": "1B"}},
                {"details": {"runner": {"id": 1}, "playIndex": 2}, "movement": {"end": "3B"}},
            ],
        }
        state = state_from_play(play)
        self.assertEqual(state["runner_on_first"], 0)
        self.assertEqual(state["runner_on_third"], 1)

    def test_future_pitch_cannot_change_post_event_inputs(self):
        class Model:
            def predict_proba(self, frame, **kwargs):
                p = .4 + .1 * frame.runner_on_first.to_numpy()
                return np.column_stack([1 - p, p])
        work = pd.DataFrame({
            "game_pk": [1, 1], "game_date": [date(2026, 9, 1)] * 2,
            "market_ticker": ["T"] * 2, "home_win": [1, 1],
            "at_bat_number": [1, 2], "pitch_number": [1, 1],
            "pregame_prob": [.5, .5], "inning": [1, 1], "inning_topbot": [1, 1],
            "outs_when_up": [0, 0], "score_diff": [0, 0], "balls": [0, 0], "strikes": [0, 0],
            "runner_on_first": [0, 0], "runner_on_second": [0, 0], "runner_on_third": [0, 0],
        })
        times = work[["game_pk", "at_bat_number", "pitch_number"]].copy()
        times["pitch_start_time"] = pd.to_datetime(["2026-09-01T12:00:00Z", "2026-09-01T12:01:00Z"])
        times["pitch_end_time"] = times.pitch_start_time + pd.Timedelta(seconds=2)
        times["completed_event"], times["completed_event_batting_home"] = "single", True
        times["atomic_play_input"], times["observed_runner_on_first"] = True, 1
        before = score_causal_updates(work, times, Model())
        work.loc[1, ["runner_on_first", "score_diff", "inning_topbot"]] = [1, 10, 0]
        after = score_causal_updates(work, times, Model())
        self.assertEqual(before.iloc[0].fair_after, .5)
        self.assertEqual(before.iloc[0].fair_after, after.iloc[0].fair_after)
        self.assertEqual(after.iloc[0].runner_on_first_after, 1)

    def test_incomplete_doubleheader_is_never_ordinally_shifted(self):
        states = pd.DataFrame({"game_pk": [1, 2], "game_date": [date(2026, 9, 1)] * 2, "home_team": ["BOS"] * 2})
        games = pd.DataFrame({"game_pk": [1, 2], "first_pitch_time": pd.to_datetime(["2026-09-01T17:00:00Z", "2026-09-01T23:00:00Z"])})
        trades = pd.DataFrame({"game_date": [date(2026, 9, 1)], "market_ticker": ["KXMLBGAME-26SEP011900NYYBOS-BOS"]})
        with self.assertRaisesRegex(RuntimeError, "No MLB games mapped"):
            map_games_to_markets(states, games, trades)
