from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch
from pathlib import Path
import tempfile
import json

import pandas as pd

from settlement_value_strategy.prepare_data import compact_execution_tape
from settlement_value_strategy.predict import MispricingPredictor
from settlement_value_strategy.build_normalized_raw import pitch_times, state_model_frame
from settlement_value_strategy.live_paper_trader import (
    SharedPaperPortfolio, PaperPosition, build_live_decision_row,
    consecutive_pitch, should_surface_worker_line, wait_for_pregame_anchor,
)


class PregameAnchorRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_game_without_pitch_time_is_retried(self):
        with (
            patch(
                "settlement_value_strategy.live_paper_trader.fetch_pregame_anchor",
                side_effect=[
                    RuntimeError(
                        "Live game has no authoritative first-pitch time"
                    ),
                    0.55,
                ],
            ) as fetch,
            patch(
                "settlement_value_strategy.live_paper_trader.asyncio.sleep",
                new=AsyncMock(),
            ) as sleep,
        ):
            result = await wait_for_pregame_anchor()
        self.assertEqual(result, 0.55)
        self.assertEqual(fetch.call_count, 2)
        sleep.assert_awaited_once()


class PipelineTests(unittest.TestCase):
    def test_main_log_surfaces_readiness_and_trades(self):
        self.assertTrue(should_surface_worker_line("TRADER READY game_pk=1"))
        self.assertTrue(should_surface_worker_line("TRADE BUY NO"))
        self.assertFalse(should_surface_worker_line("INITIALIZE_LIVE_BASELINE"))

    def test_live_pitch_sequence_rejects_polling_gap(self):
        self.assertTrue(consecutive_pitch((2, 1, "a"), (2, 2, "b")))
        self.assertTrue(consecutive_pitch((2, 3, "a"), (3, 1, "b")))
        self.assertFalse(consecutive_pitch((2, 1, "a"), (2, 3, "b")))

    def test_live_row_uses_strict_pre_signal_flow(self):
        event = pd.Timestamp("2026-07-01T12:00:05Z")
        before = {
            "inning": 1, "inning_topbot": 0, "outs_when_up": 0,
            "score_diff": 0, "balls": 0, "strikes": 0,
            "runner_on_first": 0, "runner_on_second": 0,
            "runner_on_third": 0,
        }
        after = {**before, "strikes": 1}
        trades = pd.DataFrame({
            "trade_id": ["anchor", "prior", "signal"],
            "created_time": [
                event - pd.Timedelta(seconds=3),
                event + pd.Timedelta(seconds=.5),
                event + pd.Timedelta(seconds=1.1),
            ],
            "yes_price_dollars": [.50, .51, .52],
            "count_fp": [5.0, 6.0, 100.0],
            "taker_outcome_side": ["yes", "no", "yes"],
        })
        predictor = MispricingPredictor()
        row = build_live_decision_row(
            game_pk=1, before=before, after=after, fair_before=.50,
            fair_after=.49, pitch_token=(0, 1, event.isoformat()),
            trades=trades, config=predictor.config,
        )
        self.assertIsNotNone(row)
        self.assertEqual(row["market_home_price"], .52)
        self.assertEqual(row["pre_trade_count_2s"], 1.0)
        self.assertEqual(row["pre_volume_2s"], 6.0)
        self.assertEqual(row["delta_strikes"], 1.0)
        predictor.decision(row)

    def test_live_portfolio_prevents_duplicate_game_position(self):
        now = pd.Timestamp("2026-07-01T12:00:00Z").to_pydatetime()
        position = PaperPosition("yes", 10.0, .5, .1, now, .7, "pitch")
        with tempfile.TemporaryDirectory() as directory:
            portfolio = SharedPaperPortfolio(
                Path(directory) / "paper.sqlite3", starting_cash=100.0
            )
            self.assertTrue(portfolio.open_position(1, "ticker", position))
            self.assertFalse(portfolio.open_position(1, "ticker", position))
            metrics = portfolio.metrics()
            self.assertEqual(metrics.open_positions, 1)
            self.assertAlmostEqual(metrics.cash, 94.9)

    def test_predictor_scores_packaged_row(self):
        row = pd.read_parquet(
            MispricingPredictor().root.parent / "data/settlement_value/decision_rows.parquet"
        ).iloc[0].to_dict()
        result = MispricingPredictor().decision(row)
        self.assertTrue(0 < result["settlement_probability"] < 1)
        self.assertIn(result["side"], {"yes", "no"})

    def test_compaction_excludes_same_timestamp_trade(self):
        now = pd.Timestamp("2026-01-01T00:00:00Z")
        decisions = pd.DataFrame({
            "game_pk": [1], "signal_time": [now],
            "next_update_time": [now + pd.Timedelta(seconds=3)],
        })
        trades = pd.DataFrame({
            "game_pk": [1, 1], "trade_id": [1, 2],
            "created_time": [now, now + pd.Timedelta(seconds=1)],
        })
        compact = compact_execution_tape(decisions, trades)
        self.assertEqual(compact.trade_id.tolist(), [2])

    def test_mlb_feed_produces_authoritative_pitch_times(self):
        payload = {"liveData": {"plays": {"allPlays": [{
            "atBatIndex": 2,
            "playEvents": [{
                "isPitch": True, "pitchNumber": 1,
                "startTime": "2026-07-01T12:00:00Z",
                "endTime": "2026-07-01T12:00:05Z",
            }],
        }]}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "123.json"
            path.write_text(json.dumps(payload))
            result = pitch_times(Path(directory))
        self.assertEqual(result.iloc[0].at_bat_number, 3)
        self.assertEqual(result.iloc[0].pitch_number, 1)

    def test_state_model_uses_batting_perspective(self):
        frame = pd.DataFrame({
            "pregame_prob": [.60, .60], "inning": [1, 1],
            "inning_topbot": [0, 1], "outs_when_up": [0, 0],
            "score_diff": [2, 2], "balls": [0, 0], "strikes": [0, 0],
            "runner_on_first": [0, 0], "runner_on_second": [0, 0],
            "runner_on_third": [0, 0],
        })
        features = state_model_frame(frame)
        self.assertEqual(features.pregame_batting_prob.tolist(), [.40, .60])
        self.assertEqual(features.batting_score_diff.tolist(), [-2, 2])


if __name__ == "__main__":
    unittest.main()
