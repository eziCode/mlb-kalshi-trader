from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, Mock, patch
from datetime import date
from pathlib import Path
import tempfile
import json

import pandas as pd
import settlement_value_strategy.live_paper_trader as live_paper_trader
from catboost import CatBoostClassifier

from settlement_value_strategy.prepare_data import compact_execution_tape
from settlement_value_strategy.predict import MispricingPredictor
from settlement_value_strategy.build_normalized_raw import pitch_times, state_model_frame
from settlement_value_strategy.live_paper_trader import (
    SharedPaperPortfolio, PaperPosition, build_live_decision_row,
    consecutive_pitch, execution_within_window, reconcile_final_positions,
    pregame_probability_from_rating_state, replay_fill_from_observed_trades,
    should_surface_worker_line, wait_for_pregame_anchor, discover_daily_games,
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
    def test_postponed_doubleheader_matches_original_ticker_date(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"dates": [{"games": [
            {
                "gamePk": 1, "gameDate": "2026-07-22T17:05:00Z",
                "status": {"abstractGameState": "Preview"},
                "teams": {
                    "away": {"team": {"id": 134}},
                    "home": {"team": {"id": 147}},
                },
            },
            {
                "gamePk": 2, "gameDate": "2026-07-22T23:05:00Z",
                "status": {"abstractGameState": "Preview"},
                "teams": {
                    "away": {"team": {"id": 134}},
                    "home": {"team": {"id": 147}},
                },
            },
        ]}]}

        def event(ticker):
            return {
                "event_ticker": ticker,
                "markets": [
                    {"ticker": f"{ticker}-PIT", "status": "active"},
                    {"ticker": f"{ticker}-NYY", "status": "active"},
                ],
            }

        events = [
            event("KXMLBGAME-26JUL221335PITNYY"),
            event("KXMLBGAME-26JUL211905PITNYY"),
        ]
        with (
            patch(
                "settlement_value_strategy.live_paper_trader.requests.get",
                return_value=response,
            ),
            patch(
                "settlement_value_strategy.live_paper_trader._daily_kalshi_events",
                return_value=events,
            ),
        ):
            games, warnings = discover_daily_games(date(2026, 7, 22))
        self.assertEqual([game.game_pk for game in games], [1, 2])
        self.assertEqual(warnings, [])
        self.assertIn("26JUL221335", games[0].market_ticker)
        self.assertIn("26JUL211905", games[1].market_ticker)

    def test_doubleheader_time_matches_single_listed_game(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"dates": [{"games": [
            {
                "gamePk": 1, "gameDate": "2026-07-22T17:35:00Z",
                "status": {"abstractGameState": "Preview"},
                "teams": {
                    "away": {"team": {"id": 110}},
                    "home": {"team": {"id": 111}},
                },
            },
            {
                "gamePk": 2, "gameDate": "2026-07-22T23:10:00Z",
                "status": {"abstractGameState": "Preview"},
                "teams": {
                    "away": {"team": {"id": 110}},
                    "home": {"team": {"id": 111}},
                },
            },
        ]}]}
        ticker = "KXMLBGAME-26JUL221910BALBOS"
        event = {
            "event_ticker": ticker,
            "markets": [
                {"ticker": f"{ticker}-BAL"},
                {"ticker": f"{ticker}-BOS"},
            ],
        }
        with (
            patch(
                "settlement_value_strategy.live_paper_trader.requests.get",
                return_value=response,
            ),
            patch(
                "settlement_value_strategy.live_paper_trader._daily_kalshi_events",
                return_value=[event],
            ),
        ):
            games, warnings = discover_daily_games(date(2026, 7, 22))
        self.assertEqual([game.game_pk for game in games], [2])
        self.assertEqual(games[0].market_ticker, f"{ticker}-BOS")
        self.assertIn("time-matched 1 of 2", warnings[0])

    def test_doubleheader_pairs_before_filtering_final_game(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"dates": [{"games": [
            {
                "gamePk": 1, "gameDate": "2026-07-20T17:00:00Z",
                "status": {"abstractGameState": "Final"},
                "teams": {
                    "away": {"team": {"id": 121}},
                    "home": {"team": {"id": 144}},
                },
            },
            {
                "gamePk": 2, "gameDate": "2026-07-20T23:00:00Z",
                "status": {"abstractGameState": "Preview"},
                "teams": {
                    "away": {"team": {"id": 121}},
                    "home": {"team": {"id": 144}},
                },
            },
        ]}]}

        def event(name):
            return {
                "event_ticker": name,
                "markets": [
                    {"ticker": f"{name}-NYM"},
                    {"ticker": f"{name}-ATL"},
                ],
            }

        with (
            patch(
                "settlement_value_strategy.live_paper_trader.requests.get",
                return_value=response,
            ),
            patch(
                "settlement_value_strategy.live_paper_trader._daily_kalshi_events",
                return_value=[event("KXMLBGAME-G1"), event("KXMLBGAME-G2")],
            ),
        ):
            games, warnings = discover_daily_games(date(2026, 7, 20))
        self.assertEqual(warnings, [])
        self.assertEqual(len(games), 1)
        self.assertEqual(games[0].game_pk, 2)
        self.assertEqual(games[0].market_ticker, "KXMLBGAME-G2-ATL")

    def test_live_execution_rejects_stale_signal(self):
        signal = pd.Timestamp("2026-07-01T12:00:00Z")
        self.assertTrue(execution_within_window(
            signal, (signal + pd.Timedelta(seconds=4.999)).to_pydatetime(), 5,
        ))
        self.assertFalse(execution_within_window(
            signal, (signal + pd.Timedelta(seconds=5)).to_pydatetime(), 5,
        ))
        self.assertFalse(execution_within_window(
            signal, (signal + pd.Timedelta(seconds=5.001)).to_pydatetime(), 5,
        ))
        self.assertFalse(execution_within_window(
            signal, signal.to_pydatetime(), 5,
        ))

    def test_live_pregame_prior_uses_saved_rating_team_codes(self):
        state = {
            "ratings": {"AZ": 1600, "CWS": 1400, "ATH": 1300},
            "initial_rating": 1500, "home_advantage": 0,
        }
        exact = pregame_probability_from_rating_state(state, "AZ", "CWS")
        aliased = pregame_probability_from_rating_state(
            state, "ARI", "CHW"
        )
        self.assertAlmostEqual(exact, aliased)
        self.assertGreater(exact, .5)

    def test_live_fill_uses_backtest_compatible_post_signal_trade(self):
        signal = pd.Timestamp("2026-07-01T12:00:00Z")
        trades = pd.DataFrame({
            "trade_id": [1, 2, 3],
            "created_time": [
                signal, signal + pd.Timedelta(seconds=1),
                signal + pd.Timedelta(seconds=2),
            ],
            "yes_price_dollars": [.40, .41, .42],
            "count_fp": [100, 100, 100],
            "taker_outcome_side": ["yes", "no", "yes"],
        })
        config = MispricingPredictor().config
        fill = replay_fill_from_observed_trades(
            trades, signal, .80, [], "yes", config,
        )
        self.assertIsNotNone(fill)
        self.assertEqual(fill["price"], .41)
        self.assertEqual(
            pd.Timestamp(fill["time"]), signal + pd.Timedelta(seconds=1)
        )

    def test_live_confirmation_uses_actual_order_budget_and_keeps_fee_check(self):
        signal = pd.Timestamp("2026-07-01T12:00:00Z")
        trades = pd.DataFrame({
            "trade_id": [1],
            "created_time": [signal + pd.Timedelta(seconds=1)],
            "yes_price_dollars": [.40],
            "count_fp": [5.0],
            "taker_outcome_side": ["no"],
        })
        config = MispricingPredictor().config
        fill = replay_fill_from_observed_trades(
            trades, signal, .45, [], "yes", config,
            confirmation_budget=2.0,
        )
        self.assertIsNotNone(fill)
        self.assertAlmostEqual(fill["contracts"], 5.0)
        self.assertGreaterEqual(fill["edge"], .02)
        self.assertGreaterEqual(fill["expected_pnl"], 0.0)
        self.assertIsNone(replay_fill_from_observed_trades(
            trades, signal, .41, [], "yes", config,
            confirmation_budget=2.0,
        ))

    def test_startup_reconciles_positions_from_final_games(self):
        now = pd.Timestamp("2026-07-01T12:00:00Z").to_pydatetime()
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "gameData": {"status": {"abstractGameState": "Final"}},
            "liveData": {"linescore": {"teams": {
                "away": {"runs": 5}, "home": {"runs": 2},
            }}},
        }
        with tempfile.TemporaryDirectory() as directory:
            portfolio = SharedPaperPortfolio(
                Path(directory) / "paper.sqlite3", starting_cash=100,
            )
            position = PaperPosition(
                "away_yes", 20, .5, .1, now, .7, "pitch",
            )
            self.assertTrue(portfolio.open_position(123, "AWAY", position))
            with patch(
                "settlement_value_strategy.live_paper_trader.requests.get",
                return_value=response,
            ):
                self.assertEqual(reconcile_final_positions(portfolio), 1)
            metrics = portfolio.metrics()
            self.assertEqual(metrics.open_positions, 0)
            self.assertAlmostEqual(metrics.cash, 109.9)

    def test_worker_supervisor_has_sleep_dependency(self):
        self.assertIsNotNone(live_paper_trader.time.sleep)

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

    def test_live_portfolio_stacks_after_cooldown_and_rejects_duplicates(self):
        now = pd.Timestamp("2026-07-01T12:00:00Z").to_pydatetime()
        position = PaperPosition("yes", 10.0, .5, .1, now, .7, "pitch")
        with tempfile.TemporaryDirectory() as directory:
            portfolio = SharedPaperPortfolio(
                Path(directory) / "paper.sqlite3", starting_cash=100.0
            )
            self.assertTrue(portfolio.open_position(1, "ticker", position, 30))
            self.assertFalse(portfolio.open_position(1, "ticker", position, 30))
            later = PaperPosition(
                "yes", 10.0, .5, .1,
                now + pd.Timedelta(seconds=31), .7, "next-pitch",
            )
            self.assertTrue(portfolio.open_position(1, "ticker", later, 30))
            metrics = portfolio.metrics()
            self.assertEqual(metrics.open_positions, 2)
            self.assertAlmostEqual(metrics.cash, 89.8)

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
        self.assertEqual(features.batting_score_diff.tolist(), [-2, 2])
        self.assertNotIn("pregame_prob", features.columns)
        model = CatBoostClassifier()
        model.load_model(
            MispricingPredictor().root / "model/local_win_expectancy.cbm"
        )
        self.assertEqual(model.feature_names_, features.columns.tolist())


if __name__ == "__main__":
    unittest.main()
