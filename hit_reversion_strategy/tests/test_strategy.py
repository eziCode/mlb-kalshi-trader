from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import pandas as pd

from scripts.paper_trade import (
    EventCandidate, match_games_to_home_markets, pre_pitch_trade_anchor,
    Position, replay_candidate_entry, replay_position_exit,
    SharedPaperPortfolio,
    should_surface_worker_line,
)
from trade_tape_strategy.core import (
    TradeTapeConfig,
    simulate_trade_tape,
    trade_signal,
)


class TradeTapeStrategyTests(unittest.TestCase):
    def test_cooldown_survives_position_close_without_capping_game(self):
        start = pd.Timestamp("2026-07-20T12:00:00Z").to_pydatetime()
        first = Position("no", 10, .5, .1, start, .4, .4, 1)
        too_soon = Position(
            "no", 10, .5, .1,
            start + pd.Timedelta(seconds=179), .4, .4, 2,
        )
        allowed = Position(
            "no", 10, .5, .1,
            start + pd.Timedelta(seconds=180), .4, .4, 3,
        )
        with tempfile.TemporaryDirectory() as directory:
            portfolio = SharedPaperPortfolio(Path(directory) / "paper.sqlite3")
            self.assertTrue(portfolio.open_position(1, "ticker", first, 180))
            self.assertTrue(portfolio.close_position(1, 1, 10))
            self.assertFalse(portfolio.open_position(1, "ticker", too_soon, 180))
            self.assertTrue(portfolio.open_position(1, "ticker", allowed, 180))

    def test_live_anchor_uses_last_trade_strictly_before_pitch_start(self):
        start = pd.Timestamp("2026-07-20T12:00:05Z")
        trades = pd.DataFrame({
            "created_time": [
                start - pd.Timedelta(seconds=2),
                start, start + pd.Timedelta(seconds=1),
            ],
            "yes_price_dollars": [.41, .70, .80],
        })
        self.assertEqual(pre_pitch_trade_anchor(trades, start, 5), .41)
        self.assertIsNone(pre_pitch_trade_anchor(trades, start, 1))

    def test_live_entry_requires_later_compatible_trade_after_confirmation(self):
        event = pd.Timestamp("2026-07-20T12:00:00Z")
        trades = pd.DataFrame({
            "trade_id": [1, 2, 3],
            "created_time": [
                event + pd.Timedelta(seconds=.1),
                event + pd.Timedelta(seconds=1.2),
                event + pd.Timedelta(seconds=1.3),
            ],
            "yes_price_dollars": [.30, .30, .30],
            "count_fp": [100, 100, 100],
            "taker_outcome_side": ["yes", "no", "yes"],
        })
        candidate = EventCandidate(
            side="yes", target=.60, event_id=1, event_type="double",
            observed_at=event.to_pydatetime(), event_time=event.to_pydatetime(),
            pre_market=.5, pre_fair=.5, post_fair=.6,
            material_state=(0, 0, 0, 0, 0), pitch_token=None,
        )
        config = TradeTapeConfig(
            minimum_edge=.10, confirmation_seconds=1,
            allowed_event_types=("double",),
        )
        fill = replay_candidate_entry(trades, candidate, .6, [], config)
        self.assertIsNotNone(fill)
        self.assertEqual(fill["time"], trades.created_time.iloc[2].to_pydatetime())

    def test_live_exit_requires_trade_after_reversion_with_opposite_taker(self):
        entry = pd.Timestamp("2026-07-20T12:00:00Z")
        position = Position(
            side="yes", contracts=10, entry_price=.4, entry_fee=.1,
            entry_time=entry.to_pydatetime(), anchor_target=.6,
            anchor_fair=.6, event_id=1,
        )
        trades = pd.DataFrame({
            "trade_id": [1, 2, 3],
            "created_time": [
                entry + pd.Timedelta(seconds=1),
                entry + pd.Timedelta(seconds=2),
                entry + pd.Timedelta(seconds=3),
            ],
            "yes_price_dollars": [.61, .62, .63],
            "count_fp": [100, 100, 100],
            "taker_outcome_side": ["yes", "yes", "no"],
        })
        fill, pending, scanned = replay_position_exit(trades, position, .6)
        self.assertIsNotNone(fill)
        self.assertEqual(fill["price"], .63)
        self.assertIsNone(pending)
        self.assertEqual(scanned, trades.created_time.iloc[2].to_pydatetime())

    def test_doubleheader_pairs_all_games_before_filtering_final(self):
        games = [
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
        ]
        events = [{
            "event_ticker": f"KXMLBGAME-G{number}",
            "markets": [
                {"ticker": f"KXMLBGAME-G{number}-NYM"},
                {"ticker": f"KXMLBGAME-G{number}-ATL"},
            ],
        } for number in (1, 2)]
        matched, warnings = match_games_to_home_markets(games, events)
        self.assertEqual(warnings, [])
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0].game_pk, 2)
        self.assertEqual(matched[0].market_ticker, "KXMLBGAME-G2-ATL")

    def test_main_log_surfaces_readiness_and_trades(self):
        self.assertTrue(should_surface_worker_line("TRADER READY game_pk=1"))
        self.assertTrue(should_surface_worker_line("TRADE SELL YES"))
        self.assertFalse(should_surface_worker_line("Initialized baseline"))

    def test_trade_signal_is_relative_to_yes_price_after_fees(self):
        self.assertEqual(trade_signal(0.60, 0.50, 0.05)[0], "yes")
        self.assertEqual(trade_signal(0.40, 0.50, 0.05)[0], "no")
        self.assertIsNone(trade_signal(0.52, 0.50, 0.05)[0])

    def test_trade_signal_rejects_gross_edge_consumed_by_fees(self):
        side, net_edge = trade_signal(0.56, 0.50, 0.05)
        self.assertIsNone(side)
        self.assertAlmostEqual(net_edge, 0.025)

    def test_position_has_no_time_based_exit(self):
        trades, updates = self._frames(include_reversion=False)
        result = simulate_trade_tape(
            trades, updates, TradeTapeConfig(minimum_edge=0.05)
        )
        self.assertEqual(result.trades, 1)
        self.assertEqual(result.reversion_exits, 0)
        self.assertEqual(result.settlements, 1)
        self.assertEqual(result.records[0].exit_reason, "settlement")
        self.assertEqual(result.records[0].trigger_at_bat, 1)
        self.assertEqual(
            result.records[0].trigger_event_time,
            pd.Timestamp("2026-07-01T12:00:01Z"),
        )

    def test_hit_without_meaningful_directional_fair_move_is_rejected(self):
        trades, updates = self._frames(include_reversion=False)
        updates["fair_after"] = updates["fair_before"]
        result = simulate_trade_tape(trades, updates, TradeTapeConfig())
        self.assertEqual(result.trades, 0)
        self.assertEqual(result.rejected_fair_updates, 1)

    def test_next_completed_plate_appearance_invalidates_candidate(self):
        trades, updates = self._frames(include_reversion=False)
        next_pa = updates.iloc[0].copy()
        next_pa["pitch_start_time"] = pd.Timestamp("2026-07-01T12:00:01.5Z")
        next_pa["pitch_end_time"] = pd.Timestamp("2026-07-01T12:00:02Z")
        next_pa["is_hit"] = False
        next_pa["completed_event"] = "field_out"
        next_pa["completed_event_batting_home"] = True
        next_pa["at_bat_number"] = 2
        next_pa["pitch_number"] = 1
        next_pa["fair_before"] = 0.50
        next_pa["fair_after"] = 0.48
        next_pa["outs_when_up_after"] = 1
        updates = pd.concat([updates, next_pa.to_frame().T], ignore_index=True)
        result = simulate_trade_tape(
            trades, updates, TradeTapeConfig(minimum_edge=0.05)
        )
        self.assertEqual(result.trades, 0)
        self.assertEqual(result.invalidated_candidates, 1)

    def test_next_pitch_invalidates_candidate_before_entry(self):
        trades, updates = self._frames(include_reversion=False)
        later_pitch = updates.iloc[0].copy()
        later_pitch["pitch_start_time"] = pd.Timestamp(
            "2026-07-01T12:00:01.5Z"
        )
        later_pitch["pitch_end_time"] = pd.Timestamp(
            "2026-07-01T12:00:03.25Z"
        )
        later_pitch["is_hit"] = False
        later_pitch["completed_event"] = None
        later_pitch["at_bat_number"] = 2
        later_pitch["pitch_number"] = 1
        updates = pd.concat(
            [updates, later_pitch.to_frame().T], ignore_index=True
        )

        result = simulate_trade_tape(
            trades,
            updates,
            TradeTapeConfig(minimum_edge=0.05),
        )
        self.assertEqual(result.trades, 0)
        self.assertEqual(result.invalidated_candidates, 1)

    def test_event_candidate_expires_before_late_fill(self):
        trades, updates = self._frames(include_reversion=False)
        trades.loc[
            trades["created_time"] > pd.Timestamp("2026-07-01T12:00:01Z"),
            "created_time",
        ] += pd.Timedelta(seconds=20)

        result = simulate_trade_tape(
            trades,
            updates,
            TradeTapeConfig(
                minimum_edge=0.05,
                maximum_event_to_entry_seconds=10.0,
            ),
        )
        self.assertEqual(result.trades, 0)
        self.assertEqual(result.expired_candidates, 1)

    def test_reversion_requires_a_later_compatible_trade(self):
        trades, updates = self._frames(include_reversion=True)
        result = simulate_trade_tape(
            trades, updates, TradeTapeConfig(minimum_edge=0.05)
        )
        self.assertEqual(result.trades, 1)
        self.assertEqual(result.reversion_exits, 1)
        self.assertEqual(result.settlements, 0)
        self.assertEqual(result.records[0].exit_reason, "reversion")

    def test_favorable_velocity_delays_exit_until_trailing_giveback(self):
        trades, updates = self._frames(include_reversion=False)
        momentum = pd.DataFrame({
            "game_pk": 1,
            "trade_id": ["m1", "m2", "m3", "m4", "m5"],
            "created_time": pd.to_datetime([
                "2026-07-01T12:00:03.8Z",
                "2026-07-01T12:00:04.2Z",
                "2026-07-01T12:00:04.6Z",
                "2026-07-01T12:00:05.0Z",
                "2026-07-01T12:00:05.1Z",
            ], utc=True),
            "yes_price_dollars": [0.47, 0.50, 0.54, 0.52, 0.52],
            "no_price_dollars": [0.53, 0.50, 0.46, 0.48, 0.48],
            "count_fp": 100.0,
            "taker_outcome_side": ["no", "no", "no", "no", "no"],
            "home_win": 1,
        })
        trades = pd.concat([trades, momentum], ignore_index=True)
        config = TradeTapeConfig(
            minimum_edge=0.05,
            momentum_exit_enabled=True,
            momentum_window_seconds=2.0,
            minimum_favorable_velocity=0.01,
            momentum_trailing_giveback=0.01,
            momentum_max_hold_seconds=2.0,
            minimum_momentum_trades=3,
        )
        result = simulate_trade_tape(trades, updates, config)
        self.assertEqual(result.momentum_exits, 1)
        self.assertEqual(result.records[0].exit_reason, "momentum_reversion")
        self.assertEqual(result.records[0].exit_price, 0.52)

    @staticmethod
    def _frames(include_reversion: bool):
        times = [
            "2026-07-01T11:59:59Z",  # safe pre-pitch anchor
            "2026-07-01T12:00:01.100Z",
            "2026-07-01T12:00:03.200Z",  # confirms persistence
            "2026-07-01T12:00:03.300Z",  # later compatible entry
            "2026-07-01T12:05:00Z",      # no timeout despite elapsed time
        ]
        prices = [0.40, 0.40, 0.40, 0.40, 0.41]
        taker = ["yes", "yes", "yes", "yes", "no"]
        if include_reversion:
            times += [
                "2026-07-01T12:05:01Z",  # observes reversion
                "2026-07-01T12:05:02Z",  # later compatible exit
            ]
            prices += [0.51, 0.51]
            taker += ["no", "no"]
        trades = pd.DataFrame({
            "game_pk": 1,
            "trade_id": [str(index) for index in range(len(times))],
            "created_time": pd.to_datetime(times, utc=True, format="mixed"),
            "yes_price_dollars": prices,
            "no_price_dollars": [1 - price for price in prices],
            "count_fp": 100.0,
            "taker_outcome_side": taker,
            "home_win": 1,
        })
        updates = pd.DataFrame({
            "game_pk": [1],
            "pitch_start_time": pd.to_datetime(
                ["2026-07-01T12:00:00Z"], utc=True
            ),
            "pitch_end_time": pd.to_datetime(
                ["2026-07-01T12:00:01Z"], utc=True
            ),
            "is_hit": [True],
            "completed_event": ["single"],
            "completed_event_batting_home": [True],
            "at_bat_number": [1],
            "pitch_number": [3],
            "fair_before": [0.40],
            "fair_after": [0.50],
            "score_diff_after": [0],
            "outs_when_up_after": [0],
            "runner_on_first_after": [1],
            "runner_on_second_after": [0],
            "runner_on_third_after": [0],
        })
        return trades, updates


if __name__ == "__main__":
    unittest.main()
