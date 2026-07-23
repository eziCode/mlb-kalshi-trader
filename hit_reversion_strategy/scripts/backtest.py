"""Evaluate the selected hybrid strategy on exact-timestamp Kalshi trades."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trade_tape_strategy.core import (  # noqa: E402
    TradeTapeConfig,
    simulate_trade_tape,
)


DATA_DIR = REPOSITORY_ROOT / "data/shared"
STATE_UPDATES_PATH = (
    REPOSITORY_ROOT / "data/settlement_value/state_updates.parquet"
)
MODEL_DIR = PROJECT_ROOT / "models"
CONFIG_PATH = MODEL_DIR / "trade_tape_config.json"
STUDY_DIR = PROJECT_ROOT / "artifacts"
OUTER_HOLDOUT_START = pd.Timestamp("2026-06-28").date()


def main() -> None:
    config = TradeTapeConfig(**json.loads(CONFIG_PATH.read_text()))
    trades = pd.read_parquet(DATA_DIR / "home_market_trades.parquet")
    updates = pd.read_parquet(STATE_UPDATES_PATH)
    trades["game_date"] = pd.to_datetime(trades["game_date"]).dt.date
    updates["game_date"] = pd.to_datetime(updates["game_date"]).dt.date
    test_trades = trades[trades["game_date"] >= OUTER_HOLDOUT_START].copy()
    test_games = set(test_trades["game_pk"].unique())
    test_updates = updates[updates["game_pk"].isin(test_games)].copy()

    result = simulate_trade_tape(test_trades, test_updates, config)
    records = pd.DataFrame(asdict(record) for record in result.records)
    game_pnl = records.groupby("game_pk").pnl.sum()
    segment_results = {
        f"{event_type}:{side}": {
            "trades": int(len(segment)),
            "pnl": float(segment.pnl.sum()),
            "wins": int((segment.pnl > 0).sum()),
        }
        for (event_type, side), segment in records.groupby(
            ["event_type", "side"]
        )
    }
    pnl_without_best_game = float(result.pnl - game_pnl.nlargest(1).sum())
    pnl_without_top_four_games = float(
        result.pnl - game_pnl.nlargest(min(4, len(game_pnl))).sum()
    )
    deployment_enabled = bool(
        config.enabled and result.trades >= 20 and result.pnl > 0
        and result.roi > 0 and pnl_without_best_game > 0
    )
    deployment_config = replace(config, enabled=deployment_enabled)
    summary = {
        "selected_config": asdict(config),
        "deployment_config": asdict(deployment_config),
        "games": len(test_games),
        "trade_tape_rows": len(test_trades),
        "state_updates": len(test_updates),
        "observed_hits": result.observed_hits,
        "eligible_hit_updates": result.eligible_hit_updates,
        "rejected_fair_updates": result.rejected_fair_updates,
        "invalidated_candidates": result.invalidated_candidates,
        "expired_candidates": result.expired_candidates,
        "fresh_hit_anchors": result.fresh_hit_anchors,
        "confirmed_signals": result.confirmed_signals,
        "trades": result.trades,
        "yes_trades": result.yes_trades,
        "no_trades": result.no_trades,
        "reversion_exits": result.reversion_exits,
        "momentum_exits": result.momentum_exits,
        "timeout_exits": result.timeout_exits,
        "settlements": result.settlements,
        "fees": result.fees,
        "capital": result.capital,
        "pnl": result.pnl,
        "roi": result.roi,
        "pnl_without_best_game": pnl_without_best_game,
        "pnl_without_top_four_games": pnl_without_top_four_games,
        "segment_results": segment_results,
        "time_based_exit": False,
        "state_model": "MLB-only batting-perspective local win expectancy",
    }
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    (STUDY_DIR / "holdout_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    records.to_csv(STUDY_DIR / "holdout_trades.csv", index=False)

    print("EXACT-TIMESTAMP TRADE-TAPE HYBRID")
    print(f"Live enabled:          {deployment_enabled}")
    print(f"Minimum edge:          {config.minimum_edge:.1%}")
    print(f"Confirmation:          {config.confirmation_seconds:g} seconds")
    print("Time-based exit:       none")
    print(f"Games:                 {len(test_games):,}")
    print(f"Observed trades:       {len(test_trades):,}")
    print(f"Observed hits:         {result.observed_hits:,}")
    print(f"Eligible fair moves:   {result.eligible_hit_updates:,}")
    print(f"Rejected fair moves:   {result.rejected_fair_updates:,}")
    print(f"Invalidated signals:   {result.invalidated_candidates:,}")
    print(f"Expired signals:       {result.expired_candidates:,}")
    print(f"Fresh hit anchors:     {result.fresh_hit_anchors:,}")
    print(f"Confirmed signals:     {result.confirmed_signals:,}")
    print(f"Filled trades:         {result.trades:,}")
    print(f"YES / NO:              {result.yes_trades:,} / {result.no_trades:,}")
    print(f"Reversion exits:       {result.reversion_exits:,}")
    print(f"Momentum-delayed exits:{result.momentum_exits:>9,}")
    print(f"Settlements:           {result.settlements:,}")
    print(f"Fees:                  ${result.fees:,.2f}")
    print(f"Capital:               ${result.capital:,.2f}")
    print(f"Net PnL:               ${result.pnl:,.2f}")
    print(f"ROI:                   {result.roi:.2%}")


if __name__ == "__main__":
    main()
