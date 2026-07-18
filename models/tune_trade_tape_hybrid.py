"""Tune trade-tape threshold/persistence without using outer holdout dates."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlb_kalshi.trade_tape import (  # noqa: E402
    TradeTapeConfig,
    simulate_trade_tape,
)


DATA_DIR = PROJECT_ROOT / "data/processed/trade_tape"
MODEL_DIR = PROJECT_ROOT / "models/market_reaction_model"
CONFIG_PATH = MODEL_DIR / "trade_tape_config.json"
STUDY_DIR = PROJECT_ROOT / "studies/trade_tape_hybrid"
OUTER_HOLDOUT_START = pd.Timestamp("2026-06-28").date()


def main() -> None:
    trades = pd.read_parquet(DATA_DIR / "home_market_trades.parquet")
    updates = pd.read_parquet(DATA_DIR / "state_updates.parquet")
    trades["game_date"] = pd.to_datetime(trades["game_date"]).dt.date
    updates["game_date"] = pd.to_datetime(updates["game_date"]).dt.date

    pre_holdout = trades[trades["game_date"] < OUTER_HOLDOUT_START]
    dates = sorted(pre_holdout["game_date"].unique())
    tuning_start = dates[int(len(dates) * 0.75)]
    tune_trades = pre_holdout[pre_holdout["game_date"] >= tuning_start].copy()
    tune_games = set(tune_trades["game_pk"].unique())
    tune_updates = updates[updates["game_pk"].isin(tune_games)].copy()

    rows = []
    for minimum_edge in [
        0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075,
        0.10, 0.125, 0.15, 0.20,
    ]:
        for confirmation_seconds in [1.0, 2.0, 3.0, 5.0]:
            config = TradeTapeConfig(
                minimum_edge=minimum_edge,
                confirmation_seconds=confirmation_seconds,
            )
            result = simulate_trade_tape(tune_trades, tune_updates, config)
            rows.append({
                "minimum_edge": minimum_edge,
                "confirmation_seconds": confirmation_seconds,
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
                "settlements": result.settlements,
                "fees": result.fees,
                "capital": result.capital,
                "pnl": result.pnl,
                "roi": result.roi,
            })

    grid = pd.DataFrame(rows).sort_values(
        ["roi", "pnl", "trades"], ascending=False
    )
    eligible = grid[grid["trades"] >= 30]
    selection = eligible.iloc[0] if not eligible.empty else grid.iloc[0]
    enabled = bool(selection["pnl"] > 0 and selection["roi"] > 0)
    config = TradeTapeConfig(
        enabled=enabled,
        minimum_edge=float(selection["minimum_edge"]),
        confirmation_seconds=float(selection["confirmation_seconds"]),
        minimum_fair_move=0.005,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(asdict(config), indent=2))
    grid.to_csv(STUDY_DIR / "tuning_grid.csv", index=False)
    summary = {
        "tuning_start": str(tuning_start),
        "tuning_end": str(max(dates)),
        "tuning_games": len(tune_games),
        "selection_rule": "maximum net ROI among configurations with >=30 trades",
        "selected_config": asdict(config),
        "selected_tuning_result": selection.to_dict(),
        "outer_holdout_used": False,
        "execution_model": (
            "strictly later compatible taker-side trade with sufficient reported size"
        ),
        "time_based_exit": False,
    }
    (STUDY_DIR / "tuning_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"Tuning dates: {tuning_start} through {max(dates)}")
    print(f"Tuning games: {len(tune_games):,}")
    print("Top configurations:")
    print(grid.head(12).to_string(index=False, formatters={
        "roi": "{:.2%}".format,
        "pnl": "${:,.2f}".format,
        "fees": "${:,.2f}".format,
    }))
    print(f"Saved {CONFIG_PATH} (live enabled={enabled})")


if __name__ == "__main__":
    main()
