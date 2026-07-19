"""Tune trade-tape threshold/persistence without using outer holdout dates."""

from __future__ import annotations

from dataclasses import asdict
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
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
MODEL_DIR = PROJECT_ROOT / "models"
CONFIG_PATH = MODEL_DIR / "trade_tape_config.json"
STUDY_DIR = PROJECT_ROOT / "artifacts"
OUTER_HOLDOUT_START = pd.Timestamp("2026-06-28").date()

_TUNE_TRADES: pd.DataFrame | None = None
_TUNE_UPDATES: pd.DataFrame | None = None


def _initialize_worker(
    tune_trades: pd.DataFrame, tune_updates: pd.DataFrame,
) -> None:
    global _TUNE_TRADES, _TUNE_UPDATES
    _TUNE_TRADES = tune_trades
    _TUNE_UPDATES = tune_updates


def _evaluate_configuration(parameters: tuple[float, float]) -> dict:
    if _TUNE_TRADES is None or _TUNE_UPDATES is None:
        raise RuntimeError("Tuning worker was not initialized")
    minimum_edge, confirmation_seconds = parameters
    config = TradeTapeConfig(
        minimum_edge=minimum_edge,
        confirmation_seconds=confirmation_seconds,
    )
    result = simulate_trade_tape(_TUNE_TRADES, _TUNE_UPDATES, config)
    return {
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
    }


def _process_context() -> mp.context.BaseContext:
    methods = mp.get_all_start_methods()
    return mp.get_context("fork" if "fork" in methods else methods[0])


def print_progress(index: int, total: int, row: dict) -> None:
    print(
        f"[{index}/{total}] "
        f"edge={row['minimum_edge']:.1%} "
        f"confirmation={row['confirmation_seconds']:g}s "
        f"ROI={row['roi']:.2%}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers", type=int, default=1,
        help=(
            "Parallel simulation processes. Start with 8 on a 14-core, "
            "48 GB machine; use 1 for sequential debugging."
        ),
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


def main() -> None:
    args = parse_args()
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

    configurations = [
        (minimum_edge, confirmation_seconds)
        for minimum_edge in [
            0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075,
            0.10, 0.125, 0.15, 0.20,
        ]
        for confirmation_seconds in [1.0, 2.0, 3.0, 5.0]
    ]
    _initialize_worker(tune_trades, tune_updates)
    if args.workers == 1:
        iterator = map(_evaluate_configuration, configurations)
        rows = []
        for index, row in enumerate(iterator, start=1):
            rows.append(row)
            print_progress(index, len(configurations), row)
    else:
        worker_count = min(
            args.workers, len(configurations), os.cpu_count() or 1
        )
        print(
            f"Evaluating {len(configurations)} configurations with "
            f"{worker_count} workers",
            flush=True,
        )
        rows = []
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=_process_context(),
            initializer=_initialize_worker,
            initargs=(tune_trades, tune_updates),
        ) as executor:
            iterator = executor.map(
                _evaluate_configuration, configurations, chunksize=1
            )
            for index, row in enumerate(iterator, start=1):
                rows.append(row)
                print_progress(index, len(configurations), row)


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
