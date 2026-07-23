"""Robustness-first, pre-holdout tuning for the hit-reversion policy."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from trade_tape_strategy.core import TradeTapeConfig, simulate_trade_tape


OUTER_HOLDOUT_START = pd.Timestamp("2026-06-28").date()
DATA = REPOSITORY / "data/shared/home_market_trades.parquet"
UPDATES = REPOSITORY / "data/shared/state_updates.parquet"
CONFIG = PROJECT / "models/trade_tape_config.json"
RESULTS = PROJECT / "artifacts"
MODEL = PROJECT / "models/local_win_expectancy.cbm"
MODEL_METADATA = MODEL.with_suffix(".metadata.json")
UPDATES_METADATA = UPDATES.with_suffix(".metadata.json")


def assert_leak_free_inputs(trades: pd.DataFrame) -> dict:
    model_metadata = json.loads(MODEL_METADATA.read_text())
    updates_metadata = json.loads(UPDATES_METADATA.read_text())
    strategy_start = min(trades.game_date)
    training_cutoff = pd.Timestamp(
        model_metadata["training_cutoff_exclusive"]
    ).date()
    model_hash = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    if training_cutoff > strategy_start:
        raise RuntimeError(
            f"State-model cutoff {training_cutoff} overlaps strategy data "
            f"beginning {strategy_start}"
        )
    if model_metadata["model_sha256"] != model_hash:
        raise RuntimeError("State-model provenance hash does not match model")
    if updates_metadata["model_sha256"] != model_hash:
        raise RuntimeError("State updates were not scored by the audited model")
    return {
        "strategy_data_start": str(strategy_start),
        "state_model_training_cutoff_exclusive": str(training_cutoff),
        "state_model_sha256": model_hash,
        "state_updates_sha256_verified": True,
    }


def candidate_config(parameters) -> TradeTapeConfig:
    edge, confirmation, maximum_hold, target_mode, latch = parameters
    return TradeTapeConfig(
        minimum_edge=edge,
        confirmation_seconds=confirmation,
        maximum_pre_event_trade_age_seconds=10.0,
        maximum_event_to_entry_seconds=20.0,
        invalidate_on_next_pitch=True,
        minimum_fair_move=0.0,
        minimum_seconds_between_entries=60.0,
        allowed_event_types=("single", "double", "triple"),
        maximum_hold_seconds=maximum_hold,
        minimum_reversion_move=0.0,
        side_filter="both",
        position_sizing="fixed_payout",
        require_compatible_taker=False,
        exit_target_mode=target_mode,
        latch_reversion_exit=latch,
    )


def evaluate(config, trades, updates, folds) -> dict:
    fold_metrics = []
    record_frames = []
    totals = {"trades": 0, "pnl": 0.0, "capital": 0.0, "fees": 0.0}
    for number, dates in enumerate(folds, 1):
        fold_trades = trades[trades.game_date.isin(dates)]
        games = set(fold_trades.game_pk)
        result = simulate_trade_tape(
            fold_trades, updates[updates.game_pk.isin(games)], config
        )
        item = {
            "fold": number, "games": len(games), "trades": result.trades,
            "pnl": result.pnl, "capital": result.capital, "roi": result.roi,
        }
        fold_metrics.append(item)
        for key in totals:
            totals[key] += getattr(result, key)
        if result.records:
            record_frames.append(pd.DataFrame(
                record.__dict__ for record in result.records
            ))
    records = pd.concat(record_frames, ignore_index=True)
    game_pnl = records.groupby("game_pk").pnl.sum()
    total_games = sum(item["games"] for item in fold_metrics)
    return {
        **asdict(config), **totals,
        "games": total_games,
        "trades_per_game": totals["trades"] / total_games,
        "roi": totals["pnl"] / totals["capital"],
        "pnl_without_best_game": (
            totals["pnl"] - float(game_pnl.nlargest(1).sum())
        ),
        "profitable_folds": sum(item["pnl"] > 0 for item in fold_metrics),
        "worst_fold_roi": min(item["roi"] for item in fold_metrics),
        "minimum_fold_trades_per_game": min(
            item["trades"] / item["games"] for item in fold_metrics
        ),
        "folds": fold_metrics,
    }


def main() -> None:
    trades = pd.read_parquet(DATA)
    updates = pd.read_parquet(UPDATES)
    trades["game_date"] = pd.to_datetime(trades.game_date).dt.date
    updates["game_date"] = pd.to_datetime(updates.game_date).dt.date
    leakage_audit = assert_leak_free_inputs(trades)
    pre_holdout = trades[trades.game_date < OUTER_HOLDOUT_START]
    dates = sorted(pre_holdout.game_date.unique())
    tuning_start = dates[int(len(dates) * .75)]
    tuning = pre_holdout[pre_holdout.game_date >= tuning_start].copy()
    games = set(tuning.game_pk)
    tuning_updates = updates[updates.game_pk.isin(games)].copy()
    folds = [set(values) for values in np.array_split(
        sorted(tuning.game_date.unique()), 3
    )]
    parameters = [
        (edge, confirmation, hold, target, latch)
        for edge in (.045, .05, .075)
        for confirmation in (0.0, 1.0)
        for hold in (0.0, 120.0, 300.0)
        for target in ("dynamic", "frozen")
        for latch in (False, True)
    ]
    rows = []
    for index, values in enumerate(parameters, 1):
        row = evaluate(
            candidate_config(values), tuning, tuning_updates, folds
        )
        rows.append(row)
        print(
            f"[{index}/{len(parameters)}] edge={row['minimum_edge']:.1%} "
            f"confirm={row['confirmation_seconds']:g}s "
            f"hold={row['maximum_hold_seconds']:g}s "
            f"target={row['exit_target_mode']} "
            f"latch={row['latch_reversion_exit']} ROI={row['roi']:.2%}",
            flush=True,
        )
    eligible = [row for row in rows if (
        row["trades_per_game"] >= .15
        and row["minimum_fold_trades_per_game"] >= .10
        and row["profitable_folds"] == 3
        and row["worst_fold_roi"] > 0
        and row["pnl_without_best_game"] > 0
    )]
    eligible.sort(key=lambda row: (
        row["worst_fold_roi"], row["pnl_without_best_game"],
        row["roi"], row["trades"],
    ), reverse=True)
    if not eligible:
        raise RuntimeError("No candidate passed the predeclared robustness gates")
    selected = eligible[0]
    config = candidate_config((
        selected["minimum_edge"], selected["confirmation_seconds"],
        selected["maximum_hold_seconds"], selected["exit_target_mode"],
        selected["latch_reversion_exit"],
    ))
    RESULTS.mkdir(exist_ok=True)
    serializable = [{
        key: value for key, value in row.items() if key != "folds"
    } for row in rows]
    pd.DataFrame(serializable).to_csv(
        RESULTS / "robust_tuning_grid.csv", index=False
    )
    summary = {
        "selection_data_end": str(OUTER_HOLDOUT_START),
        "outer_holdout_used_for_selection": False,
        "leakage_audit": leakage_audit,
        "selection_rule": (
            "minimum 0.15 trades/game overall and 0.10 in each fold; all "
            "three chronological folds profitable; worst-fold ROI positive; "
            "profit positive without best game; rank worst-fold ROI first"
        ),
        "eligible_candidates": len(eligible),
        "selected_config": asdict(config),
        "selected_tuning_result": selected,
    }
    (RESULTS / "robust_tuning_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    CONFIG.write_text(json.dumps(asdict(config), indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
