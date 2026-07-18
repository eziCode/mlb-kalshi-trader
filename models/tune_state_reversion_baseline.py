"""Tune a transparent state-overreaction policy without machine learning."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlb_kalshi.state_overshoot import (  # noqa: E402
    OvershootConfig, build_state_overshoot_candidates, simulate_state_reversion,
)


DATA_DIR = PROJECT_ROOT / "data/processed/trade_tape"
MODEL_DIR = PROJECT_ROOT / "models/market_reaction_model"
STUDY_DIR = PROJECT_ROOT / "studies/state_reversion"
CONFIG_PATH = MODEL_DIR / "state_reversion_baseline_config.json"
TUNE_START = pd.Timestamp("2026-06-22").date()
HOLDOUT_START = pd.Timestamp("2026-06-28").date()


def evaluate(frame: pd.DataFrame, config: OvershootConfig):
    return simulate_state_reversion(
        frame, np.ones(len(frame)), config, expected_pnls=np.ones(len(frame))
    )


def main() -> None:
    trades = pd.read_parquet(DATA_DIR / "home_market_trades.parquet")
    updates = pd.read_parquet(DATA_DIR / "state_updates.parquet")
    for frame in (trades, updates):
        frame["game_date"] = pd.to_datetime(frame["game_date"]).dt.date
    tune_games = set(updates.loc[
        (updates.game_date >= TUNE_START) & (updates.game_date < HOLDOUT_START),
        "game_pk",
    ])
    tune_trades = trades[trades.game_pk.isin(tune_games)].copy()
    tune_updates = updates[updates.game_pk.isin(tune_games)].copy()
    tune_dates = sorted(tune_updates.game_date.unique())
    fold_dates = [set(values) for values in np.array_split(tune_dates, 3)]

    rows = []
    for fraction in [0.25, 0.50, 0.75, 1.0]:
        for state_exit in ["none", "next_pitch", "next_plate_appearance"]:
            outcome_seconds = 300.0 if state_exit == "next_plate_appearance" else 120.0
            build_config = OvershootConfig(
                minimum_logit_residual=0.04,
                minimum_fair_logit_move=0.02,
                minimum_target_profit=0.25,
                observation_latency_buffer_seconds=2.0,
                maximum_outcome_seconds=outcome_seconds,
                entry_execution="maker", exit_execution="maker",
                reversion_fraction=fraction, state_exit=state_exit,
            )
            candidates = build_state_overshoot_candidates(
                tune_trades, tune_updates, build_config
            )
            for residual in [0.04, 0.08, 0.12, 0.16, 0.20, 0.30]:
                for max_latency in [3.0, 5.0, 7.0, 10.0]:
                    for maximum_inning in [6, 8, 9, 99]:
                      for side_filter in ["both", "yes", "no"]:
                        selected = candidates[
                            (candidates.absolute_logit_residual >= residual)
                            & (candidates.entry_latency_seconds <= max_latency)
                            & (candidates.inning_after <= maximum_inning)
                        ]
                        if side_filter != "both":
                            selected = selected[selected.side.eq(side_filter)]
                        policy = OvershootConfig(
                            minimum_logit_residual=residual,
                            minimum_fair_logit_move=0.02,
                            minimum_target_profit=0.25,
                            observation_latency_buffer_seconds=2.0,
                            entry_execution="maker", exit_execution="maker",
                            maximum_outcome_seconds=outcome_seconds,
                            reversion_fraction=fraction, state_exit=state_exit,
                            minimum_reversion_probability=0.0,
                            minimum_expected_pnl=0.0,
                        )
                        result = evaluate(selected, policy)
                        row = {
                            "reversion_fraction": fraction,
                            "state_exit": state_exit,
                            "minimum_logit_residual": residual,
                            "maximum_selected_entry_latency": max_latency,
                            "maximum_inning": maximum_inning,
                            "side_filter": side_filter,
                            "trades": result.accepted, "pnl": result.pnl,
                            "capital": result.capital, "roi": result.roi,
                            "reversion_exits": result.reversion_exits,
                            "thesis_invalidations": result.thesis_invalidations,
                            "timeout_exits": result.timeout_exits,
                        }
                        fold_pnls, fold_rois, fold_counts = [], [], []
                        for fold_index, dates in enumerate(fold_dates, start=1):
                            fold = selected[selected.game_date.isin(dates)]
                            fold_result = evaluate(fold, policy)
                            row[f"fold_{fold_index}_trades"] = fold_result.accepted
                            row[f"fold_{fold_index}_pnl"] = fold_result.pnl
                            row[f"fold_{fold_index}_roi"] = fold_result.roi
                            fold_counts.append(fold_result.accepted)
                            fold_pnls.append(fold_result.pnl)
                            fold_rois.append(fold_result.roi)
                        row["minimum_fold_trades"] = min(fold_counts)
                        row["profitable_folds"] = sum(
                            value > 0 for value in fold_pnls
                        )
                        row["worst_fold_roi"] = min(fold_rois)
                        rows.append(row)

    grid = pd.DataFrame(rows)
    stable = grid[
        (grid.trades >= 20) & (grid.minimum_fold_trades >= 3)
        & (grid.profitable_folds == 3)
    ].sort_values(["worst_fold_roi", "roi", "pnl"], ascending=False)
    aggregate = grid[grid.trades >= 20].sort_values(
        ["roi", "pnl", "trades"], ascending=False
    )
    selected = stable.iloc[0] if not stable.empty else aggregate.iloc[0]
    passed = bool(not stable.empty and selected.pnl > 0)
    selected_config = OvershootConfig(
        enabled=False,
        minimum_logit_residual=float(selected.minimum_logit_residual),
        minimum_fair_logit_move=0.02, minimum_target_profit=0.25,
        observation_latency_buffer_seconds=2.0,
        entry_execution="maker", exit_execution="maker",
        reversion_fraction=float(selected.reversion_fraction),
        state_exit=str(selected.state_exit),
        maximum_outcome_seconds=(
            300.0 if selected.state_exit == "next_plate_appearance" else 120.0
        ),
        minimum_reversion_probability=0.0, minimum_expected_pnl=0.0,
    )
    payload = {
        **asdict(selected_config),
        "maximum_selected_entry_latency": float(
            selected.maximum_selected_entry_latency
        ),
        "maximum_inning": int(selected.maximum_inning),
        "side_filter": str(selected.side_filter),
        "tuning_passed": passed, "validation_passed": False,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(payload, indent=2))
    grid.sort_values(["roi", "pnl"], ascending=False).to_csv(
        STUDY_DIR / "deterministic_tuning_grid.csv", index=False
    )
    summary = {
        "selection_rule": (
            "no ML; >=20 trades, >=3 per fold, all three folds profitable, "
            "maximize worst-fold ROI"
        ),
        "selected_config": payload,
        "selected_tuning_result": selected.to_dict(),
        "outer_holdout_used": False,
    }
    (STUDY_DIR / "deterministic_training_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    # Build the selected outcome semantics once across all dates. Selection is
    # already frozen; the holdout is not inspected here.
    full_candidates = build_state_overshoot_candidates(
        trades, updates, selected_config
    )
    full_candidates.to_parquet(
        STUDY_DIR / "deterministic_candidates.parquet", index=False
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
