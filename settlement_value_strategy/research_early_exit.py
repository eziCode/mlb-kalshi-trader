"""Walk-forward research for a causal settlement-value early-exit overlay."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd

from settlement_value_strategy.backtest_live_policy import (
    DATA, FOLDS, RESULTS, deployed_config,
)
from settlement_value_strategy.early_exit import (
    EarlyExitConfig, apply_early_exits, prepare_exit_tapes,
)
from settlement_value_strategy.research_latency import add_future_market_target, period
from settlement_value_strategy.strategy import (
    _expit, _logit, mispricing_feature_frame, simulate_paired_both,
)


def record_metrics(frame: pd.DataFrame) -> dict:
    exits = frame.exit_time.notna() if "exit_time" in frame else pd.Series(False, index=frame.index)
    fees = frame.entry_fee.sum() + (
        frame.exit_fee.fillna(0).sum() if "exit_fee" in frame else 0.0
    )
    return {
        "trades": int(len(frame)),
        "early_exits": int(exits.sum()),
        "settlements": int((~exits).sum()),
        "fees": float(fees),
        "capital": float((frame.contracts * frame.fill_price + frame.entry_fee).sum()),
        "pnl": float(frame.pnl.sum()),
        "roi": float(frame.pnl.sum() / (
            frame.contracts * frame.fill_price + frame.entry_fee
        ).sum()) if len(frame) else 0.0,
    }


def main() -> None:
    frame = pd.read_parquet(DATA / "decision_rows.parquet")
    home = pd.read_parquet(DATA / "execution_trades.parquet")
    away = pd.read_parquet(DATA / "away_execution_trades.parquet")
    labeled = add_future_market_target(frame, home)
    config = deployed_config()
    fold_data = {}

    for fold_number, (name, (start, end)) in enumerate(FOLDS.items()):
        fit = period(labeled, "2025-01-01", start)
        test = period(labeled, start, end).copy()
        counts = fit.groupby("game_pk").size()
        model = CatBoostRegressor(
            iterations=300, depth=5, learning_rate=.025, l2_leaf_reg=40,
            loss_function="Huber:delta=0.05", random_seed=117 + fold_number,
            verbose=False, allow_writing_files=False,
        )
        model.fit(
            mispricing_feature_frame(fit),
            np.clip(fit.future_logit_move.to_numpy(float), -.75, .75),
            sample_weight=fit.game_pk.map(1.0 / counts),
        )
        test["fair_probability"] = _expit(
            _logit(test.market_home_price)
            + np.clip(model.predict(mispricing_feature_frame(test)), -.5, .5)
        )
        games = set(test.game_pk)
        game_home = home[home.game_pk.isin(games)]
        game_away = away[away.game_pk.isin(games)]
        baseline = simulate_paired_both(
            test, test.fair_probability, game_home, game_away, config,
        )
        fold_data[name] = (
            pd.DataFrame(baseline.records), test, game_home, game_away,
            prepare_exit_tapes(game_home, game_away),
        )

    candidates = [
        EarlyExitConfig(
            minimum_hold_seconds=hold,
            minimum_exit_inning=inning,
            stop_loss_points=stop,
            exit_edge_threshold=None,
        )
        for hold in (30.0, 60.0, 120.0)
        for inning in (1, 5, 7)
        for stop in (.05, .10, .15)
    ]
    tuning_names = [name for name in FOLDS if name != "final_holdout"]
    grid = []
    for candidate in candidates:
        folds = {}
        for name in tuning_names:
            records, decisions, game_home, game_away, prepared = fold_data[name]
            exited = apply_early_exits(
                records, decisions, game_home, game_away, candidate, prepared,
            )
            folds[name] = record_metrics(exited)
        pnl = sum(value["pnl"] for value in folds.values())
        baseline_pnl = sum(
            float(fold_data[name][0].pnl.sum()) for name in tuning_names
        )
        grid.append({
            "config": asdict(candidate), "folds": folds,
            "tuning_pnl": pnl, "tuning_pnl_delta": pnl - baseline_pnl,
            "profitable_folds": sum(value["pnl"] > 0 for value in folds.values()),
        })
    selected = max(
        grid,
        key=lambda row: (
            row["profitable_folds"], row["tuning_pnl"],
            -row["folds"]["early_june"]["early_exits"],
        ),
    )
    selected_config = EarlyExitConfig(**selected["config"])
    baseline_summary = {}
    selected_summary = {}
    output_records = []
    for name, (records, decisions, game_home, game_away, prepared) in fold_data.items():
        baseline_summary[name] = record_metrics(records)
        exited = apply_early_exits(
            records, decisions, game_home, game_away, selected_config, prepared,
        )
        selected_summary[name] = record_metrics(exited)
        exited["evaluation_fold"] = name
        output_records.append(exited)
    summary = {
        "method": "fixed entries; causal state-triggered exit with strictly later execution",
        "selection_folds": tuning_names,
        "untouched_holdout": "final_holdout",
        "selected_config": asdict(selected_config),
        "baseline": baseline_summary,
        "early_exit": selected_summary,
        "grid": grid,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "early_exit_research_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    pd.concat(output_records, ignore_index=True).to_csv(
        RESULTS / "early_exit_research_trades.csv", index=False
    )
    print(json.dumps({
        key: value for key, value in summary.items() if key != "grid"
    }, indent=2))


if __name__ == "__main__":
    main()
