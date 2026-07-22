"""Chronological research harness for conservative post-event repricing.

This is intentionally separate from the frozen live predictor.  It learns the
market's short-horizon log-odds move, not the final game result, and evaluates
the resulting anchored probabilities with the production fill simulator.
"""

from __future__ import annotations

import json
from pathlib import Path

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd

from settlement_value_strategy.strategy import (
    MISPRICING_FEATURES,
    MispricingConfig,
    _expit,
    _logit,
    mispricing_feature_frame,
    simulate_paired_both,
)


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data/settlement_value"
RESULTS = ROOT / "results"


def add_future_market_target(
    frame: pd.DataFrame, trades: pd.DataFrame, horizon_seconds: float = 10.0,
) -> pd.DataFrame:
    """Attach the last observed home price 3-10 seconds after each signal.

    The horizon is truncated immediately before the next MLB state update so
    the label cannot include information from a later pitch.
    """
    pieces: list[pd.DataFrame] = []
    for game_pk, rows in frame.groupby("game_pk", sort=False):
        tape = trades[trades.game_pk.eq(game_pk)].sort_values(
            ["created_time", "trade_id"]
        )
        if tape.empty:
            continue
        times = pd.to_datetime(tape.created_time, utc=True).array.as_unit("ns").asi8
        prices = tape.yes_price_dollars.to_numpy(float)
        selected = rows.copy()
        signal = pd.to_datetime(selected.signal_time, utc=True).array.as_unit("ns").asi8
        horizon = signal + int(horizon_seconds * 1e9)
        next_update = pd.to_datetime(selected.next_update_time, utc=True)
        has_next = next_update.notna().to_numpy()
        next_ns = next_update.array.as_unit("ns").asi8
        horizon[has_next] = np.minimum(horizon[has_next], next_ns[has_next] - 1)
        target_i = np.searchsorted(times, horizon, side="right") - 1
        minimum = signal + 3_000_000_000
        valid = (target_i >= 0) & (horizon >= minimum)
        valid &= np.where(target_i >= 0, times[np.maximum(target_i, 0)] >= minimum, False)
        selected = selected.loc[valid].copy()
        chosen = target_i[valid]
        selected["future_market_price"] = prices[chosen]
        selected["future_market_seconds"] = (
            times[chosen] - signal[valid]
        ) / 1e9
        selected["future_logit_move"] = (
            _logit(selected.future_market_price)
            - _logit(selected.market_home_price)
        )
        pieces.append(selected)
    if not pieces:
        return frame.iloc[:0].copy()
    return pd.concat(pieces, ignore_index=True)


def period(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    dates = pd.to_datetime(frame.game_date).dt.date
    return frame[
        (dates >= pd.Timestamp(start).date())
        & (dates < pd.Timestamp(end).date())
    ].copy()


def metrics(result) -> dict:
    records = pd.DataFrame(result.records)
    if records.empty:
        return {"trades": 0, "pnl": 0.0, "roi": 0.0}
    games = records.groupby("game_pk").pnl.sum()
    days = records.groupby("game_date").pnl.sum()
    return {
        "trades": int(result.trades),
        "games": int(records.game_pk.nunique()),
        "pnl": float(result.pnl),
        "roi": float(result.roi),
        "win_rate": float(records.pnl.gt(0).mean()),
        "pnl_without_top_4_games": float(
            result.pnl - games.nlargest(min(4, len(games))).sum()
        ),
        "profitable_day_fraction": float(days.gt(0).mean()),
        "worst_day_pnl": float(days.min()),
    }


def main() -> None:
    frame = pd.read_parquet(DATA / "decision_rows.parquet")
    home = pd.read_parquet(DATA / "execution_trades.parquet")
    away = pd.read_parquet(DATA / "away_execution_trades.parquet")
    labeled = add_future_market_target(frame, home)
    fold_ranges = {
        "april": ("2026-04-01", "2026-05-01"),
        "may": ("2026-05-01", "2026-06-01"),
        "early_june": ("2026-06-01", "2026-06-17"),
        "late_june": ("2026-06-17", "2026-06-28"),
        "early_july": ("2026-06-28", "2026-07-11"),
        "final": ("2026-07-11", "2026-07-18"),
    }
    partitions = {
        name: period(labeled, start, end)
        for name, (start, end) in fold_ranges.items()
    }
    predicted: dict[str, np.ndarray] = {}
    final_model = None
    for fold_number, (name, (start, _)) in enumerate(fold_ranges.items()):
        fit = period(labeled, "2025-01-01", start)
        counts = fit.groupby("game_pk").size()
        weights = fit.game_pk.map(1.0 / counts)
        model = CatBoostRegressor(
            iterations=300,
            depth=5,
            learning_rate=.025,
            l2_leaf_reg=40,
            loss_function="Huber:delta=0.05",
            random_seed=117 + fold_number,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(
            mispricing_feature_frame(fit),
            np.clip(fit.future_logit_move.to_numpy(float), -.75, .75),
            sample_weight=weights,
        )
        predicted[name] = model.predict(
            mispricing_feature_frame(partitions[name])
        )
        if name == "final":
            final_model = model
    tuning_folds = [name for name in fold_ranges if name != "final"]
    grid: list[dict] = []
    for shrinkage in (.25, .5, .75, 1.0):
        for edge in (.01, .015, .02, .025, .03, .04):
            for maximum_positions in (1, 2):
                config = MispricingConfig(
                    minimum_expected_pnl=0.0,
                    minimum_probability_edge=edge,
                    side_filter="both",
                    execution_contract="paired_both",
                    maximum_positions_per_game=maximum_positions,
                    minimum_seconds_between_entries=200.0,
                    conditional_stacking=True,
                )
                item = {
                    "shrinkage": shrinkage,
                    "minimum_probability_edge": edge,
                    "maximum_positions_per_game": maximum_positions,
                }
                for name in tuning_folds:
                    rows = partitions[name]
                    probability = _expit(
                        _logit(rows.market_home_price)
                        + shrinkage * np.clip(predicted[name], -.5, .5)
                    )
                    games = set(rows.game_pk)
                    result = simulate_paired_both(
                        rows,
                        probability,
                        home[home.game_pk.isin(games)],
                        away[away.game_pk.isin(games)],
                        config,
                    )
                    item[name] = metrics(result)
                grid.append(item)
    eligible = []
    for item in grid:
        fold_metrics = [item[name] for name in tuning_folds]
        total_trades = sum(value["trades"] for value in fold_metrics)
        total_pnl = sum(value["pnl"] for value in fold_metrics)
        profitable_folds = sum(value["roi"] > 0 for value in fold_metrics)
        robust_folds = sum(
            value.get("pnl_without_top_4_games", -1) > 0
            for value in fold_metrics
        )
        item["tuning_total_trades"] = total_trades
        item["tuning_total_pnl"] = total_pnl
        item["profitable_folds"] = profitable_folds
        item["robust_folds"] = robust_folds
        if (
            total_trades >= 200
            and total_pnl > 0
            and profitable_folds >= 4
            and robust_folds >= 3
        ):
            eligible.append(item)
    eligible.sort(key=lambda item: (
        item["profitable_folds"],
        item["robust_folds"],
        item["tuning_total_pnl"],
        item["tuning_total_trades"],
    ), reverse=True)
    selected = eligible[0] if eligible else None
    final_result = None
    selected_records: list[pd.DataFrame] = []
    if selected is not None:
        config = MispricingConfig(
            minimum_expected_pnl=0.0,
            minimum_probability_edge=selected["minimum_probability_edge"],
            side_filter="both",
            execution_contract="paired_both",
            maximum_positions_per_game=selected["maximum_positions_per_game"],
            minimum_seconds_between_entries=200.0,
            conditional_stacking=True,
        )
        for name, rows in partitions.items():
            probability = _expit(
                _logit(rows.market_home_price)
                + selected["shrinkage"] * np.clip(predicted[name], -.5, .5)
            )
            games = set(rows.game_pk)
            result = simulate_paired_both(
                rows,
                probability,
                home[home.game_pk.isin(games)],
                away[away.game_pk.isin(games)],
                config,
            )
            records = pd.DataFrame(result.records)
            if not records.empty:
                records["evaluation_fold"] = name
                selected_records.append(records)
            if name == "final":
                final_result = result
    summary = {
        "target": "causal 3-10 second home-market logit move",
        "features": list(MISPRICING_FEATURES),
        "rows": {name: len(rows) for name, rows in partitions.items()},
        "eligible_configurations": len(eligible),
        "selected": selected,
        "final": metrics(final_result) if final_result is not None else None,
        "grid": grid,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "latency_research_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    if selected_records:
        pd.concat(selected_records, ignore_index=True).to_csv(
            RESULTS / "latency_research_trades.csv", index=False
        )
    if final_model is not None:
        final_model.save_model(RESULTS / "latency_research.cbm")
    print(json.dumps({key: value for key, value in summary.items() if key != "grid"}, indent=2))


if __name__ == "__main__":
    main()
