"""Evaluate the frozen settlement-value strategy on development holdout."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


STRATEGY_DIR = Path(__file__).resolve().parent
if str(STRATEGY_DIR.parent) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR.parent))

from settlement_value_strategy.strategy import (  # noqa: E402
    MispricingConfig, market_adjusted_probability, mispricing_feature_frame,
    simulate_away_yes, simulate_mispricing, simulate_paired_both,
)


DATA_DIR = STRATEGY_DIR.parent / "data/settlement_value"
MODEL_DIR = STRATEGY_DIR / "model"
STUDY_DIR = STRATEGY_DIR / "results"
HOLDOUT_START = pd.Timestamp("2026-06-28").date()


def main() -> None:
    raw_config = json.loads((MODEL_DIR / "config.json").read_text())
    config = MispricingConfig(**{
        key: value for key, value in raw_config.items()
        if key in MispricingConfig.__dataclass_fields__
    })
    calibration = json.loads((MODEL_DIR / "calibration.json").read_text())
    frame = pd.read_parquet(DATA_DIR / "decision_rows.parquet")
    tape_name = (
        "away_execution_trades.parquet"
        if config.execution_contract == "away_yes"
        else "execution_trades.parquet"
    )
    trades = pd.read_parquet(DATA_DIR / tape_name)
    away_trades = (
        pd.read_parquet(DATA_DIR / "away_execution_trades.parquet")
        if config.execution_contract == "paired_both"
        else None
    )
    frame["game_date"] = pd.to_datetime(frame.game_date).dt.date
    trades["game_date"] = pd.to_datetime(trades.game_date).dt.date
    frame = frame[frame.game_date >= HOLDOUT_START].copy()
    games = set(frame.game_pk)
    trades = trades[trades.game_pk.isin(games)].copy()
    if away_trades is not None:
        away_trades = away_trades[away_trades.game_pk.isin(games)].copy()
    model = CatBoostClassifier()
    model.load_model(MODEL_DIR / "settlement_value.cbm")
    raw = np.clip(
        model.predict_proba(mispricing_feature_frame(frame))[:, 1], 1e-6, 1 - 1e-6
    )
    probability = market_adjusted_probability(
        raw, frame.market_home_price.to_numpy(float), calibration
    )
    if config.execution_contract == "paired_both":
        result = simulate_paired_both(
            frame, probability, trades, away_trades, config
        )
    else:
        simulator = (
            simulate_away_yes
            if config.execution_contract == "away_yes"
            else simulate_mispricing
        )
        result = simulator(frame, probability, trades, config)
    records = pd.DataFrame(result.records)
    if records.empty:
        game_pnl = pd.Series(dtype=float)
        daily = pd.DataFrame(columns=["trades", "pnl", "win_rate", "roi"])
    else:
        game_pnl = records.groupby("game_pk").pnl.sum()
        daily = records.groupby("game_date").agg(
            trades=("pnl", "size"), pnl=("pnl", "sum"),
            win_rate=("pnl", lambda values: values.gt(0).mean()),
        )
        daily["roi"] = daily.pnl / (daily.trades * config.bet_size)
    top_count = min(4, len(game_pnl))
    top_game_pnl = float(game_pnl.nlargest(top_count).sum())
    pnl_without_top_games = float(result.pnl - top_game_pnl)
    profitable_day_fraction = (
        float(daily.pnl.gt(0).mean()) if len(daily) else 0.0
    )
    worst_day_roi = float(daily.roi.min()) if len(daily) else 0.0
    robustness = {
        "top_game_count": top_count,
        "top_game_pnl": top_game_pnl,
        "pnl_without_top_games": pnl_without_top_games,
        "profitable_day_fraction": profitable_day_fraction,
        "worst_day_roi": worst_day_roi,
    }
    if len(game_pnl):
        game_capital = records.groupby("game_pk").apply(
            lambda group: float(
                config.bet_size * len(group) + group.entry_fee.sum()
            ),
            include_groups=False,
        )
        rng = np.random.default_rng(20260720)
        indices = rng.integers(0, len(game_pnl), size=(10_000, len(game_pnl)))
        pnl_values = game_pnl.to_numpy(float)[indices].sum(axis=1)
        capital_values = game_capital.to_numpy(float)[indices].sum(axis=1)
        bootstrap_roi = pnl_values / capital_values
        robustness["game_bootstrap_roi_95_low"] = float(
            np.quantile(bootstrap_roi, .025)
        )
        robustness["game_bootstrap_roi_95_high"] = float(
            np.quantile(bootstrap_roi, .975)
        )
        robustness["game_bootstrap_probability_positive"] = float(
            np.mean(bootstrap_roi > 0)
        )
    passed = bool(
        result.trades >= 20
        and result.pnl > 0
        and result.roi > .10
        and pnl_without_top_games > 0
        and profitable_day_fraction >= .60
        and worst_day_roi >= -.35
    )
    raw_config["validation_passed"] = passed
    raw_config["enabled"] = False
    (MODEL_DIR / "config.json").write_text(json.dumps(raw_config, indent=2))
    summary = {
        "holdout_start": str(HOLDOUT_START), "decision_rows": len(frame),
        "games": len(games), "config": raw_config,
        "orders": result.orders, "trades": result.trades,
        "yes_trades": result.yes_trades, "no_trades": result.no_trades,
        "fees": result.fees, "capital": result.capital,
        "pnl": result.pnl, "roi": result.roi,
        "robustness": robustness,
        "probability_metrics": {
            "roc_auc": float(roc_auc_score(frame.home_win, probability)),
            "log_loss": float(log_loss(frame.home_win, probability)),
            "brier": float(brier_score_loss(frame.home_win, probability)),
        },
        "validation_passed": passed,
    }
    (STUDY_DIR / "holdout_summary.json").write_text(json.dumps(summary, indent=2))
    records.to_csv(STUDY_DIR / "holdout_trades.csv", index=False)
    if not records.empty:
        segments = records.groupby("side").agg(
            trades=("pnl", "size"), pnl=("pnl", "sum"),
            mean_pnl=("pnl", "mean"), win_rate=("pnl", lambda x: x.gt(0).mean()),
        )
        segments["roi"] = segments.pnl / (segments.trades * 10.0)
        segments.to_csv(STUDY_DIR / "holdout_side_summary.csv")
        daily.to_csv(STUDY_DIR / "holdout_daily_summary.csv")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
