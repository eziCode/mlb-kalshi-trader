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
    MispricingConfig, mispricing_feature_frame, simulate_away_yes,
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
    trades = pd.read_parquet(DATA_DIR / "away_execution_trades.parquet")
    frame["game_date"] = pd.to_datetime(frame.game_date).dt.date
    trades["game_date"] = pd.to_datetime(trades.game_date).dt.date
    frame = frame[frame.game_date >= HOLDOUT_START].copy()
    games = set(frame.game_pk)
    trades = trades[trades.game_pk.isin(games)].copy()
    model = CatBoostClassifier()
    model.load_model(MODEL_DIR / "settlement_value.cbm")
    raw = np.clip(
        model.predict_proba(mispricing_feature_frame(frame))[:, 1], 1e-6, 1 - 1e-6
    )
    logits = np.log(raw / (1 - raw))
    probability = 1 / (1 + np.exp(-(
        calibration["intercept"] + calibration["coefficient"] * logits
    )))
    result = simulate_away_yes(frame, probability, trades, config)
    passed = bool(result.trades >= 20 and result.pnl > 0 and result.roi > .10)
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
        "probability_metrics": {
            "roc_auc": float(roc_auc_score(frame.home_win, probability)),
            "log_loss": float(log_loss(frame.home_win, probability)),
            "brier": float(brier_score_loss(frame.home_win, probability)),
        },
        "validation_passed": passed,
    }
    (STUDY_DIR / "holdout_summary.json").write_text(json.dumps(summary, indent=2))
    records = pd.DataFrame(result.records)
    records.to_csv(STUDY_DIR / "holdout_trades.csv", index=False)
    if not records.empty:
        segments = records.groupby("side").agg(
            trades=("pnl", "size"), pnl=("pnl", "sum"),
            mean_pnl=("pnl", "mean"), win_rate=("pnl", lambda x: x.gt(0).mean()),
        )
        segments["roi"] = segments.pnl / (segments.trades * 10.0)
        segments.to_csv(STUDY_DIR / "holdout_side_summary.csv")
        daily = records.groupby("game_date").agg(
            trades=("pnl", "size"), pnl=("pnl", "sum"),
            win_rate=("pnl", lambda x: x.gt(0).mean()),
        )
        daily["roi"] = daily.pnl / (daily.trades * 10.0)
        daily.to_csv(STUDY_DIR / "holdout_daily_summary.csv")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
