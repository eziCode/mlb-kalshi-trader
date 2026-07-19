"""Train a calibrated event-agnostic settlement-value model."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


STRATEGY_DIR = Path(__file__).resolve().parent
if str(STRATEGY_DIR.parent) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR.parent))

from settlement_value_strategy.strategy import (  # noqa: E402
    MispricingConfig, build_mispricing_dataset, mispricing_feature_frame,
    simulate_mispricing,
)


DATA_DIR = STRATEGY_DIR.parent / "data/settlement_value"
MODEL_DIR = STRATEGY_DIR / "model"
STUDY_DIR = STRATEGY_DIR / "results"
DATASET_PATH = DATA_DIR / "decision_rows.parquet"
MODEL_PATH = MODEL_DIR / "settlement_value.cbm"
CALIBRATION_PATH = MODEL_DIR / "calibration.json"
CONFIG_PATH = MODEL_DIR / "config.json"
FIT_END = pd.Timestamp("2026-06-17").date()
CAL_END = pd.Timestamp("2026-06-22").date()
HOLDOUT_START = pd.Timestamp("2026-06-28").date()


def load_data():
    trades = pd.read_parquet(DATA_DIR / "execution_trades.parquet")
    if DATASET_PATH.exists():
        frame = pd.read_parquet(DATASET_PATH)
        if "dataset_version" in frame and frame.dataset_version.eq(1).all():
            return frame, trades
    raise FileNotFoundError(
        "data/decision_rows.parquet is required; raw dataset rebuilding is "
        "intentionally outside this self-contained training package"
    )


def calibrated_probability(model, frame, calibration):
    raw = np.clip(
        model.predict_proba(mispricing_feature_frame(frame))[:, 1],
        1e-6, 1 - 1e-6,
    )
    logits = np.log(raw / (1 - raw))
    values = calibration["intercept"] + calibration["coefficient"] * logits
    return 1 / (1 + np.exp(-values))


def metrics(frame, probability):
    return {
        "rows": len(frame), "games": int(frame.game_pk.nunique()),
        "roc_auc": float(roc_auc_score(frame.home_win, probability)),
        "log_loss": float(log_loss(frame.home_win, probability)),
        "brier": float(brier_score_loss(frame.home_win, probability)),
    }


def main() -> None:
    frame, trades = load_data()
    frame["game_date"] = pd.to_datetime(frame.game_date).dt.date
    trades["game_date"] = pd.to_datetime(trades.game_date).dt.date
    fit = frame[frame.game_date < FIT_END].copy()
    cal = frame[(frame.game_date >= FIT_END) & (frame.game_date < CAL_END)].copy()
    tune = frame[(frame.game_date >= CAL_END) & (frame.game_date < HOLDOUT_START)].copy()
    if min(len(fit), len(cal), len(tune)) == 0:
        raise RuntimeError("Mispricing chronological partitions are empty")
    counts = fit.groupby("game_pk").size()
    weights = fit.game_pk.map(1.0 / counts)
    model = CatBoostClassifier(
        iterations=350, depth=5, learning_rate=.025, l2_leaf_reg=25,
        loss_function="Logloss", random_seed=91, verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        mispricing_feature_frame(fit), fit.home_win, sample_weight=weights
    )
    raw_cal = np.clip(
        model.predict_proba(mispricing_feature_frame(cal))[:, 1], 1e-6, 1 - 1e-6
    )
    calibrator = LogisticRegression(C=1.0, random_state=91)
    cal_counts = cal.groupby("game_pk").size()
    cal_weights = cal.game_pk.map(1.0 / cal_counts)
    calibrator.fit(
        np.log(raw_cal / (1 - raw_cal)).reshape(-1, 1), cal.home_win,
        sample_weight=cal_weights,
    )
    calibration = {
        "coefficient": float(calibrator.coef_[0, 0]),
        "intercept": float(calibrator.intercept_[0]),
    }
    tune_probability = calibrated_probability(model, tune, calibration)
    tune_games = set(tune.game_pk)
    tune_trades = trades[trades.game_pk.isin(tune_games)].copy()
    dates = sorted(tune.game_date.unique())
    folds = [set(values) for values in np.array_split(dates, 3)]
    rows = []
    for minimum_ev in [0.0, .25, .50, 1.0, 1.5, 2.0]:
        for edge in [.01, .02, .03, .04, .05, .075, .10, .15]:
          for side_filter in ["both", "yes", "no"]:
            config = MispricingConfig(
                minimum_expected_pnl=minimum_ev,
                minimum_probability_edge=edge,
                side_filter=side_filter,
            )
            result = simulate_mispricing(tune, tune_probability, tune_trades, config)
            row = {
                "minimum_expected_pnl": minimum_ev,
                "minimum_probability_edge": edge,
                "side_filter": side_filter,
                "trades": result.trades, "yes_trades": result.yes_trades,
                "no_trades": result.no_trades, "pnl": result.pnl,
                "fees": result.fees, "capital": result.capital, "roi": result.roi,
            }
            fold_pnls, fold_counts, fold_rois = [], [], []
            for index, fold_dates in enumerate(folds, start=1):
                mask = tune.game_date.isin(fold_dates).to_numpy()
                fold_frame = tune.loc[mask]
                games = set(fold_frame.game_pk)
                fold_result = simulate_mispricing(
                    fold_frame, tune_probability[mask],
                    tune_trades[tune_trades.game_pk.isin(games)], config,
                )
                row[f"fold_{index}_trades"] = fold_result.trades
                row[f"fold_{index}_pnl"] = fold_result.pnl
                row[f"fold_{index}_roi"] = fold_result.roi
                fold_counts.append(fold_result.trades)
                fold_pnls.append(fold_result.pnl)
                fold_rois.append(fold_result.roi)
            row["minimum_fold_trades"] = min(fold_counts)
            row["profitable_folds"] = sum(value > 0 for value in fold_pnls)
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
    config = MispricingConfig(
        enabled=False,
        minimum_expected_pnl=float(selected.minimum_expected_pnl),
        minimum_probability_edge=float(selected.minimum_probability_edge),
        side_filter=str(selected.side_filter),
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    CALIBRATION_PATH.write_text(json.dumps(calibration, indent=2))
    CONFIG_PATH.write_text(json.dumps({
        **asdict(config),
        "tuning_passed": bool(not stable.empty and selected.pnl > 0),
        "validation_passed": False,
    }, indent=2))
    grid.sort_values(["roi", "pnl"], ascending=False).to_csv(
        STUDY_DIR / "tuning_grid.csv", index=False
    )
    fit_probability = calibrated_probability(model, fit, calibration)
    summary = {
        "event_labels_used": False,
        "target": "home team settlement outcome",
        "fit_metrics": metrics(fit, fit_probability),
        "calibration_metrics": metrics(
            cal, calibrated_probability(model, cal, calibration)
        ),
        "tuning_metrics": metrics(tune, tune_probability),
        "selected_config": asdict(config),
        "selected_tuning_result": selected.to_dict(),
        "outer_holdout_used": False,
    }
    (STUDY_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
