"""Train separate residual-alpha, maker-fill, and horizon policy models."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from catboost import CatBoostClassifier, CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlb_kalshi.residual_path import (  # noqa: E402
    PATH_HORIZONS, build_residual_path_dataset, evaluate_path_policy,
    residual_path_feature_frame,
)


DATA_DIR = PROJECT_ROOT / "data/processed/trade_tape"
MODEL_DIR = PROJECT_ROOT / "models/market_reaction_model/residual_path"
STUDY_DIR = PROJECT_ROOT / "studies/residual_path_policy"
DATASET_PATH = STUDY_DIR / "residual_paths.parquet"
CONFIG_PATH = MODEL_DIR / "policy_config.json"
FIT_END = pd.Timestamp("2026-06-17").date()
CAL_END = pd.Timestamp("2026-06-22").date()
HOLDOUT_START = pd.Timestamp("2026-06-28").date()


def load_dataset() -> pd.DataFrame:
    if DATASET_PATH.exists():
        frame = pd.read_parquet(DATASET_PATH)
        if "dataset_version" in frame and frame.dataset_version.eq(1).all():
            return frame
    trades = pd.read_parquet(DATA_DIR / "home_market_trades.parquet")
    updates = pd.read_parquet(DATA_DIR / "state_updates.parquet")
    frame = build_residual_path_dataset(trades, updates)
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(DATASET_PATH, index=False)
    return frame


def calibrated_fill(model, frame, calibration):
    raw = np.clip(
        model.predict_proba(residual_path_feature_frame(frame))[:, 1],
        1e-6, 1 - 1e-6,
    )
    logit = np.log(raw / (1 - raw))
    value = calibration["intercept"] + calibration["coefficient"] * logit
    return 1 / (1 + np.exp(-value))


def main() -> None:
    frame = load_dataset()
    frame["game_date"] = pd.to_datetime(frame.game_date).dt.date
    fit = frame[frame.game_date < FIT_END].copy()
    calibration_frame = frame[
        (frame.game_date >= FIT_END) & (frame.game_date < CAL_END)
    ].copy()
    tune = frame[
        (frame.game_date >= CAL_END) & (frame.game_date < HOLDOUT_START)
    ].copy()
    if min(len(fit), len(calibration_frame), len(tune)) == 0:
        raise RuntimeError("Residual-path chronological partitions are empty")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    STUDY_DIR.mkdir(parents=True, exist_ok=True)

    fill_model = CatBoostClassifier(
        iterations=250, depth=4, learning_rate=.03, l2_leaf_reg=20,
        loss_function="Logloss", random_seed=71, verbose=False,
        allow_writing_files=False,
    )
    fill_model.fit(residual_path_feature_frame(fit), fit.maker_filled)
    raw_cal = np.clip(
        fill_model.predict_proba(
            residual_path_feature_frame(calibration_frame)
        )[:, 1], 1e-6, 1 - 1e-6,
    )
    calibrator = LogisticRegression(C=1.0, random_state=71)
    calibrator.fit(
        np.log(raw_cal / (1 - raw_cal)).reshape(-1, 1),
        calibration_frame.maker_filled,
    )
    calibration = {
        "coefficient": float(calibrator.coef_[0, 0]),
        "intercept": float(calibrator.intercept_[0]),
    }
    fill_model.save_model(MODEL_DIR / "maker_fill.cbm")
    (MODEL_DIR / "maker_fill_calibration.json").write_text(
        json.dumps(calibration, indent=2)
    )

    alpha_metrics = {}
    pnl_models = {}
    for horizon in PATH_HORIZONS:
        contraction_label = f"contraction_{horizon}s"
        alpha_train = fit.dropna(subset=[contraction_label])
        alpha_model = CatBoostRegressor(
            iterations=250, depth=4, learning_rate=.03, l2_leaf_reg=20,
            loss_function="RMSE", random_seed=100 + horizon, verbose=False,
            allow_writing_files=False,
        )
        alpha_model.fit(
            residual_path_feature_frame(alpha_train),
            alpha_train[contraction_label].clip(-2, 2),
        )
        alpha_model.save_model(MODEL_DIR / f"contraction_{horizon}s.cbm")
        valid = tune.dropna(subset=[contraction_label])
        predicted = alpha_model.predict(residual_path_feature_frame(valid))
        alpha_metrics[str(horizon)] = {
            "examples": len(valid),
            "correlation": float(np.corrcoef(valid[contraction_label], predicted)[0, 1]),
            "actual_mean": float(valid[contraction_label].mean()),
            "predicted_mean": float(np.mean(predicted)),
        }

        pnl_label = f"net_pnl_{horizon}s"
        pnl_train = fit.dropna(subset=[pnl_label])
        pnl_model = CatBoostRegressor(
            iterations=250, depth=4, learning_rate=.03, l2_leaf_reg=20,
            loss_function="MAE", random_seed=200 + horizon, verbose=False,
            allow_writing_files=False,
        )
        pnl_model.fit(
            residual_path_feature_frame(pnl_train), pnl_train[pnl_label]
        )
        pnl_model.save_model(MODEL_DIR / f"conditional_pnl_{horizon}s.cbm")
        pnl_models[horizon] = pnl_model

    tune_fill = calibrated_fill(fill_model, tune, calibration)
    tune_dates = sorted(tune.game_date.unique())
    folds = [set(values) for values in np.array_split(tune_dates, 3)]
    rows = []
    for horizon, pnl_model in pnl_models.items():
        predicted_pnl = pnl_model.predict(residual_path_feature_frame(tune))
        for fill_floor in [.1, .2, .3, .4, .5, .6]:
            for ev_floor in [-.25, 0, .05, .10, .25, .50]:
                result = evaluate_path_policy(
                    tune, tune_fill, predicted_pnl, horizon, fill_floor, ev_floor
                )
                row = {
                    "horizon_seconds": horizon,
                    "minimum_fill_probability": fill_floor,
                    "minimum_expected_pnl": ev_floor,
                    **{key: result[key] for key in ["attempts", "trades", "pnl", "capital", "roi"]},
                }
                fold_pnls, fold_counts, fold_rois = [], [], []
                for index, dates in enumerate(folds, start=1):
                    mask = tune.game_date.isin(dates).to_numpy()
                    fold = evaluate_path_policy(
                        tune.loc[mask], tune_fill[mask], predicted_pnl[mask],
                        horizon, fill_floor, ev_floor,
                    )
                    row[f"fold_{index}_trades"] = fold["trades"]
                    row[f"fold_{index}_pnl"] = fold["pnl"]
                    row[f"fold_{index}_roi"] = fold["roi"]
                    fold_counts.append(fold["trades"])
                    fold_pnls.append(fold["pnl"])
                    fold_rois.append(fold["roi"])
                row["profitable_folds"] = sum(value > 0 for value in fold_pnls)
                row["minimum_fold_trades"] = min(fold_counts)
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
    config = {
        "enabled": False,
        "horizon_seconds": int(selected.horizon_seconds),
        "minimum_fill_probability": float(selected.minimum_fill_probability),
        "minimum_expected_pnl": float(selected.minimum_expected_pnl),
        "tuning_passed": bool(not stable.empty and selected.pnl > 0),
        "validation_passed": False,
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2))
    grid.sort_values(["roi", "pnl"], ascending=False).to_csv(
        STUDY_DIR / "policy_tuning_grid.csv", index=False
    )
    summary = {
        "data_counts": {
            "fit": len(fit), "calibration": len(calibration_frame), "tune": len(tune),
        },
        "event_labels_used": False,
        "fill_auc_calibration": float(roc_auc_score(
            calibration_frame.maker_filled,
            calibrated_fill(fill_model, calibration_frame, calibration),
        )),
        "fill_auc_tuning": float(roc_auc_score(tune.maker_filled, tune_fill)),
        "alpha_metrics": alpha_metrics,
        "selected_config": config,
        "selected_tuning_result": selected.to_dict(),
        "outer_holdout_used": False,
    }
    (STUDY_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
