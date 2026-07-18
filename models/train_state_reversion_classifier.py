"""Train and tune the two-stage event-agnostic overshoot EV policy."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

from catboost import CatBoostClassifier, CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    brier_score_loss, log_loss, mean_absolute_error, roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlb_kalshi.state_overshoot import (  # noqa: E402
    OvershootConfig, build_state_overshoot_candidates, simulate_state_reversion,
)
from mlb_kalshi.state_reversion import overshoot_reversion_feature_frame  # noqa: E402


DATA_DIR = PROJECT_ROOT / "data/processed/trade_tape"
MODEL_DIR = PROJECT_ROOT / "models/market_reaction_model"
STUDY_DIR = PROJECT_ROOT / "studies/state_reversion"
EXAMPLES_PATH = STUDY_DIR / "overshoot_candidates.parquet"
MODEL_PATH = MODEL_DIR / "state_reversion.cbm"
WIN_MODEL_PATH = MODEL_DIR / "state_reversion_win_pnl.cbm"
LOSS_MODEL_PATH = MODEL_DIR / "state_reversion_loss_pnl.cbm"
CALIBRATION_PATH = MODEL_DIR / "state_reversion_calibration.json"
CONFIG_PATH = MODEL_DIR / "state_reversion_config.json"
FIT_END = pd.Timestamp("2026-06-17").date()
CALIBRATION_END = pd.Timestamp("2026-06-22").date()
HOLDOUT_START = pd.Timestamp("2026-06-28").date()


def calibrated_probabilities(model, frame, calibration: dict) -> np.ndarray:
    raw = np.clip(
        model.predict_proba(overshoot_reversion_feature_frame(frame))[:, 1],
        1e-6, 1 - 1e-6,
    )
    logits = np.log(raw / (1 - raw))
    values = calibration["intercept"] + calibration["coefficient"] * logits
    return 1 / (1 + np.exp(-values))


def two_stage_predictions(
    classifier, win_model, loss_model, frame, calibration: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = overshoot_reversion_feature_frame(frame)
    probability = calibrated_probabilities(classifier, frame, calibration)
    predicted_win = np.maximum(win_model.predict(features), 0.0)
    predicted_loss = np.minimum(loss_model.predict(features), 0.0)
    raw_expected_pnl = (
        probability * predicted_win + (1 - probability) * predicted_loss
    )
    expected_pnl = (
        calibration.get("ev_intercept", 0.0)
        + calibration.get("ev_coefficient", 1.0) * raw_expected_pnl
    )
    return probability, predicted_win, predicted_loss, expected_pnl


def classification_metrics(labels, probabilities) -> dict:
    labels = np.asarray(labels, int)
    probabilities = np.asarray(probabilities, float)
    return {
        "examples": int(len(labels)), "positive_rate": float(labels.mean()),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
    }


def prediction_metrics(frame, probabilities, expected_pnl) -> dict:
    result = classification_metrics(frame["policy_profitable"], probabilities)
    result.update({
        "actual_mean_pnl": float(frame["pnl"].mean()),
        "predicted_mean_pnl": float(np.mean(expected_pnl)),
        "pnl_mae": float(mean_absolute_error(frame["pnl"], expected_pnl)),
        "pnl_correlation": float(np.corrcoef(frame["pnl"], expected_pnl)[0, 1]),
    })
    return result


def load_or_build_examples() -> pd.DataFrame:
    if EXAMPLES_PATH.exists():
        cached = pd.read_parquet(EXAMPLES_PATH)
        if "policy_version" in cached and cached["policy_version"].eq(2).all():
            return cached
    trades = pd.read_parquet(DATA_DIR / "home_market_trades.parquet")
    updates = pd.read_parquet(DATA_DIR / "state_updates.parquet")
    examples = build_state_overshoot_candidates(
        trades, updates, OvershootConfig(minimum_logit_residual=0.02)
    )
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    examples.to_parquet(EXAMPLES_PATH, index=False)
    return examples


def make_frontier(grid: pd.DataFrame) -> pd.DataFrame:
    """Best tuning configuration available at increasing activity levels."""
    rows = []
    for minimum_trades in [1, 5, 10, 20, 30, 50, 75, 100, 150, 250]:
        eligible = grid[grid["trades"] >= minimum_trades]
        if eligible.empty:
            continue
        best = eligible.sort_values(
            ["roi", "pnl", "trades"], ascending=False
        ).iloc[0].to_dict()
        best["minimum_trades"] = minimum_trades
        rows.append(best)
    return pd.DataFrame(rows)


def main() -> None:
    examples = load_or_build_examples()
    examples["game_date"] = pd.to_datetime(examples["game_date"]).dt.date
    examples["policy_profitable"] = examples["pnl"].gt(0).astype(int)
    fit = examples[examples["game_date"] < FIT_END].copy()
    calibration_frame = examples[
        (examples["game_date"] >= FIT_END)
        & (examples["game_date"] < CALIBRATION_END)
    ].copy()
    tune = examples[
        (examples["game_date"] >= CALIBRATION_END)
        & (examples["game_date"] < HOLDOUT_START)
    ].copy()
    if min(len(fit), len(calibration_frame), len(tune)) == 0:
        raise RuntimeError("Fit, calibration, and tuning partitions must be non-empty")
    winners, losers = fit[fit["pnl"] > 0], fit[fit["pnl"] <= 0]
    if min(len(winners), len(losers)) < 100:
        raise RuntimeError("Conditional PnL models require at least 100 outcomes each")

    classifier = CatBoostClassifier(
        iterations=180, depth=3, learning_rate=0.03, l2_leaf_reg=20,
        loss_function="Logloss", random_seed=42, verbose=False,
        allow_writing_files=False,
    )
    win_model = CatBoostRegressor(
        iterations=180, depth=3, learning_rate=0.03, l2_leaf_reg=20,
        loss_function="MAE", random_seed=43, verbose=False,
        allow_writing_files=False,
    )
    loss_model = CatBoostRegressor(
        iterations=180, depth=3, learning_rate=0.03, l2_leaf_reg=20,
        loss_function="MAE", random_seed=44, verbose=False,
        allow_writing_files=False,
    )
    classifier.fit(
        overshoot_reversion_feature_frame(fit), fit["policy_profitable"]
    )
    win_model.fit(overshoot_reversion_feature_frame(winners), winners["pnl"])
    loss_model.fit(overshoot_reversion_feature_frame(losers), losers["pnl"])

    raw_calibration = np.clip(
        classifier.predict_proba(
            overshoot_reversion_feature_frame(calibration_frame)
        )[:, 1], 1e-6, 1 - 1e-6,
    )
    calibrator = LogisticRegression(C=1.0, random_state=42)
    calibrator.fit(
        np.log(raw_calibration / (1 - raw_calibration)).reshape(-1, 1),
        calibration_frame["policy_profitable"],
    )
    calibration = {
        "method": "platt_logit",
        "coefficient": float(calibrator.coef_[0, 0]),
        "intercept": float(calibrator.intercept_[0]),
        "fit_start": str(FIT_END), "fit_end_exclusive": str(CALIBRATION_END),
    }
    _, _, _, raw_calibration_ev = two_stage_predictions(
        classifier, win_model, loss_model, calibration_frame, calibration
    )
    ev_calibrator = LinearRegression()
    ev_calibrator.fit(
        raw_calibration_ev.reshape(-1, 1), calibration_frame["pnl"]
    )
    calibration["ev_method"] = "linear"
    calibration["ev_coefficient"] = float(ev_calibrator.coef_[0])
    calibration["ev_intercept"] = float(ev_calibrator.intercept_)
    tune_probability, _, _, tune_ev = two_stage_predictions(
        classifier, win_model, loss_model, tune, calibration
    )

    rows = []
    for residual in [0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20, 0.30]:
        for probability in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]:
            # Negative cutoffs are included to expose the complete activity/
            # ROI frontier when absolute EV calibration is conservative.  A
            # configuration is still deployable only after positive tuning
            # and outer-holdout PnL with the minimum sample requirement.
            for minimum_ev in [
                -1.0, -0.75, -0.50, -0.25, -0.10,
                0.0, 0.05, 0.10, 0.25, 0.50, 1.0,
            ]:
                config = OvershootConfig(
                    minimum_logit_residual=residual,
                    minimum_reversion_probability=probability,
                    minimum_expected_pnl=minimum_ev,
                )
                result = simulate_state_reversion(
                    tune, tune_probability, config, expected_pnls=tune_ev
                )
                rows.append({
                    "minimum_logit_residual": residual,
                    "minimum_reversion_probability": probability,
                    "minimum_expected_pnl": minimum_ev,
                    "trades": result.accepted, "rejected": result.rejected,
                    "reversion_exits": result.reversion_exits,
                    "adverse_stop_exits": result.adverse_stop_exits,
                    "timeout_exits": result.timeout_exits,
                    "settlements": result.settlements, "fees": result.fees,
                    "capital": result.capital, "pnl": result.pnl, "roi": result.roi,
                })
    grid = pd.DataFrame(rows).sort_values(
        ["roi", "pnl", "trades"], ascending=False
    )
    frontier = make_frontier(grid)
    eligible = grid[grid["trades"] >= 20]
    selected = (
        eligible.iloc[0] if not eligible.empty
        else grid.sort_values("trades", ascending=False).iloc[0]
    )
    tuning_passed = bool(
        not eligible.empty and selected.pnl > 0 and selected.roi > 0
    )
    config = OvershootConfig(
        enabled=False,
        minimum_logit_residual=float(selected.minimum_logit_residual),
        minimum_reversion_probability=float(selected.minimum_reversion_probability),
        minimum_expected_pnl=float(selected.minimum_expected_pnl),
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    classifier.save_model(MODEL_PATH)
    win_model.save_model(WIN_MODEL_PATH)
    loss_model.save_model(LOSS_MODEL_PATH)
    CALIBRATION_PATH.write_text(json.dumps(calibration, indent=2))
    CONFIG_PATH.write_text(json.dumps({
        **asdict(config), "calibration_path": CALIBRATION_PATH.name,
        "win_model_path": WIN_MODEL_PATH.name,
        "loss_model_path": LOSS_MODEL_PATH.name,
        "target": "positive_net_policy_pnl",
        "tuning_passed": tuning_passed, "validation_passed": False,
    }, indent=2))
    grid.to_csv(STUDY_DIR / "tuning_grid.csv", index=False)
    frontier.to_csv(STUDY_DIR / "trade_count_roi_frontier.csv", index=False)

    fit_probability, _, _, fit_ev = two_stage_predictions(
        classifier, win_model, loss_model, fit, calibration
    )
    calibration_probability, _, _, calibration_ev = two_stage_predictions(
        classifier, win_model, loss_model, calibration_frame, calibration
    )
    summary = {
        "strategy": "two-stage event-agnostic policy-level expected PnL",
        "target": "positive net PnL under the complete exit policy",
        "fit_dates": f"before {FIT_END}",
        "calibration_dates": f"{FIT_END} through 2026-06-21",
        "tuning_dates": f"{CALIBRATION_END} through 2026-06-27",
        "holdout_dates": f"{HOLDOUT_START} onward (unused)",
        "fit_metrics": prediction_metrics(fit, fit_probability, fit_ev),
        "calibration_metrics": prediction_metrics(
            calibration_frame, calibration_probability, calibration_ev
        ),
        "tuning_metrics": prediction_metrics(tune, tune_probability, tune_ev),
        "selected_config": asdict(config),
        "selected_tuning_result": selected.to_dict(),
        "minimum_tuning_trades": 20, "outer_holdout_used": False,
        "frontier_path": "trade_count_roi_frontier.csv",
    }
    (STUDY_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
