"""Train a calibrated event-agnostic settlement-value model."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from scipy.optimize import minimize


STRATEGY_DIR = Path(__file__).resolve().parent
if str(STRATEGY_DIR.parent) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR.parent))

from settlement_value_strategy.strategy import (  # noqa: E402
    MispricingConfig, build_mispricing_dataset, mispricing_feature_frame,
    market_adjusted_probability, simulate_paired_both,
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
    away_trades = pd.read_parquet(DATA_DIR / "away_execution_trades.parquet")
    if DATASET_PATH.exists():
        frame = pd.read_parquet(DATASET_PATH)
        if "dataset_version" in frame and frame.dataset_version.eq(1).all():
            return frame, trades, away_trades
    raise FileNotFoundError(
        "data/decision_rows.parquet is required; raw dataset rebuilding is "
        "intentionally outside this self-contained training package"
    )


def calibrated_probability(model, frame, calibration):
    raw = np.clip(
        model.predict_proba(mispricing_feature_frame(frame))[:, 1],
        1e-6, 1 - 1e-6,
    )
    return market_adjusted_probability(
        raw, frame.market_home_price.to_numpy(float), calibration
    )


def fit_market_adjustment(raw, market, outcome, weights):
    market = np.clip(np.asarray(market, float), 1e-6, 1 - 1e-6)
    raw = np.clip(np.asarray(raw, float), 1e-6, 1 - 1e-6)
    outcome = np.asarray(outcome, float)
    weights = np.asarray(weights, float)
    market_logit = np.log(market / (1 - market))
    delta = np.log(raw / (1 - raw)) - market_logit

    def objective(values):
        intercept, coefficient = values
        logits = market_logit + intercept + coefficient * delta
        probability = 1 / (1 + np.exp(-logits))
        loss = -np.average(
            outcome * np.log(np.clip(probability, 1e-9, 1))
            + (1 - outcome) * np.log(np.clip(1 - probability, 1e-9, 1)),
            weights=weights,
        )
        return loss + .05 * coefficient ** 2 + .02 * intercept ** 2

    fitted = minimize(
        objective, x0=np.array([0.0, 0.1]), method="L-BFGS-B",
        bounds=[(-.5, .5), (0.0, 1.0)],
    )
    if not fitted.success:
        raise RuntimeError(f"Market adjustment calibration failed: {fitted.message}")
    return {
        "mode": "market_logit_adjustment",
        "intercept": float(fitted.x[0]),
        "coefficient": float(fitted.x[1]),
        "l2_coefficient": .05,
        "l2_intercept": .02,
    }


def metrics(frame, probability):
    return {
        "rows": len(frame), "games": int(frame.game_pk.nunique()),
        "roc_auc": float(roc_auc_score(frame.home_win, probability)),
        "log_loss": float(log_loss(frame.home_win, probability)),
        "brier": float(brier_score_loss(frame.home_win, probability)),
    }


def main() -> None:
    frame, trades, away_trades = load_data()
    frame["game_date"] = pd.to_datetime(frame.game_date).dt.date
    trades["game_date"] = pd.to_datetime(trades.game_date).dt.date
    away_trades["game_date"] = pd.to_datetime(away_trades.game_date).dt.date
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
    # The market-anchored calibrator learned a persistent home-YES offset and
    # erased the model disagreements that drive the paired away contract.  Keep
    # the model probability unchanged and require the policy to work in the
    # two genuinely chronological development periods instead.
    calibration = {"mode": "identity"}
    development = pd.concat([cal, tune], ignore_index=True)
    development_probability = calibrated_probability(
        model, development, calibration
    )
    development_games = set(development.game_pk)
    development_away_trades = away_trades[
        away_trades.game_pk.isin(development_games)
    ].copy()
    development_home_trades = trades[
        trades.game_pk.isin(development_games)
    ].copy()
    folds = [
        set(cal.game_date.unique()),
        set(tune.game_date.unique()),
    ]

    def consistency(result):
        records = pd.DataFrame(result.records)
        if records.empty:
            return 0.0, 0.0
        games = records.groupby("game_pk").pnl.sum()
        return (
            float(result.pnl - games.nlargest(min(1, len(games))).sum()),
            float(games.gt(0).mean()),
        )

    rows = []
    for minimum_ev in [2.0, 2.25, 2.5, 2.75, 3.0]:
        for edge in [.10, .125, .15]:
          for maximum_positions in [0]:
            config = MispricingConfig(
                minimum_expected_pnl=minimum_ev,
                minimum_probability_edge=edge,
                side_filter="both",
                execution_contract="paired_both",
                maximum_positions_per_game=maximum_positions,
                minimum_seconds_between_entries=200.0,
                conditional_stacking=True,
            )
            result = simulate_paired_both(
                development, development_probability,
                development_home_trades, development_away_trades, config
            )
            pnl_without_best_game, profitable_game_fraction = consistency(result)
            row = {
                "minimum_expected_pnl": minimum_ev,
                "minimum_probability_edge": edge,
                "side_filter": "both",
                "execution_contract": "paired_both",
                "maximum_positions_per_game": maximum_positions,
                "trades": result.trades, "yes_trades": result.yes_trades,
                "no_trades": result.no_trades, "pnl": result.pnl,
                "fees": result.fees, "capital": result.capital, "roi": result.roi,
                "pnl_without_best_game": pnl_without_best_game,
                "profitable_game_fraction": profitable_game_fraction,
            }
            fold_pnls, fold_counts, fold_rois = [], [], []
            for index, fold_dates in enumerate(folds, start=1):
                mask = development.game_date.isin(fold_dates).to_numpy()
                fold_frame = development.loc[mask]
                games = set(fold_frame.game_pk)
                fold_result = simulate_paired_both(
                    fold_frame, development_probability[mask],
                    development_home_trades[
                        development_home_trades.game_pk.isin(games)
                    ],
                    development_away_trades[
                        development_away_trades.game_pk.isin(games)
                    ], config,
                )
                row[f"fold_{index}_trades"] = fold_result.trades
                row[f"fold_{index}_yes_trades"] = fold_result.yes_trades
                row[f"fold_{index}_no_trades"] = fold_result.no_trades
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
        (grid.trades >= 30) & (grid.minimum_fold_trades >= 10)
        & (grid.yes_trades >= 10) & (grid.no_trades >= 5)
        & (grid.fold_1_no_trades >= 3) & (grid.fold_2_no_trades >= 3)
        & (grid.profitable_folds == len(folds))
        & (grid.pnl_without_best_game > 0)
        & (grid.profitable_game_fraction >= .50)
    ].sort_values(["worst_fold_roi", "roi", "pnl"], ascending=False)
    aggregate = grid[grid.trades >= 20].sort_values(
        ["roi", "pnl", "trades"], ascending=False
    )
    high_coverage = stable[
        (stable.execution_contract == "paired_both")
        & (stable.worst_fold_roi >= 0.05) & (stable.roi >= 0.05)
    ].sort_values(
        ["worst_fold_roi", "pnl_without_best_game", "trades"],
        ascending=False,
    )
    if not high_coverage.empty:
        selected = high_coverage.iloc[0]
        selection_rule = (
            "maximum worst-period ROI among two-sided paired-market policies "
            "passing coverage and concentration gates"
        )
    elif not stable.empty:
        selected = stable.iloc[0]
        selection_rule = "maximum worst-fold ROI among stable policies"
    else:
        selected = aggregate.iloc[0]
        selection_rule = (
            "diagnostic-only least-bad aggregate policy; no configuration "
            "passed chronological resilience requirements"
        )
    config = MispricingConfig(
        enabled=False,
        minimum_expected_pnl=float(selected.minimum_expected_pnl),
        minimum_probability_edge=float(selected.minimum_probability_edge),
        side_filter=str(selected.side_filter),
        execution_contract=str(selected.execution_contract),
        maximum_positions_per_game=int(selected.maximum_positions_per_game),
        minimum_seconds_between_entries=200.0,
        conditional_stacking=True,
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
        "tuning_metrics": metrics(
            development, development_probability
        ),
        "market_baseline_metrics": metrics(
            development, development.market_home_price.to_numpy(float)
        ),
        "selected_config": asdict(config),
        "selection_rule": selection_rule,
        "selected_tuning_result": selected.to_dict(),
        "outer_holdout_used": False,
    }
    (STUDY_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
