"""Train and evaluate the 2026 empirical-reaction/reversion strategy."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

from catboost import CatBoostClassifier, CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlb_kalshi.empirical_reaction import (  # noqa: E402
    REACTION_CATEGORICAL_FEATURES,
    REACTION_FEATURES,
    REVERSION_CATEGORICAL_FEATURES,
    REVERSION_FEATURES,
    EmpiricalStrategyConfig,
    build_reversion_candidates,
    evaluate_candidates,
    reaction_feature_frame,
    reversion_feature_frame,
)


DATA_DIR = PROJECT_ROOT / "data/processed/trade_tape"
EVENTS_PATH = DATA_DIR / "empirical_reaction_events.parquet"
CANDIDATES_PATH = DATA_DIR / "empirical_reversion_candidates.parquet"
TRADES_PATH = DATA_DIR / "home_market_trades.parquet"
MODEL_DIR = PROJECT_ROOT / "models/market_reaction_model"
REACTION_MODEL_PATH = MODEL_DIR / "empirical_reaction.cbm"
REVERSION_MODEL_PATH = MODEL_DIR / "empirical_reversion.cbm"
CONFIG_PATH = MODEL_DIR / "empirical_strategy_config.json"
STUDY_DIR = PROJECT_ROOT / "studies/empirical_reversion_strategy"

REACTION_FIT_END = pd.Timestamp("2026-06-01").date()
CLASSIFIER_FIT_END = pd.Timestamp("2026-06-17").date()
TUNING_END = pd.Timestamp("2026-06-28").date()
HOLDOUT_END = pd.Timestamp("2026-07-11").date()


def game_balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("game_pk")["game_pk"].transform("size")
    weights = 1.0 / counts.to_numpy(dtype=float)
    return weights / weights.mean()


def reaction_metrics(frame: pd.DataFrame, predictions) -> dict:
    target = frame["actual_batting_logit_move"].to_numpy(dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    return {
        "events": len(frame),
        "games": int(frame["game_pk"].nunique()),
        "mae_logit": float(np.mean(np.abs(target - predictions))),
        "rmse_logit": float(np.sqrt(np.mean((target - predictions) ** 2))),
    }


def classifier_metrics(frame: pd.DataFrame) -> dict:
    target = frame["profitable_reversion"].to_numpy(dtype=int)
    probability = frame["predicted_reversion_probability"].to_numpy(float)
    return {
        "candidates": len(frame),
        "games": int(frame["game_pk"].nunique()),
        "positive_rate": float(target.mean()),
        "roc_auc": float(roc_auc_score(target, probability)),
        "log_loss": float(log_loss(target, probability)),
        "brier_score": float(brier_score_loss(target, probability)),
    }


def result_dict(result) -> dict:
    return {
        "trades": result.trades,
        "reversion_exits": result.reversion_exits,
        "settlements": result.settlements,
        "fees": result.fees,
        "capital": result.capital,
        "pnl": result.pnl,
        "roi": result.roi,
    }


def accepted_trade_rows(candidates: pd.DataFrame, result) -> pd.DataFrame:
    accepted = candidates[candidates["candidate_id"].isin(result.accepted_ids)].copy()
    if accepted.empty:
        return accepted
    settlement_won = (
        ((accepted["entry_side"] == "yes") & (accepted["home_win"] == 1))
        | ((accepted["entry_side"] == "no") & (accepted["home_win"] == 0))
    )
    accepted["exit_reason"] = np.where(
        accepted["profitable_reversion"].astype(bool), "reversion", "settlement"
    )
    accepted["exit_time"] = accepted["reversion_exit_time"]
    accepted["exit_price"] = np.where(
        accepted["profitable_reversion"].astype(bool),
        accepted["reversion_exit_price"],
        np.where(settlement_won, 1.0, 0.0),
    )
    accepted["pnl"] = np.where(
        accepted["profitable_reversion"].astype(bool),
        accepted["contracts"] * accepted["reversion_exit_price"]
        - accepted["reversion_exit_fee"]
        - accepted["contracts"] * accepted["entry_price"]
        - accepted["entry_fee"],
        np.where(settlement_won, accepted["contracts"], 0.0)
        - accepted["contracts"] * accepted["entry_price"]
        - accepted["entry_fee"],
    )
    columns = [
        "candidate_id", "game_pk", "game_date", "event_type", "entry_side",
        "event_end_time", "entry_time", "exit_time", "exit_reason",
        "pre_batting_price", "predicted_batting_logit_move",
        "actual_batting_logit_move", "probability_residual",
        "predicted_reversion_probability", "expected_home_price",
        "entry_home_price", "entry_price", "exit_price", "contracts",
        "entry_fee", "reversion_exit_fee", "home_win", "pnl",
    ]
    return accepted.loc[:, columns].sort_values(["game_date", "entry_time"])


def main() -> None:
    events = pd.read_parquet(EVENTS_PATH)
    trades = pd.read_parquet(TRADES_PATH)
    event_dates = pd.to_datetime(events["game_date"]).dt.date
    reaction_fit = events[event_dates < REACTION_FIT_END].copy()
    if reaction_fit.empty:
        raise RuntimeError("The pre-June empirical-reaction fit split is empty")

    reaction_model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.03,
        depth=5,
        loss_function="RMSE",
        l2_leaf_reg=10.0,
        random_seed=42,
        allow_writing_files=False,
        verbose=False,
    )
    reaction_model.fit(
        reaction_feature_frame(reaction_fit),
        reaction_fit["actual_batting_logit_move"],
        cat_features=list(REACTION_CATEGORICAL_FEATURES),
        sample_weight=game_balanced_weights(reaction_fit),
    )
    eligible_events = events[
        (event_dates >= REACTION_FIT_END) & (event_dates < HOLDOUT_END)
    ].copy()
    candidates = build_reversion_candidates(
        eligible_events, trades, reaction_model, minimum_entry_residual=0.01
    )
    if candidates.empty:
        raise RuntimeError("No causally fillable empirical-reversion candidates")
    candidates.to_parquet(CANDIDATES_PATH, index=False)

    dates = pd.to_datetime(candidates["game_date"]).dt.date
    classifier_fit = candidates[dates < CLASSIFIER_FIT_END].copy()
    tuning = candidates[
        (dates >= CLASSIFIER_FIT_END) & (dates < TUNING_END)
    ].copy()
    holdout = candidates[
        (dates >= TUNING_END) & (dates < HOLDOUT_END)
    ].copy()
    if classifier_fit.empty or tuning.empty or holdout.empty:
        raise RuntimeError("A chronological classifier split is empty")

    reversion_model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.03,
        depth=3,
        loss_function="Logloss",
        eval_metric="AUC",
        l2_leaf_reg=20.0,
        random_seed=43,
        allow_writing_files=False,
        verbose=False,
    )
    reversion_model.fit(
        reversion_feature_frame(classifier_fit),
        classifier_fit["profitable_reversion"],
        cat_features=list(REVERSION_CATEGORICAL_FEATURES),
        sample_weight=game_balanced_weights(classifier_fit),
    )
    for split in (classifier_fit, tuning, holdout):
        split["predicted_reversion_probability"] = reversion_model.predict_proba(
            reversion_feature_frame(split)
        )[:, 1]

    grid_rows = []
    for margin in [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10]:
        config = EmpiricalStrategyConfig(
            minimum_probability_residual=0.01,
            minimum_reversion_probability=0.50,
            minimum_reversion_probability_margin=margin,
        )
        result = evaluate_candidates(tuning, config)
        grid_rows.append({
            "minimum_probability_residual": 0.01,
            "minimum_reversion_probability": 0.50,
            "minimum_reversion_probability_margin": margin,
            **result_dict(result),
        })
    grid = pd.DataFrame(grid_rows).sort_values(
        ["roi", "pnl", "trades"], ascending=False
    )
    eligible = grid[grid["trades"] >= 20]
    selection = (
        eligible.iloc[0]
        if not eligible.empty
        else grid[grid["minimum_reversion_probability_margin"] == 0.0].iloc[0]
    )
    selected = EmpiricalStrategyConfig(
        enabled=False,
        minimum_probability_residual=float(
            selection["minimum_probability_residual"]
        ),
        minimum_reversion_probability=float(
            selection["minimum_reversion_probability"]
        ),
        minimum_reversion_probability_margin=float(
            selection["minimum_reversion_probability_margin"]
        ),
    )
    holdout_result = evaluate_candidates(holdout, selected)
    validated = bool(
        selection["pnl"] > 0
        and selection["roi"] > 0
        and holdout_result.trades >= 30
        and holdout_result.pnl > 0
        and holdout_result.roi > 0
    )
    selected = EmpiricalStrategyConfig(
        enabled=validated,
        minimum_probability_residual=selected.minimum_probability_residual,
        minimum_reversion_probability=selected.minimum_reversion_probability,
        minimum_reversion_probability_margin=(
            selected.minimum_reversion_probability_margin
        ),
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    reaction_model.save_model(REACTION_MODEL_PATH)
    reversion_model.save_model(REVERSION_MODEL_PATH)
    CONFIG_PATH.write_text(json.dumps(asdict(selected), indent=2) + "\n")
    grid.to_csv(STUDY_DIR / "tuning_grid.csv", index=False)
    accepted_trade_rows(holdout, holdout_result).to_csv(
        STUDY_DIR / "holdout_trades.csv", index=False
    )

    reaction_summary = {
        "data_scope": "2026 only; no 2025 data used",
        "fit_end_exclusive": str(REACTION_FIT_END),
        "features": list(REACTION_FEATURES),
        "target": "five-second batting-team log-odds market move after a hit",
        "fit": reaction_metrics(
            reaction_fit, reaction_model.predict(reaction_feature_frame(reaction_fit))
        ),
        "classifier_fit_period": reaction_metrics(
            events[(event_dates >= REACTION_FIT_END) & (event_dates < CLASSIFIER_FIT_END)],
            reaction_model.predict(reaction_feature_frame(events[
                (event_dates >= REACTION_FIT_END) & (event_dates < CLASSIFIER_FIT_END)
            ])),
        ),
        "tuning_period": reaction_metrics(
            events[(event_dates >= CLASSIFIER_FIT_END) & (event_dates < TUNING_END)],
            reaction_model.predict(reaction_feature_frame(events[
                (event_dates >= CLASSIFIER_FIT_END) & (event_dates < TUNING_END)
            ])),
        ),
        "holdout_period": reaction_metrics(
            events[(event_dates >= TUNING_END) & (event_dates < HOLDOUT_END)],
            reaction_model.predict(reaction_feature_frame(events[
                (event_dates >= TUNING_END) & (event_dates < HOLDOUT_END)
            ])),
        ),
    }
    (STUDY_DIR / "reaction_model_summary.json").write_text(
        json.dumps(reaction_summary, indent=2) + "\n"
    )
    training_summary = {
        "classifier_fit_dates": "2026-06-01 through 2026-06-16",
        "tuning_dates": "2026-06-17 through 2026-06-27",
        "outer_holdout_dates": "2026-06-28 through 2026-07-10",
        "features": list(REVERSION_FEATURES),
        "label": (
            "observed executable cross of empirical expected price before "
            "settlement with positive round-trip PnL after fees"
        ),
        "classifier_fit": classifier_metrics(classifier_fit),
        "tuning": classifier_metrics(tuning),
        "holdout": classifier_metrics(holdout),
        "selection_rule": (
            "maximum tuning ROI among fee-adjusted probability margins with "
            ">=20 trades; if none qualify, retain the most permissive margin "
            "and disable the strategy"
        ),
        "minimum_trade_requirement_met": bool(not eligible.empty),
        "selected_config": asdict(selected),
        "selected_tuning_result": selection.to_dict(),
        "selected_holdout_result": result_dict(holdout_result),
        "execution_model": (
            "strictly later compatible taker-side trade with sufficient reported size"
        ),
        "time_based_exit": False,
    }
    (STUDY_DIR / "training_summary.json").write_text(
        json.dumps(training_summary, indent=2) + "\n"
    )

    print(
        f"Reaction fit: {len(reaction_fit):,} events; candidates: "
        f"{len(candidates):,}"
    )
    print(
        "Classifier fit/tune/holdout candidates: "
        f"{len(classifier_fit):,}/{len(tuning):,}/{len(holdout):,}"
    )
    print("Top tuning configurations:")
    print(grid.head(12).to_string(index=False, formatters={
        "roi": "{:.2%}".format,
        "pnl": "${:,.2f}".format,
        "fees": "${:,.2f}".format,
    }))
    print("Holdout:", json.dumps(result_dict(holdout_result), indent=2))
    print(f"Saved config to {CONFIG_PATH} (enabled={selected.enabled})")


if __name__ == "__main__":
    main()
