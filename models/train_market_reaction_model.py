"""Train local win expectancy and market-reaction models without row leakage."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import log_loss


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlb_kalshi.strategy import (  # noqa: E402
    CONFIG,
    REACTION_FEATURES,
    STATE_FEATURES,
    add_reaction_features,
    batting_win_label,
    predict_home_probability,
    reaction_feature_frame,
    state_feature_frame,
)


TRAIN_DATA = PROJECT_ROOT / "data/processed/train/training_dataset.parquet"
MODEL_DIR = PROJECT_ROOT / "models/market_reaction_model"
STATE_MODEL_PATH = MODEL_DIR / "local_win_expectancy.cbm"
REACTION_MODEL_PATH = MODEL_DIR / "reaction_model.cbm"


def weights(frame: pd.DataFrame) -> np.ndarray:
    rows_per_game = frame.groupby("game_pk")["game_pk"].transform("size")
    return (1.0 / rows_per_game).to_numpy()


def state_pool(frame: pd.DataFrame, label: bool = True) -> Pool:
    kwargs = {"data": state_feature_frame(frame)}
    if label:
        kwargs.update(label=batting_win_label(frame), weight=weights(frame))
    return Pool(**kwargs)


def reaction_pool(frame: pd.DataFrame, label: bool = True) -> Pool:
    fair = frame["fair_prob"].clip(1e-4, 1 - 1e-4)
    kwargs = {
        "data": reaction_feature_frame(frame),
        "baseline": np.log(fair / (1 - fair)),
    }
    if label:
        kwargs.update(label=frame["home_win"], weight=weights(frame))
    return Pool(**kwargs)


def params(iterations: int, *, state: bool = False) -> dict:
    result = {
        "iterations": iterations,
        "learning_rate": 0.03,
        "depth": 4,
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "l2_leaf_reg": 15.0,
        "random_seed": 42,
        "allow_writing_files": False,
        "verbose": 100,
    }
    if state:
        result["monotone_constraints"] = {
            "pregame_batting_prob": 1,
            "outs_when_up": -1,
            "batting_score_diff": 1,
            "balls": 1,
            "strikes": -1,
            "runner_on_first": 1,
            "runner_on_second": 1,
            "runner_on_third": 1,
        }
    return result


def date_partition(frame: pd.DataFrame, first_fraction: float, second_fraction: float):
    dates = pd.to_datetime(frame["game_date"]).dt.date
    unique = sorted(dates.unique())
    first_date = unique[int(len(unique) * first_fraction)]
    second_date = unique[int(len(unique) * second_fraction)]
    first = frame[dates < first_date].copy()
    second = frame[(dates >= first_date) & (dates < second_date)].copy()
    third = frame[dates >= second_date].copy()
    return first, second, third


def best_iterations(model: CatBoostClassifier) -> int:
    best = model.get_best_iteration()
    return int(best + 1 if best is not None and best >= 0 else model.tree_count_)


def main() -> None:
    raw = pd.read_parquet(TRAIN_DATA).sort_values("decision_time")
    state_fit, state_tune, reaction_dates = date_partition(raw, 0.60, 0.75)
    if min(len(state_fit), len(state_tune), len(reaction_dates)) == 0:
        raise RuntimeError("Not enough chronological dates for model training")

    print(
        f"State fit/tune/reaction rows: {len(state_fit):,} / "
        f"{len(state_tune):,} / {len(reaction_dates):,}"
    )
    provisional_state = CatBoostClassifier(**params(1000, state=True))
    provisional_state.fit(
        state_pool(state_fit),
        eval_set=state_pool(state_tune),
        early_stopping_rounds=100,
    )
    state_iterations = best_iterations(provisional_state)

    # Reaction labels see only local-state predictions from later games that
    # were not used to fit or early-stop the state model.
    fair_oos = predict_home_probability(provisional_state, reaction_dates)
    reaction_dates = add_reaction_features(reaction_dates, fair_oos)
    reaction_fit, reaction_tune, _ = date_partition(
        reaction_dates, 0.70, 0.85
    )
    # Use the final chronological slice as part of tuning because the outer
    # test dataset remains completely untouched.
    tune_dates = pd.to_datetime(reaction_dates["game_date"]).dt.date
    cutoff = sorted(tune_dates.unique())[int(len(tune_dates.unique()) * 0.70)]
    reaction_fit = reaction_dates[tune_dates < cutoff].copy()
    reaction_tune = reaction_dates[tune_dates >= cutoff].copy()

    provisional_reaction = CatBoostClassifier(**params(1000))
    provisional_reaction.fit(
        reaction_pool(reaction_fit),
        eval_set=reaction_pool(reaction_tune),
        early_stopping_rounds=100,
    )
    reaction_iterations = best_iterations(provisional_reaction)
    prediction = provisional_reaction.predict_proba(
        reaction_pool(reaction_tune, label=False)
    )[:, 1]
    print(f"Reaction chronological tune log loss: {log_loss(reaction_tune.home_win, prediction):.4f}")

    # Refit deployment models using the selected tree counts. The reaction
    # model retains genuinely out-of-state-model-sample fair probabilities.
    final_state = CatBoostClassifier(**params(state_iterations, state=True))
    final_state.fit(state_pool(raw))
    final_reaction = CatBoostClassifier(**params(reaction_iterations))
    final_reaction.fit(reaction_pool(reaction_dates))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    final_state.save_model(STATE_MODEL_PATH)
    final_reaction.save_model(REACTION_MODEL_PATH)
    metadata = {
        "state_features": list(STATE_FEATURES),
        "reaction_features": list(REACTION_FEATURES),
        "edge_threshold": CONFIG.edge_threshold,
        "state_iterations": state_iterations,
        "reaction_iterations": reaction_iterations,
        "scaled_features": [],
        "state_probability_orientation": "batting_team_converted_to_home",
        "monotone_constraints": params(1, state=True)["monotone_constraints"],
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved models and metadata to {MODEL_DIR}")


if __name__ == "__main__":
    main()
