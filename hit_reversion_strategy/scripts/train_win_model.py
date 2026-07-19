"""Retrain the local win-expectancy model used by event targets."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trade_tape_strategy.strategy import state_feature_frame  # noqa: E402


TRAIN_DATA = REPOSITORY_ROOT / "data/hit_reversion/training_dataset.parquet"
MODEL_PATH = PROJECT_ROOT / "models/local_win_expectancy.cbm"


def weights(frame: pd.DataFrame) -> np.ndarray:
    rows_per_game = frame.groupby("game_pk")["game_pk"].transform("size")
    return (1.0 / rows_per_game).to_numpy()


def state_pool(frame: pd.DataFrame) -> Pool:
    return Pool(
        data=state_feature_frame(frame),
        label=frame["home_win"],
        weight=weights(frame),
    )


def parameters(iterations: int) -> dict:
    return {
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


def main() -> None:
    raw = pd.read_parquet(TRAIN_DATA).sort_values("decision_time")
    dates = pd.to_datetime(raw["game_date"]).dt.date
    unique_dates = sorted(dates.unique())
    tune_start = unique_dates[int(len(unique_dates) * 0.60)]
    reaction_start = unique_dates[int(len(unique_dates) * 0.75)]
    fit = raw[dates < tune_start].copy()
    tune = raw[(dates >= tune_start) & (dates < reaction_start)].copy()
    if fit.empty or tune.empty:
        raise RuntimeError("Not enough chronological dates for model training")

    provisional = CatBoostClassifier(**parameters(1000))
    provisional.fit(
        state_pool(fit),
        eval_set=state_pool(tune),
        early_stopping_rounds=100,
    )
    best = provisional.get_best_iteration()
    iterations = int(
        best + 1 if best is not None and best >= 0 else provisional.tree_count_
    )
    final = CatBoostClassifier(**parameters(iterations))
    final.fit(state_pool(raw))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.save_model(MODEL_PATH)
    print(f"Saved {MODEL_PATH} with {iterations} trees")


if __name__ == "__main__":
    main()
