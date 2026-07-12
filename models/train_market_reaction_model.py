"""
train_market_reaction_model.py

CHANGE LOG (fix for stacked-model leakage):

Previously, fair_prob for the reaction model's TRAINING set was generated
by calling the fully-trained state_model.predict_proba() directly on the
exact same rows it was trained on. A gradient-boosted model fits its own
training data more closely than genuinely unseen data, so fair_prob (and
therefore market_error = kalshi_price - fair_prob) was optimistically
biased on the training set in a way it never is in production, where the
state model has never seen the pitch it's scoring. The reaction model
would learn to interpret market_error using a state model that's
artificially "too good" -- a relationship that doesn't hold once the
*actually* deployed state model (only as accurate as its true
generalization performance) is producing fair_prob at inference time.

Fixed with out-of-fold (cross-fitted) fair_prob: the training set is split
into chronological folds via TimeSeriesSplit, a fresh state-model CLONE is
trained on each fold's past-only data and used to predict the
next fold, and those held-out predictions are stitched together into a
genuinely out-of-sample fair_prob column. TimeSeriesSplit (not plain
KFold) is used deliberately -- plain KFold would let a fold's model see
chronologically FUTURE rows when predicting an earlier fold, which is
its own, subtler form of the same leak this is meant to fix.

The EARLIEST chunk of the training set (before any fold has a
"predecessor" fold to train on) has no valid out-of-fold prediction and
is dropped from reaction-model training rather than filled with a biased
guess.

Note: the state_model loaded from disk (trained on 100% of the training
set) is still exactly what's used for real inference in
evaluate_strategy.py, scored on the genuinely held-out test set -- that
path was already correct and is unchanged. This fix only concerns how the
REACTION MODEL's TRAINING features are generated.
"""

import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

TRAIN_DATA = "data/processed/train/training_dataset.parquet"
STATE_MODEL_PATH = "models/baseball_state_model/baseball_model.cbm"
OUTPUT_DIR = "models/market_reaction_model/reaction_model.cbm"

STATE_DROP_COLS = [
    "volume",
    "open_interest",
    "seconds_since_price_update",
    "kalshi_price",
    "spread",
    "home_win",
]

# Same hyperparameters as the production state model (train_baseball_model.py),
# so the out-of-fold fair_prob distribution resembles what the real,
# fully-trained state model actually produces.
STATE_MODEL_PARAMS = dict(
    iterations=1000,
    learning_rate=0.03,
    depth=8,
    loss_function="Logloss",
    eval_metric="Logloss",
    verbose=False,
)

N_OOF_SPLITS = 5


def generate_oof_fair_prob(df: pd.DataFrame, target: str = "home_win",
                            n_splits: int = N_OOF_SPLITS) -> pd.Series:
    """
    Returns a Series of out-of-fold fair_prob, aligned to df's index.
    Rows in the earliest fold (no predecessor fold to train on) get NaN
    and should be dropped by the caller before training the reaction model.
    """
    state_features = [c for c in df.columns if c not in STATE_DROP_COLS]

    oof = pd.Series(np.nan, index=df.index, dtype="float64")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    for fold, (train_idx, holdout_idx) in enumerate(tscv.split(df), start=1):
        print(f"  OOF fold {fold}/{n_splits}: training on {len(train_idx):,} rows, "
              f"predicting {len(holdout_idx):,} held-out rows...")

        fold_model = CatBoostClassifier(**STATE_MODEL_PARAMS)
        fold_model.fit(
            df.iloc[train_idx][state_features],
            df.iloc[train_idx][target],
        )

        preds = fold_model.predict_proba(df.iloc[holdout_idx][state_features])[:, 1]
        oof.iloc[holdout_idx] = preds

    n_missing = oof.isna().sum()
    print(f"  OOF complete: {oof.notna().sum():,} / {len(df):,} rows have a valid "
          f"out-of-fold fair_prob ({n_missing:,} rows in the earliest fold have "
          f"none and will be dropped).")

    return oof


def main():
    print(f"Loading data from {TRAIN_DATA}...")
    df = pd.read_parquet(TRAIN_DATA)

    print(f"\nGenerating OUT-OF-FOLD fair_prob ({N_OOF_SPLITS}-fold "
          f"TimeSeriesSplit, retrains a state-model clone per fold)...")
    df["fair_prob"] = generate_oof_fair_prob(df)

    before = len(df)
    df = df[df["fair_prob"].notna()].copy()
    print(f"  Dropped {before - len(df):,} rows with no OOF fair_prob "
          f"(earliest fold, no predecessor to train from).")

    df["market_error"] = df["kalshi_price"] - df["fair_prob"]

    TARGET = "home_win"

    FEATURES = [
        "fair_prob",
        "market_error",
        "volume",
        "spread",
        "open_interest",
        "seconds_since_price_update",
        "inning",
        "outs_when_up",
        "score_diff",
        "runner_on_first",
        "runner_on_second",
        "runner_on_third",
    ]
    FEATURES = [f for f in FEATURES if f in df.columns]

    X = df[FEATURES]
    y = df[TARGET]

    # Chronological time split (df is already sorted upstream by
    # build_kalshi_join.py, and OOF generation above preserved row order).
    split = int(len(df) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    print("\nTraining Market Reaction Model...")
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        loss_function="Logloss",
        eval_metric="Logloss",
        verbose=100,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=100,
    )

    pred_proba = model.predict_proba(X_test)[:, 1]
    pred_class = (pred_proba > 0.5).astype(int)

    print("\n--- Market Reaction Model Performance ---")
    print(f"Log loss: {log_loss(y_test, pred_proba):.4f}")
    print(f"Accuracy: {accuracy_score(y_test, pred_class):.1%}")
    print(f"ROC AUC:  {roc_auc_score(y_test, pred_proba):.4f}")

    os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)
    model.save_model(OUTPUT_DIR)
    print(f"\nSaved Market Reaction Model to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()