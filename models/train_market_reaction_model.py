"""
train_market_reaction_model.py
"""

import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score

TRAIN_DATA = "data/processed/train/training_dataset.parquet"
OUTPUT_DIR = "models/market_reaction_model/reaction_model.cbm"

def main():
    print(f"Loading data from {TRAIN_DATA}...")
    df = pd.read_parquet(TRAIN_DATA)
    
    # Apply Log5 Anchor to fair_prob
    # Odds(Adj) = Odds(WE) * (Odds(P) / Odds(WE0))
    we = df["home_win_exp"].clip(0.001, 0.999)
    p = df["pregame_prob"].clip(0.001, 0.999)
    we0 = df["pregame_home_win_exp"].clip(0.001, 0.999)
    
    odds_we = we / (1 - we)
    odds_p = p / (1 - p)
    odds_we0 = we0 / (1 - we0)
    
    odds_adj = odds_we * (odds_p / odds_we0)
    df["fair_prob"] = odds_adj / (1 + odds_adj)
    
    df["market_error"] = df["kalshi_price"] - df["fair_prob"]

    TARGET = "home_win"

    FEATURES = [
        "market_error",
        "kalshi_price",
        "pregame_prob",
        "volume",
        "spread",
        "seconds_since_price_update",
        "inning",
    ]
    FEATURES = [f for f in FEATURES if f in df.columns]

    X = df[FEATURES]
    y = df[TARGET]

    # Calculate baseline in log-odds for CatBoost
    fp = df["fair_prob"].clip(0.0001, 0.9999)
    baseline = np.log(fp / (1 - fp))

    # Chronological time split
    split = int(len(df) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]
    
    baseline_train = baseline.iloc[:split]
    baseline_test = baseline.iloc[split:]

    train_pool = Pool(data=X_train, label=y_train, baseline=baseline_train)
    test_pool = Pool(data=X_test, label=y_test, baseline=baseline_test)

    print("\nTraining Market Reaction Model...")
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=3,
        loss_function="Logloss",
        eval_metric="Logloss",
        verbose=100,
    )

    model.fit(
        train_pool,
        eval_set=test_pool,
        early_stopping_rounds=100,
    )

    pred_proba = model.predict_proba(test_pool)[:, 1]
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