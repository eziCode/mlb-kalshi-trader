import os
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score

TRAIN_DATA = "data/processed/train/training_dataset.parquet"
STATE_MODEL_PATH = "models/baseball_state_model/baseball_model.cbm"
OUTPUT_DIR = "models/market_reaction_model/reaction_model.cbm"

def generate_reaction_features(df, state_model):
    """Uses the State Model to generate fair probabilities in-memory."""
    print("Generating fair probabilities from State Model...")
    
    # 1. Reconstruct exactly what the state model saw
    state_drop_cols = [
        "volume",
        "open_interest",
        "seconds_since_price_update",
        "kalshi_price",
        "spread",
        "home_win"
    ]
    state_features = [c for c in df.columns if c not in state_drop_cols]
    
    # 2. Predict Fair Probability
    df["fair_prob"] = state_model.predict_proba(df[state_features])[:, 1]
    
    # 3. Calculate Market Error (Kalshi's over/under reaction)
    df["market_error"] = df["kalshi_price"] - df["fair_prob"]
    
    return df

def main():
    print(f"Loading data from {TRAIN_DATA}...")
    df = pd.read_parquet(TRAIN_DATA)
    
    print(f"Loading Baseball State Model from {STATE_MODEL_PATH}...")
    state_model = CatBoostClassifier()
    state_model.load_model(STATE_MODEL_PATH)
    
    # Preprocess in-memory
    df = generate_reaction_features(df, state_model)
    
    # The target is STILL whether the home team won, because the Reaction Model 
    # needs to figure out if following the market (or fading the market) is the 
    # better path to the actual outcome.
    TARGET = "home_win"
    
    # Features for the Reaction Model
    # Now we explicitly give it the market signals + the fair_prob
    FEATURES = [
        "fair_prob",
        "market_error",
        "volume",
        "spread",
        "open_interest",
        "seconds_since_price_update",
        # Including a few state variables gives the model context on *when* the 
        # market tends to overreact (e.g. late innings, close games).
        "inning",
        "outs_when_up",
        "score_diff",
        "runner_on_first",
        "runner_on_second",
        "runner_on_third"
    ]
    
    # Ensure all features exist
    FEATURES = [f for f in FEATURES if f in df.columns]
    
    X = df[FEATURES]
    y = df[TARGET]
    
    # Chronological time split
    split = int(len(df) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    print("\nTraining Market Reaction Model...")
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6, # Slightly shallower depth is often better for meta-models
        loss_function="Logloss",
        eval_metric="Logloss",
        verbose=100
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=100
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
