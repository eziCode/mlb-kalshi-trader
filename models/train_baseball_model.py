import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, accuracy_score

DATA = "data/processed/train/training_dataset.parquet"
OUTPUT_DIR = "models/baseball_state_model/baseball_model.cbm"
df = pd.read_parquet(DATA)

# The target variable `home_win` is already present in the dataset,
# calculated in the data pipeline.

# remove market information
drop_cols = [
    "volume",
    "open_interest",
    "seconds_since_price_update",
    "kalshi_price",
    "spread"
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# target
TARGET = "home_win"

if TARGET not in df.columns:
    raise KeyError(
        f"'{TARGET}' is missing! We must update the data folder scripts to include "
        f"the game outcome before we can train the model."
    )

FEATURES = [
    c for c in df.columns
    if c != TARGET
]

X = df[FEATURES]
y = df[TARGET]

# time split
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.03,
    depth=8,
    loss_function="Logloss",
    eval_metric="Logloss",
    verbose=100
)

model.fit(
    X_train,
    y_train,
    eval_set=(
        X_test,
        y_test
    ),
    early_stopping_rounds=100
)

pred = model.predict_proba(X_test)[:, 1]

print("Log loss:", log_loss(y_test, pred))

import os
os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)
model.save_model(OUTPUT_DIR)
