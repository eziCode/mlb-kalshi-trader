"""
1. convert categorical variables
currently, inning_topbot and runner_state are strings
inning_topbot -> top = 0, bot = 1

runner_state -> split it, create runner_on_first, runner_on_second, runner_on_third

2. normalize continuous variables

3. add market signals
spread = yes_ask_close - yes_bid_close

"""

import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

TRAIN_PATH = Path("data/processed/train/training_dataset.parquet")
TEST_PATH = Path("data/processed/test/test_dataset.parquet")
SCALER_PATH = Path("data/processed/train/scaler.joblib")

def preprocess(df: pd.DataFrame, scaler: StandardScaler = None, is_train: bool = True):
    df = df.copy()
    
    # -------------------------------------------------------------------------
    # 1. Convert categorical variables
    # -------------------------------------------------------------------------
    if "inning_topbot" in df.columns:
        df["inning_topbot"] = df["inning_topbot"].map({"Top": 0, "Bot": 1})
    
    if "runner_state" in df.columns:
        # runner_state is a string like "000", "101", etc.
        df["runner_on_first"] = df["runner_state"].str[0].astype(int)
        df["runner_on_second"] = df["runner_state"].str[1].astype(int)
        df["runner_on_third"] = df["runner_state"].str[2].astype(int)
        df = df.drop(columns=["runner_state"])

    # -------------------------------------------------------------------------
    # 3. Add market signals
    # -------------------------------------------------------------------------
    if "yes_ask_close" in df.columns and "yes_bid_close" in df.columns:
        df["spread"] = df["yes_ask_close"] - df["yes_bid_close"]
        # Now that we have kalshi_price (midpoint) and spread, we can drop raw bid/ask
        df = df.drop(columns=["yes_ask_close", "yes_bid_close"])
        
    # We can also drop the OHLC columns as they are intra-minute volatility noise
    ohlc_cols = ["price_open", "price_high", "price_low"]
    df = df.drop(columns=[c for c in ohlc_cols if c in df.columns])

    # -------------------------------------------------------------------------
    # 2. Normalize continuous variables
    # -------------------------------------------------------------------------
    # We leave inning, outs_when_up, balls, strikes as is (ordinal).
    # We leave kalshi_price and spread as is (probabilities/cents 0-1).
    continuous_cols = [
        "hitter_form_7d", 
        "hitter_form_21d", 
        "pitch_number", 
        "delta_home_win_exp", 
        "volume", 
        "open_interest", 
        "seconds_since_price_update",
        "pitcher_game_pitch_count",
        "hist_entry_score_diff",
    ]
    
    cols_to_scale = [c for c in continuous_cols if c in df.columns]
    
    # Fill missing values in continuous columns with 0 before scaling
    df[cols_to_scale] = df[cols_to_scale].fillna(0)
    
    if is_train:
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    else:
        if scaler is None:
            raise ValueError("A fitted scaler must be provided when processing test data.")
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])
        
    return df, scaler


def main():
    print(f"Loading {TRAIN_PATH} ...")
    train_df = pd.read_parquet(TRAIN_PATH)
    
    print(f"Loading {TEST_PATH} ...")
    test_df = pd.read_parquet(TEST_PATH)
    
    print("\nPreprocessing training data...")
    train_df, scaler = preprocess(train_df, is_train=True)
    
    print("Preprocessing test data...")
    test_df, _ = preprocess(test_df, scaler=scaler, is_train=False)
    
    print(f"\nSaving fitted scaler -> {SCALER_PATH}")
    joblib.dump(scaler, SCALER_PATH)
    
    print(f"Overwriting {TRAIN_PATH} ...")
    train_df.to_parquet(TRAIN_PATH, index=False)
    
    print(f"Overwriting {TEST_PATH} ...")
    test_df.to_parquet(TEST_PATH, index=False)
    
    print("\nDone! Final schema:")
    print(train_df.dtypes.to_string())

if __name__ == "__main__":
    main()