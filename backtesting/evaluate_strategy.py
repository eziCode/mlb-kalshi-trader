import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

TEST_DATA = "data/processed/test/test_dataset.parquet"
STATE_MODEL_PATH = "models/baseball_state_model/baseball_model.cbm"
REACTION_MODEL_PATH = "models/market_reaction_model/reaction_model.cbm"

EDGE_THRESHOLD = 0.05  # Requires a 5% difference between our model and Kalshi to bet
BET_SIZE = 10.0        # Risk $10 per bet

def main():
    print(f"Loading test dataset from {TEST_DATA}...")
    df = pd.read_parquet(TEST_DATA)
    
    print("Loading models...")
    state_model = CatBoostClassifier()
    state_model.load_model(STATE_MODEL_PATH)
    
    reaction_model = CatBoostClassifier()
    reaction_model.load_model(REACTION_MODEL_PATH)
    
    # ---------------------------------------------------------
    # 1. State Model Inference (Fair Probabilities)
    # ---------------------------------------------------------
    state_drop_cols = [
        "volume", "open_interest", "seconds_since_price_update",
        "kalshi_price", "spread", "home_win"
    ]
    state_features = [c for c in df.columns if c not in state_drop_cols]
    
    print("Computing fair probabilities...")
    df["fair_prob"] = state_model.predict_proba(df[state_features])[:, 1]
    df["market_error"] = df["kalshi_price"] - df["fair_prob"]
    
    # ---------------------------------------------------------
    # 2. Reaction Model Inference (Final Probabilities)
    # ---------------------------------------------------------
    reaction_features = [
        "fair_prob", "market_error", "volume", "spread", "open_interest",
        "seconds_since_price_update", "inning", "outs_when_up", "score_diff",
        "runner_on_first", "runner_on_second", "runner_on_third"
    ]
    # Keep only features that exist in the dataframe
    reaction_features = [f for f in reaction_features if f in df.columns]
    
    print("Computing final model probabilities...")
    df["final_prob"] = reaction_model.predict_proba(df[reaction_features])[:, 1]
    
    # ---------------------------------------------------------
    # 3. Simulate Trading
    # ---------------------------------------------------------
    print("Simulating trading strategy...")
    
    # Edge = How much higher our model's probability is compared to the market
    df["edge"] = df["final_prob"] - df["kalshi_price"]
    
    # We place a "YES" bet if our model thinks the home team is highly undervalued
    df["bet_yes"] = df["edge"] > EDGE_THRESHOLD
    
    # We place a "NO" bet if our model thinks the home team is highly overvalued
    df["bet_no"] = df["edge"] < -EDGE_THRESHOLD
    
    # PnL for 1 contract (assuming Kalshi payouts are $1.00)
    df["pnl_per_contract"] = 0.0
    
    # YES bets payout: 
    # If they win, we get $1.00 back (Profit = 1 - cost). If they lose, we lose the cost.
    df.loc[df["bet_yes"] & (df["home_win"] == 1), "pnl_per_contract"] = 1.0 - df["kalshi_price"]
    df.loc[df["bet_yes"] & (df["home_win"] == 0), "pnl_per_contract"] = -df["kalshi_price"]
    
    # NO bets payout:
    # Cost is (1 - kalshi_price). If home loses (NO wins), we get $1.00 back.
    df.loc[df["bet_no"] & (df["home_win"] == 0), "pnl_per_contract"] = df["kalshi_price"]
    df.loc[df["bet_no"] & (df["home_win"] == 1), "pnl_per_contract"] = -(1.0 - df["kalshi_price"])
    
    # Scale PnL based on a flat $10 bet
    df["contracts_bought"] = 0.0
    # For YES bets, $10 buys us (10 / price) contracts
    df.loc[df["bet_yes"], "contracts_bought"] = BET_SIZE / df["kalshi_price"]
    # For NO bets, $10 buys us (10 / (1 - price)) contracts
    df.loc[df["bet_no"], "contracts_bought"] = BET_SIZE / (1.0 - df["kalshi_price"])
    
    # Multiply contract PnL by the number of contracts we bought
    df["trade_pnl"] = df["pnl_per_contract"] * df["contracts_bought"]
    
    # ---------------------------------------------------------
    # 4. Results & Metrics
    # ---------------------------------------------------------
    total_bets = df["bet_yes"].sum() + df["bet_no"].sum()
    yes_bets = df["bet_yes"].sum()
    no_bets = df["bet_no"].sum()
    
    if total_bets == 0:
        print(f"\nNo bets placed at edge threshold {EDGE_THRESHOLD*100}%. Market is perfectly priced.")
        return
        
    wins = ((df["bet_yes"] & (df["home_win"] == 1)) | (df["bet_no"] & (df["home_win"] == 0))).sum()
    win_rate = wins / total_bets
    total_pnl = df["trade_pnl"].sum()
    total_risked = total_bets * BET_SIZE
    roi = total_pnl / total_risked if total_risked > 0 else 0
    
    print("\n" + "="*50)
    print("BACKTEST RESULTS (Test Set: June 28 - July 10)")
    print("="*50)
    print(f"Total Pitches Evaluated: {len(df):,}")
    print(f"Total Bets Placed:       {total_bets:,}")
    print(f"  - YES Bets:            {yes_bets:,}")
    print(f"  - NO Bets:             {no_bets:,}")
    print(f"Winning Bets:            {wins:,}")
    print(f"Win Rate:                {win_rate:.2%}")
    print("-" * 50)
    print(f"Total Capital Risked:    ${total_risked:,.2f}")
    print(f"Net Profit (PnL):        ${total_pnl:,.2f}")
    print(f"ROI:                     {roi:.2%}")
    print("="*50)

if __name__ == "__main__":
    main()
