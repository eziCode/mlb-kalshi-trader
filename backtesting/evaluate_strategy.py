import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool

TEST_DATA = "data/processed/test/test_dataset.parquet"
REACTION_MODEL_PATH = "models/market_reaction_model/reaction_model.cbm"

EDGE_THRESHOLD = 0.15  # Requires a 15% difference between our model and Kalshi to bet
BET_SIZE = 10.0        # Risk $10 per bet

def main():
    print(f"Loading test dataset from {TEST_DATA}...")
    df = pd.read_parquet(TEST_DATA)
    
    print("Loading model...")
    reaction_model = CatBoostClassifier()
    reaction_model.load_model(REACTION_MODEL_PATH)
    
    # ---------------------------------------------------------
    # 1. State Model Inference (Fair Probabilities)
    # ---------------------------------------------------------
    print("Using Log5 anchored MLB home_win_exp as fair probabilities...")
    we = df["home_win_exp"].clip(0.001, 0.999)
    p = df["pregame_prob"].clip(0.001, 0.999)
    we0 = df["pregame_home_win_exp"].clip(0.001, 0.999)
    
    odds_we = we / (1 - we)
    odds_p = p / (1 - p)
    odds_we0 = we0 / (1 - we0)
    
    odds_adj = odds_we * (odds_p / odds_we0)
    df["fair_prob"] = odds_adj / (1 + odds_adj)
    df["market_error"] = df["kalshi_price"] - df["fair_prob"]
    
    # ---------------------------------------------------------
    # 2. Reaction Model Inference (Final Probabilities)
    # ---------------------------------------------------------
    reaction_features = [
        "market_error", "kalshi_price", "pregame_prob", "volume", "spread",
        "seconds_since_price_update", "inning"
    ]
    # Keep only features that exist in the dataframe
    reaction_features = [f for f in reaction_features if f in df.columns]
    
    print("Computing final model probabilities using Reaction Model...")
    # Calculate baseline in log-odds for CatBoost
    fp = df["fair_prob"].clip(0.0001, 0.9999)
    baseline = np.log(fp / (1 - fp))
    
    eval_pool = Pool(data=df[reaction_features], baseline=baseline)
    df["final_prob"] = reaction_model.predict_proba(eval_pool)[:, 1]
    
    # ---------------------------------------------------------
    # 3. Simulate Stateful Portfolio Trading
    # ---------------------------------------------------------
    print("Simulating stateful trading strategy...")
    
    # Pre-calculate prices and edges
    df["ask_price"] = (df["kalshi_price"] + (df["spread"] / 2.0)).clip(0.01, 0.99)
    df["bid_price"] = (df["kalshi_price"] - (df["spread"] / 2.0)).clip(0.01, 0.99)
    df["edge_yes"] = df["final_prob"] - df["ask_price"]
    df["edge_no"] = df["bid_price"] - df["final_prob"]

    total_bets_placed = 0
    yes_bets = 0
    no_bets = 0
    early_exits = 0
    total_capital_risked = 0.0
    cash = 0.0
    
    # We simulate chronologically per game
    for game_pk, game_df in df.groupby("game_pk"):
        position = 0.0  # Positive = YES contracts, Negative = NO contracts
        
        for row in game_df.itertuples():
            ask = row.ask_price
            bid = row.bid_price
            f_prob = row.final_prob
            
            # --- 1. Exiting existing positions ---
            if position > 0:
                # We hold YES. If market is willing to pay more than true worth, sell!
                # Holding YES is -EV if final_prob < bid
                if f_prob < bid:
                    cash += position * bid
                    position = 0.0
                    early_exits += 1
            
            elif position < 0:
                # We hold NO. We bought at (1 - entry_bid). To exit, we buy YES at ask.
                # Holding NO is -EV if final_prob > ask
                if f_prob > ask:
                    cash += abs(position) * (1 - ask)
                    position = 0.0
                    early_exits += 1
                    
            # --- 2. Opening new positions ---
            if position == 0.0:
                if row.edge_yes > EDGE_THRESHOLD:
                    # Buy YES
                    contracts = BET_SIZE / ask
                    position = contracts
                    cash -= BET_SIZE
                    total_capital_risked += BET_SIZE
                    total_bets_placed += 1
                    yes_bets += 1
                elif row.edge_no > EDGE_THRESHOLD:
                    # Buy NO
                    contracts = BET_SIZE / (1 - bid)
                    position = -contracts
                    cash -= BET_SIZE
                    total_capital_risked += BET_SIZE
                    total_bets_placed += 1
                    no_bets += 1

        # --- 3. End of game settlement ---
        # Get the final result for this game from the last row
        hw = game_df.iloc[-1]["home_win"]
        if position > 0:
            if hw == 1:
                cash += position * 1.0
        elif position < 0:
            if hw == 0:
                cash += abs(position) * 1.0

    # ---------------------------------------------------------
    # 4. Results & Metrics
    # ---------------------------------------------------------
    if total_bets_placed == 0:
        print(f"\nNo bets placed at edge threshold {EDGE_THRESHOLD*100}%.")
        return
        
    roi = cash / total_capital_risked if total_capital_risked > 0 else 0
    
    print("\n" + "="*50)
    print("BACKTEST RESULTS (Test Set: June 28 - July 10)")
    print("="*50)
    print(f"Total Pitches Evaluated: {len(df):,}")
    print(f"Total Trades Opened:     {total_bets_placed:,}")
    print(f"  - YES Trades:          {yes_bets:,}")
    print(f"  - NO Trades:           {no_bets:,}")
    print(f"Positions Traded Out:    {early_exits:,} (Hedging / Early Profit/Loss)")
    print("-" * 50)
    print(f"Total Capital Risked:    ${total_capital_risked:,.2f}")
    print(f"Net Profit (PnL):        ${cash:,.2f}")
    print(f"ROI:                     {roi:.2%}")
    print("="*50)
    
    # We no longer generate an error analysis table because trades span multiple rows 
    # and aren't simple static 1-row bets anymore.
if __name__ == "__main__":
    main()
