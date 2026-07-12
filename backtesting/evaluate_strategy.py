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
    # 3. Simulate Trading
    # ---------------------------------------------------------
    print("Simulating trading strategy...")
    
    # Calculate effective Ask and Bid prices from kalshi_price and spread
    # (Assuming kalshi_price is the mid-price)
    df["ask_price"] = (df["kalshi_price"] + (df["spread"] / 2.0)).clip(0.01, 0.99)
    df["bid_price"] = (df["kalshi_price"] - (df["spread"] / 2.0)).clip(0.01, 0.99)
    
    # YES Bet: We buy at the ASK. Win prob = final_prob.
    # Expected value edge = final_prob - ask_price
    df["edge_yes"] = df["final_prob"] - df["ask_price"]
    
    # NO Bet: We sell at the BID (equivalent to buying NO at 1 - bid).
    # Win prob = 1 - final_prob. Cost = 1 - bid_price.
    # Expected value edge = (1 - final_prob) - (1 - bid_price) = bid_price - final_prob
    df["edge_no"] = df["bid_price"] - df["final_prob"]
    
    # We place bets only if the edge crosses our threshold
    df["bet_yes"] = df["edge_yes"] > EDGE_THRESHOLD
    df["bet_no"] = df["edge_no"] > EDGE_THRESHOLD
    
    # PnL for 1 contract (assuming Kalshi payouts are $1.00)
    df["pnl_per_contract"] = 0.0
    
    # YES bets payout: 
    # If they win, we get $1.00 back (Profit = 1 - cost). If they lose, we lose the cost.
    df.loc[df["bet_yes"] & (df["home_win"] == 1), "pnl_per_contract"] = 1.0 - df["ask_price"]
    df.loc[df["bet_yes"] & (df["home_win"] == 0), "pnl_per_contract"] = -df["ask_price"]
    
    # NO bets payout:
    # Cost is (1 - bid_price). If home loses (NO wins), we get $1.00 back.
    df.loc[df["bet_no"] & (df["home_win"] == 0), "pnl_per_contract"] = df["bid_price"]
    df.loc[df["bet_no"] & (df["home_win"] == 1), "pnl_per_contract"] = -(1.0 - df["bid_price"])
    
    # Scale PnL based on a flat $10 bet
    df["contracts_bought"] = 0.0
    # For YES bets, $10 buys us (10 / ask_price) contracts
    df.loc[df["bet_yes"], "contracts_bought"] = BET_SIZE / df["ask_price"]
    # For NO bets, $10 buys us (10 / (1 - bid_price)) contracts
    df.loc[df["bet_no"], "contracts_bought"] = BET_SIZE / (1.0 - df["bid_price"])
    
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

    # ---------------------------------------------------------
    # 5. Error Analysis (Worst Losing Trades)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("ERROR ANALYSIS: TOP 50 WORST LOSING TRADES")
    print("="*50)
    
    # Filter for trades that were placed and lost money
    losing_trades = df[((df["bet_yes"]) | (df["bet_no"])) & (df["trade_pnl"] < 0)].copy()
    
    if len(losing_trades) > 0:
        # Sort by worst PnL
        losing_trades = losing_trades.sort_values("trade_pnl", ascending=True).head(50)
        
        # Calculate which edge triggered the bet for display
        losing_trades["bet_type"] = np.where(losing_trades["bet_yes"], "YES", "NO")
        losing_trades["active_edge"] = np.where(losing_trades["bet_yes"], losing_trades["edge_yes"], losing_trades["edge_no"])
        
        # Select columns of interest for investigation
        cols_to_show = [
            "inning", "outs_when_up", "score_diff",
            "bet_type", "active_edge", "trade_pnl",
            "kalshi_price", "spread", "fair_prob", "final_prob"
        ]
        
        # Only include columns that actually exist
        cols_to_show = [c for c in cols_to_show if c in losing_trades.columns]
        
        # Round numerical columns for display
        losing_trades_display = losing_trades[cols_to_show].round(3)
        
        # Print with pandas option to show all columns
        with pd.option_context('display.max_rows', 50, 'display.max_columns', 15, 'display.width', 150):
            print(losing_trades_display)
    else:
        print("No losing trades found!")

if __name__ == "__main__":
    main()
