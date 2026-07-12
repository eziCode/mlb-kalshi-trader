import os
import csv
import time
import requests
import asyncio
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from catboost import CatBoostClassifier, Pool

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
GAME_PK = 823358  # Brewers vs Pirates (July 12)
MARKET_TICKER = "KXMLBGAME-26JUL121215MILPIT-PIT"

EDGE_THRESHOLD = 0.07
BET_SIZE = 10.0

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "../models/market_reaction_model/reaction_model.cbm")
SCALER_PATH = os.path.join(SCRIPT_DIR, "../data/processed/train/scaler.joblib")

# ---------------------------------------------------------
# State Management
# ---------------------------------------------------------
class PaperPortfolio:
    def __init__(self, starting_cash=1000.0):
        self.cash = starting_cash
        self.position = 0.0  # Positive = YES, Negative = NO
        self.entry_price = 0.0
        self.bet_count = 0
        
    def log_trade(self, action, size, price, edge, current_prob, current_inning):
        print(f"\n[{time.strftime('%H:%M:%S')}] 🚨 [EXECUTE] {action} {size:.2f} CONTRACTS @ ${price:.2f}")
        print(f"   => Inning: {current_inning} | Edge: {edge:.1%} | Model Prob: {current_prob:.1%}")
        print(f"   => Portfolio Cash: ${self.cash:.2f} | Open Pos: {self.position:.2f}\n")

# ---------------------------------------------------------
# Data Fetching
# ---------------------------------------------------------
async def fetch_kalshi_orderbook(ticker):
    """Polls Kalshi's public REST API for the orderbook"""
    url = f"https://external-api.kalshi.com/trade-api/v2/markets/{ticker}/orderbook"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        
        ob = data.get("orderbook_fp", {})
        yes_bids = ob.get("yes_dollars", [])
        no_bids = ob.get("no_dollars", [])
        
        best_bid = float(yes_bids[-1][0]) if yes_bids else 0.01
        best_no_bid = float(no_bids[-1][0]) if no_bids else 0.01
        best_ask = round(1.0 - best_no_bid, 2)
        
        # Guard against zero-liquidity spreads
        if best_ask <= best_bid:
            best_ask = best_bid + 0.01
            
        midpoint = round((best_bid + best_ask) / 2.0, 3)
        spread = round(best_ask - best_bid, 3)
        
        return midpoint, spread, best_bid, best_ask
    except Exception as e:
        print(f"Error fetching Kalshi orderbook: {e}")
        return None, None, None, None

async def fetch_mlb_live_state(game_pk):
    """Polls MLB Stats API for the live game state"""
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        
        live_data = data.get("liveData", {})
        
        # Extract Inning
        inning = live_data.get("linescore", {}).get("currentInning", 1)
        
        # Extract home win probability (Vegas/MLB baseline)
        plays = live_data.get("plays", {})
        current_play = plays.get("currentPlay", {})
        about = current_play.get("about", {})
        
        # Fallback to about, but override with contextMetrics which is live
        try:
            metrics_url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/contextMetrics"
            metrics_data = requests.get(metrics_url, timeout=5).json()
            hwe_percent = metrics_data.get("homeWinProbability", about.get("homeWinProbability", 50.0))
        except:
            hwe_percent = about.get("homeWinProbability", 50.0)
        home_win_exp = hwe_percent / 100.0
        
        # Get game status
        status = data.get("gameData", {}).get("status", {}).get("abstractGameState", "Preview")
        
        # Get count
        balls = live_data.get("linescore", {}).get("balls", 0)
        strikes = live_data.get("linescore", {}).get("strikes", 0)
        outs = live_data.get("linescore", {}).get("outs", 0)
        
        return inning, home_win_exp, status, balls, strikes, outs
    except Exception as e:
        print(f"Error fetching MLB live state: {e}")
        return 1, 0.5, "Preview", 0, 0, 0

async def get_pregame_anchors(game_pk, market_ticker):
    """
    Dynamically fetches the exact Kalshi midpoint and MLB probability from 
    the minute before the first pitch was thrown.
    """
    # 1. Get Game Start Time and Status from MLB
    url_live = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    try:
        data = requests.get(url_live, timeout=5).json()
        dt_str = data["gameData"]["datetime"]["dateTime"]
        start_ts = int(datetime.fromisoformat(dt_str).timestamp())
        status = data.get("gameData", {}).get("status", {}).get("abstractGameState", "Preview")
    except Exception as e:
        print(f"Error fetching game start time: {e}")
        return 0.5, 0.5
        
    # 2. Set Historical Pregame MLB Prob
    if status == "Preview":
        try:
            url_metrics = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/contextMetrics"
            metrics = requests.get(url_metrics, timeout=5).json()
            pregame_hwe = metrics.get("homeWinProbability", 50.0) / 100.0
            print(f"  -> Game is in Preview. Fetched precise MLB pregame prob: {pregame_hwe:.1%}")
        except:
            pregame_hwe = 0.50
    else:
        print("  -> Game already Live. Falling back to 50.0% MLB pregame prob to avoid corruption.")
        pregame_hwe = 0.50
    # 3. Get Historical Pregame Kalshi Price
    kalshi_url = f"https://api.elections.kalshi.com/trade-api/v2/series/KXMLBGAME/markets/{market_ticker}/candlesticks"
    params = {"start_ts": start_ts - 120, "end_ts": start_ts, "period_interval": 1}
    try:
        candles = requests.get(kalshi_url, params=params, timeout=5).json().get("candlesticks", [])
        if candles:
            last_candle = candles[-1]
            bid = float(last_candle.get("yes_bid", {}).get("close_dollars", 0))
            ask = float(last_candle.get("yes_ask", {}).get("close_dollars", 0))
            if ask <= bid:
                ask = bid + 0.01
            pregame_prob = round((bid + ask) / 2.0, 3)
        else:
            pregame_prob = 0.5
    except Exception as e:
        print(f"Error fetching Kalshi candlesticks: {e}")
        pregame_prob = 0.5
        
    return pregame_prob, pregame_hwe

# ---------------------------------------------------------
# Log5 Formula
# ---------------------------------------------------------
def apply_log5(home_win_exp, pregame_prob, pregame_home_win_exp):
    we = np.clip(home_win_exp, 0.001, 0.999)
    p = np.clip(pregame_prob, 0.001, 0.999)
    we0 = np.clip(pregame_home_win_exp, 0.001, 0.999)
    
    odds_we = we / (1 - we)
    odds_p = p / (1 - p)
    odds_we0 = we0 / (1 - we0)
    
    odds_adj = odds_we * (odds_p / odds_we0)
    fair_prob = odds_adj / (1 + odds_adj)
    return fair_prob

# ---------------------------------------------------------
# Main Event Loop
# ---------------------------------------------------------
async def live_trading_loop():
    print("="*50)
    print(f"Starting Paper Trading Engine for Game {GAME_PK}")
    print(f"Monitoring Market: {MARKET_TICKER}")
    print("="*50)
    
    portfolio = PaperPortfolio()
    
    # Load Models (Paths adjusted for execution from live_trading_engine dir)
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    pregame_prob = None
    pregame_hwe = None
    last_price_update = time.time()
    last_midpoint = 0.5
    
    # Setup CSV Logging
    log_file = os.path.join(SCRIPT_DIR, "logs", f"paper_trade_log_{GAME_PK}_{int(time.time())}.csv")
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "status", "inning", "kalshi_mid", "spread", "mlb_hwe", "fair_prob", "model_prob", "edge_yes", "edge_no", "portfolio_cash", "open_position"])
    print(f"Logging live data to: {log_file}")
    
    # Dynamically Fetch Anchor
    print("\nFetching historical pre-game anchors autonomously...")
    pregame_prob, pregame_hwe = await get_pregame_anchors(GAME_PK, MARKET_TICKER)
    print(f"[ANCHOR SET] Pregame Kalshi: {pregame_prob:.1%} | Pregame MLB: {pregame_hwe:.1%}\n")
    
    while True:
        # 1. Fetch live market prices
        midpoint, spread, bid, ask = await fetch_kalshi_orderbook(MARKET_TICKER)
        
        # Track seconds since price update
        if midpoint != last_midpoint:
            last_price_update = time.time()
            last_midpoint = midpoint
        seconds_since_update = time.time() - last_price_update
        
        # 2. Fetch live MLB state
        inning, hwe, status, balls, strikes, outs = await fetch_mlb_live_state(GAME_PK)
        
        print(f"[{time.strftime('%H:%M:%S')}] Status: {status} | Inn: {inning} | Outs: {outs} | Count: {balls}-{strikes} | Kalshi Mid: {midpoint} | MLB HWE: {hwe:.1%}")
        
        if status == "Preview" or midpoint is None:
            await asyncio.sleep(5)
            continue
            
        if status == "Final":
            print("Game has ended. Settling portfolio...")
            # We would look at final score here to print final PnL
            break
            
        # 3. Calculate Fair Prob (Log5)
        fair_prob = apply_log5(hwe, pregame_prob, pregame_hwe)
        market_error = midpoint - fair_prob
        
        # 5. Predict Edge (Reaction Model)
        
        # Scale continuous features correctly using the pre-fitted scaler
        scaled_vals = scaler.transform(pd.DataFrame([{
            "volume": 1000, 
            "seconds_since_price_update": seconds_since_update
        }]))
        
        features = pd.DataFrame([{
            "market_error": market_error,
            "kalshi_price": midpoint,
            "pregame_prob": pregame_prob,
            "volume": scaled_vals[0][0], # The scaled volume
            "spread": spread,
            "seconds_since_price_update": scaled_vals[0][1], # The scaled seconds_since_update
            "inning": inning
        }])
        
        fp = np.clip(fair_prob, 0.0001, 0.9999)
        baseline = np.log(fp / (1 - fp))
        
        pool = Pool(data=features, baseline=[baseline])
        final_prob = model.predict_proba(pool)[0, 1]
        
        edge_yes = final_prob - ask
        edge_no = bid - final_prob
        
        # 5.5 Write Log
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.time(), status, inning, midpoint, spread, hwe, fair_prob, final_prob, edge_yes, edge_no, portfolio.cash, portfolio.position
            ])
        
        # 6. Portfolio Execution Logic
        # -- Early Exits / Hedging --
        if portfolio.position > 0:
            if final_prob < bid:
                cash_val = portfolio.position * bid
                portfolio.cash += cash_val
                portfolio.log_trade("SELL (CLOSE YES)", portfolio.position, bid, 0.0, final_prob, inning)
                portfolio.position = 0.0
                
        elif portfolio.position < 0:
            if final_prob > ask:
                cash_val = abs(portfolio.position) * (1 - ask)
                portfolio.cash += cash_val
                portfolio.log_trade("SELL (CLOSE NO)", abs(portfolio.position), 1 - ask, 0.0, final_prob, inning)
                portfolio.position = 0.0
                
        # -- Entries --
        if portfolio.position == 0.0:
            if edge_yes > EDGE_THRESHOLD:
                contracts = BET_SIZE / ask
                portfolio.position = contracts
                portfolio.cash -= BET_SIZE
                portfolio.log_trade("BUY YES", contracts, ask, edge_yes, final_prob, inning)
            
            elif edge_no > EDGE_THRESHOLD:
                contracts = BET_SIZE / (1 - bid)
                portfolio.position = -contracts
                portfolio.cash -= BET_SIZE
                portfolio.log_trade("BUY NO", contracts, 1 - bid, edge_no, final_prob, inning)
        
        # Poll every 5 seconds
        await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(live_trading_loop())
    except KeyboardInterrupt:
        print("\nPaper Trader stopped by user.")
