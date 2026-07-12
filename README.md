# MLB Kalshi Trader

An automated machine learning trading system designed to predict and exploit market overreactions in MLB live betting markets on [Kalshi](https://kalshi.com/).

## Architecture

This system avoids the common trap of building an MLB probability model from scratch. Instead, it relies on a two-step hybrid approach that combines classical baseball statistics with modern machine learning:

1. **The Log5 Mathematical Baseline**
   We start with MLB Statcast's official live game state Markov probabilities (`home_win_exp`). Because this baseline ignores specific starting pitchers and pre-game Vegas odds, we apply a **Log5 Formula** transformation. This mathematically anchors the baseline curve so that it perfectly matches the Kalshi opening market price at the first pitch of the game. The result is a mathematically pure, perfectly anchored true win probability (`fair_prob`) that moves exactly according to game events.

2. **Market Reaction Model (The Trader)**
   The reaction model (CatBoost) does *not* try to predict baseball. It uses the `fair_prob` directly as a mathematical `baseline` (base margin in log-odds). The model is explicitly fed only market-level features (`market_error`, `kalshi_price`, `spread`, `volume`) to isolate and identify when the market diverges from the mathematical truth. By modeling just the residuals (the overreactions), it effectively spots when human traders panic or underreact, and corrects the probability.

3. **The Edge Calculation**
   Once the CatBoost model outputs the `final_prob` (its ultimate prediction of the true win probability), the system compares it against the live Kalshi order book to find mathematical edges:
   - `edge_yes = final_prob - ask`: The edge if we buy YES (predicting the team will win) at the current asking price.
   - `edge_no = bid - final_prob`: The edge if we sell YES / buy NO (predicting the team will lose) to current buyers at the bid price.
   If either edge exceeds the configured `EDGE_THRESHOLD` (e.g., >7%), the bot executes the trade.

4. **Stateful Portfolio Manager (The Executioner)**
   Rather than placing static, hold-to-expiration bets, the system runs a chronological, stateful simulation. It prevents duplicate bets on the same game, manages inventory, and actively "trades out" of positions (hedging/selling early) when the mathematical edge vanishes, successfully locking in profits before the game ends.

## Performance

Backtesting on an unseen chronologically-split holdout dataset (June 28, 2026 - July 10, 2026) yielded the following per-pitch simulated performance. 

*Assumption: Flat $10 bets on any pitch where the Market Reaction Model identifies a >15% edge against the Kalshi midpoint. The position is actively managed and sold early if the edge flips.*

- **Total Trades Opened:** 46
- **Positions Traded Out Early:** 38 (82% of trades were hedged or locked for profit before expiration)
- **Total Capital Risked:** $460.00
- **Net Profit (PnL):** $180.93
- **ROI:** 39.33%

## Pipeline Structure

The codebase is split into three distinct phases: Data Processing, Modeling, and Backtesting.

### 1. Data Processing
* `data/processed/scripts/build_event_state_features.py`: Parses raw statcast pitches and extracts the baseline `home_win_exp`.
* `data/processed/scripts/build_kalshi_join.py`: Chronologically joins the Statcast pitch data with Kalshi 1-minute candlestick order books using a backward ASOF join to prevent look-ahead bias. It also explicitly extracts the 1st-pitch `pregame_prob` to fuel the Log5 anchor.
* `data/processed/scripts/apply_feature_preprocessing.py`: Normalizes and processes features, ensuring no look-ahead data leakage.

### 2. Modeling
* `models/train_market_reaction_model.py`: Calculates the Log5 anchored `fair_prob`, builds the CatBoost `Pool` with the baseline log-odds, and trains a highly constrained Market Reaction Model to isolate market inefficiencies.

### 3. Backtesting
* `backtesting/evaluate_strategy.py`: Runs the chronological stateful portfolio simulation on the unseen `test_dataset.parquet`. It evaluates Log5 probabilities, runs the CatBoost residual inference, executes trades at a >15% edge threshold, and actively hedges/exits positions dynamically, outputting final ROI and PnL.

## Usage

To rebuild the entire pipeline and evaluate the strategy from scratch, run the scripts in sequence:

```bash
# 1. Process data and build datasets
python data/processed/scripts/build_event_state_features.py
python data/processed/scripts/build_kalshi_join.py
python data/processed/scripts/apply_feature_preprocessing.py

# 2. Train models
python models/train_market_reaction_model.py

# 3. Evaluate economic edge
python backtesting/evaluate_strategy.py
```
