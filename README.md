# MLB Kalshi Trader

An automated machine learning trading system designed to predict and exploit market overreactions in MLB live betting markets on [Kalshi](https://kalshi.com/).

## Architecture

This system avoids the common trap of trying to build a single model that understands both baseball and financial markets. Instead, it uses a two-model meta-learning architecture:

1. **Baseball State Model (The Baseline)**
   An unbiased model trained solely on MLB play-by-play data (Statcast). It is completely blinded to the betting market and predicts the "fair probability" of a home team win based strictly on the current game state (inning, outs, score differential, base runners, hitter form, etc.).

2. **Market Reaction Model (The Trader)**
   A meta-model that takes the `fair_probability` from the State Model and compares it to live `kalshi_price` order books. It is trained to identify when Kalshi bettors are systematically overvaluing or undervaluing a team, predicting whether fading or tailing the market is the profitable play.

## Performance

Backtesting on an unseen chronologically-split holdout dataset (June 28, 2026 - July 10, 2026) yielded the following per-pitch simulated performance. 

*Assumption: Flat $10 bets on any pitch where the Market Reaction Model identifies a >5% edge against the Kalshi midpoint.*

- **Win Rate:** 48.07% *(Profitable due to asymmetric payouts on undervalued bets)*
- **Total Capital Risked:** $368,200.00
- **Net Profit (PnL):** $186,559.38
- **ROI:** 50.67%

## Pipeline Structure

The codebase is split into three distinct phases: Data Processing, Modeling, and Backtesting.

### 1. Data Processing
* `data/processed/scripts/build_event_state_features.py`: Parses raw statcast pitches into current game state features.
* `data/processed/scripts/build_kalshi_join.py`: Chronologically joins the Statcast pitch data with Kalshi 1-minute candlestick order books using a backward ASOF join to prevent look-ahead bias. Calculates final `home_win` outcomes and performs the 80/20 train/test split.
* `data/processed/scripts/apply_feature_preprocessing.py`: Normalizes continuous variables using a standard scaler (fitted only on the training set to prevent data leakage) and engineers categorical market signals (e.g. bid/ask spread).

### 2. Modeling
* `models/train_baseball_model.py`: Trains the foundational CatBoost Baseball State Model.
* `models/train_market_reaction_model.py`: Uses the trained State Model to generate fair probabilities in-memory, calculates the market error, and trains the CatBoost Market Reaction Model.

### 3. Backtesting
* `backtesting/evaluate_strategy.py`: Simulates placing trades on the unseen `test_dataset.parquet` at a >5% edge threshold, outputting final ROI, win rates, and PnL.

## Usage

To rebuild the entire pipeline and evaluate the strategy from scratch, run the scripts in sequence:

```bash
# 1. Process data and build datasets
python data/processed/scripts/build_event_state_features.py
python data/processed/scripts/build_kalshi_join.py
python data/processed/scripts/apply_feature_preprocessing.py

# 2. Train models
python models/train_baseball_model.py
python models/train_market_reaction_model.py

# 3. Evaluate economic edge
python backtesting/evaluate_strategy.py
```
