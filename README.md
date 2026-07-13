# MLB Kalshi Trader

Research and paper-trading pipeline for testing MLB in-game Kalshi strategies
with identical historical and live feature semantics.

## Hybrid event-residual strategy

The hybrid does not trade the local model's absolute disagreement with
Kalshi. For each safely observed completed hit, it:

1. Saves the pre-event Kalshi midpoint and local fair probability.
2. Applies the local model's fair **log-odds change** to that Kalshi anchor.
3. Compares the resulting target with the executable post-event bid/ask.
4. Requires the excess move to survive one later historical quote (or two
   seconds in live paper trading) before entering.
5. Updates the target for later game-state changes and exits on reversion or a
   tuned time limit.

Historical hit results are not considered observable until the following
pitch starts because the source data contains pitch-start—not pitch-end—times.
Candles containing more than one completed plate appearance are skipped.

## Corrected architecture

1. `build_kalshi_join.py` creates one decision row per completed market candle.
   Each row uses the candle's actual closing bid and ask and sees only an MLB
   state whose timestamp is no later than the decision time.
2. A local win-expectancy CatBoost model uses raw game state: inning half,
   outs, score, count, bases, and the pregame Kalshi anchor. Live trading no
   longer waits for MLB `contextMetrics`.
3. The reaction model uses the same unscaled raw features in training,
   backtesting, and live trading. There is no `StandardScaler`, hard-coded
   volume, or mismatched price-age feature.
4. Historical signals execute only at a later actual bid/ask observation and
   only if the edge remains at least 15%. Entry/exit taker fees are included.
5. Live snapshots are fetched concurrently, missing anchors and one-sided
   books fail closed, and every decision logs market/state receipt timestamps.

Shared definitions and validation live in `mlb_kalshi/strategy.py`.

## Current causal result

On the chronological June 28–July 10 holdout:

- 27,450 market decision rows across 175 games
- 349 later-observation fills
- $280.82 in fees
- −$523.09 net PnL
- −14.32% ROI

The earlier 39.33% result used a newly updated pitch state with an older
one-minute candle and treated that historical candle as executable. It also
used different feature definitions and thresholds in live trading. The result
does not survive causal observation-time alignment.

The hybrid was tuned without the outer holdout. No training-period setting
was profitable after spread and fees, so live execution is disabled. The
selected setting (4% edge, eight-minute maximum hold) returned
**−9.93% ROI** on the June 28–July 10 holdout: 461 trades, −$473.88 PnL, and
$311.13 in fees. This is better than the absolute-probability strategy but is
not evidence of a tradable edge.

## Rebuild

```bash
.venv/bin/python data/processed/scripts/build_kalshi_join.py
.venv/bin/python data/processed/scripts/apply_feature_preprocessing.py
.venv/bin/python models/train_market_reaction_model.py
.venv/bin/python models/tune_hybrid_strategy.py
.venv/bin/python backtesting/evaluate_strategy.py
.venv/bin/python backtesting/evaluate_hybrid_strategy.py
```

Run tests with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Paper trader

Set the game and home-team market before starting:

```bash
MLB_GAME_PK=824491 \
KALSHI_MARKET_TICKER=KXMLBGAME-26JUL121340CHCCIN-CIN \
.venv/bin/python live_trading_engine/paper_trader.py
```

The paper trader refuses invalid/missing pregame anchors and uses the local
state model plus the current actual order book. It does not place real orders.
Because validation was negative, it currently refuses to start. To collect
paper observations without enabling any real trading, explicitly set
`ALLOW_UNVALIDATED_HYBRID=1`.
