# Settlement-value strategy

This strategy estimates the home team's final settlement probability after any
safely observed pitch transition. It buys YES or NO only when the calibrated
settlement value remains sufficiently far from the executable Kalshi price
after fees. A filled position is held until the game settles.

The folder was formerly named `mispricing_strategy`. “Settlement value” is
more precise: the model predicts the binary game outcome, not a short-term
price move.

## Strategy thesis

Kalshi already embeds substantial information in its pre-event market price.
After a pitch changes the baseball state, a local win-expectancy model
estimates how much the fair home-win log-odds should move. The strategy anchors
that state move to the last safely observed pre-event Kalshi execution, then
asks a second model whether the currently observed market price is too high or
too low relative to eventual settlement.

For pre-event market probability `M0`, local fair probability before the pitch
`F0`, and local fair probability after it `F1`, the state-adjusted market target
is:

```text
target = logistic(logit(M0) + logit(F1) - logit(F0))
```

This preserves Kalshi as the prior while applying only the state model’s
incremental information.

## Causal decision timeline

For every completed pitch:

1. Use the last Kalshi execution strictly before `pitch_end - anchor_buffer`.
2. Reject the event if that anchor is stale.
3. Wait the configured observation delay after the authoritative MLB pitch-end
   timestamp.
4. Use the first exact Kalshi execution at or after that delay as the signal
   price.
5. Build flow features only from executions strictly before the signal trade.
6. Score settlement probability and evaluate fee-aware YES and NO value.
7. Search only strictly later executions inside the causal fill window.

The next pitch invalidates the fill window. Same-timestamp and pre-signal
executions cannot fill an order.

## Model features

The calibrated CatBoost settlement model uses:

- current home-market price and log-odds;
- local fair probability before and after the pitch;
- fair and market log-odds moves;
- anchored state target and market residual;
- inning, half, outs, score differential, count, and runners;
- state deltas caused by the pitch;
- anchor age and actual observation delay;
- two-second pre-signal trade count, volume, flow imbalance, and volatility.

Event names such as single, walk, strikeout, or home run are deliberately not
features. The contract is event-agnostic and represents the observable state
transition instead.

Raw CatBoost probabilities are calibrated by a one-dimensional logistic model
fit on a later chronological interval.

## Entry, fill, and settlement

For a fixed dollar stake, the strategy computes expected PnL for YES and NO
after Kalshi’s rounded taker fee. It selects the stronger side, then requires:

- the configured side filter;
- minimum probability edge;
- minimum expected net PnL;
- sufficient reported size at a compatible later taker-side execution.

All thresholds are rechecked at the eventual fill price. It permits at most
one position per game. There is no early exit in this strategy; profit and loss
are determined by final game settlement.

The selected high-coverage policy is NO-only with a 4-point minimum edge and $0.50
minimum predicted net value on a $10 stake. Selection maximizes tuning trades only
among policies that are profitable in every chronological fold, retain at least
20% ROI in the worst fold, and earn at least 25% aggregate tuning ROI.

## Data and training flow

Run commands from the repository root. Download and process data specifically
for this strategy:

```bash
.venv/bin/python setup_data.py mispricing
```

This writes common inputs under `data/shared/` and strategy-specific derived
rows under `data/settlement_value/`:

```text
data/settlement_value/decision_rows.parquet
data/settlement_value/execution_trades.parquet
```

Then train and evaluate:

```bash
.venv/bin/python -m settlement_value_strategy.train
.venv/bin/python -m settlement_value_strategy.backtest
```

`setup_data.py mispricing` already runs the preparation step. To rerun only
that derivation:

```bash
.venv/bin/python -m settlement_value_strategy.prepare_data
```

Training chronology is fixed:

- fit before June 17, 2026;
- calibrate June 17-21;
- tune thresholds and side June 22-27;
- evaluate the outer development holdout from June 28 onward.

`train` rewrites the model, calibration, policy configuration, tuning grid,
and training summary. `backtest` rewrites holdout summaries and trade records.
Always retrain after regenerating shared data because a frozen model can be
incompatible with changed anchors or preprocessing even if schemas match.

## Tests

```bash
.venv/bin/python -m unittest \
  settlement_value_strategy.test_strategy \
  settlement_value_strategy.test_pipeline -v
```

## Paper trading

Offline JSONL scoring:

```bash
.venv/bin/python -m settlement_value_strategy.paper_trader \
  --input decisions.jsonl
```

Live single-game paper trading:

```bash
MLB_GAME_PK=... KALSHI_MARKET_TICKER=... \
ALLOW_UNVALIDATED_MISPRICING=1 \
  .venv/bin/python -m settlement_value_strategy.live_paper_trader
```

Daily discovery and multi-game paper mode:

```bash
.venv/bin/python -m settlement_value_strategy.live_paper_trader \
  --discover-only --date YYYY-MM-DD

ALLOW_UNVALIDATED_MISPRICING=1 \
  .venv/bin/python -m settlement_value_strategy.live_paper_trader \
  --all-games --date YYYY-MM-DD
```

The live trader reconstructs the same pitch/trade feature contract, rejects
polling gaps, shares cash across workers in SQLite, recovers open positions on
restart, and settles at game end. It never submits real orders. The override
permits paper observation while deployment is disabled.

## Docker and reference result

The external strategy selector remains `mispricing` for command compatibility:

```bash
docker build -t mlb-kalshi-trader .
docker run --rm mlb-kalshi-trader mispricing pipeline
docker run --rm mlb-kalshi-trader mispricing backtest
```

The current development holdout contains 103 trades across 220 games,
$236.92 net PnL, and 22.24% ROI. It is development evidence, not a pristine final test; deployment remains
disabled pending forward paper validation.

