# Settlement-value strategy

This strategy predicts the causal short-horizon market repricing after a safely
observed pitch transition, anchors that residual to the current Kalshi price,
and buys only when the resulting probability remains sufficiently far from the
executable price after fees. The deployed policy normally holds positions to
game settlement; early exits are currently disabled.

The folder was formerly named `mispricing_strategy`. The current deployed model
is a latency-residual regressor rather than the legacy unrestricted binary
winner classifier retained for research compatibility.

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

The deployment backtest advances the effective order time by 4.72 seconds,
the measured median settlement-value submission latency in the live risk
ledger. A print observed before that effective time cannot fill the order.

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

The deployed CatBoost regressor predicts the home market's causal 3-10 second
log-odds move. That bounded residual is added to the signal-time market logit;
the legacy settlement classifier and its probability calibration are not used
by the live policy.

## Entry, fill, and settlement

The paper deployment uses the market-anchored latency model in
`model/live_config.json`.  It predicts the causal 3-10 second post-pitch market
log-odds move and adds that bounded residual to the observed market logit.  It
does not use the legacy unrestricted winner classifier as an absolute fair
probability.  Retrain it with:

```bash
python -m settlement_value_strategy.train_latency
```

The walk-forward research harness is
`python -m settlement_value_strategy.research_latency`. The checked-in latency
configuration is enabled and records `tuning_passed` and `validation_passed`
as true. Executions from 45 through 55 cents are excluded because this
maximum-fee/maximum-uncertainty band was negative in pre-final walk-forward
results. Real-money execution still requires the independent
`LIVE_TRADING_ENABLED` acknowledgement and account-level capital limits.

For a fixed dollar stake, the strategy computes expected PnL after Kalshi’s
rounded taker fee. Each signal independently chooses YES or NO on the home-team
market according to the larger fee-adjusted expected value. It requires:

- the configured side filter;
- minimum probability edge;
- minimum expected net PnL;
- sufficient reported size at a compatible later taker-side execution.

All thresholds are rechecked at the eventual fill price. There is no early
exit in this strategy; profit and loss are determined by final game settlement.

The policy buys home YES for home-team signals and routes away-team signals to
the paired away-YES market. Entries start in inning 2, require a compatible
post-signal trade within five seconds, allow at most two open positions per
game, and must be separated by at least 120 seconds. The research selector
requires at least 0.75 fills per scheduled game, positive aggregate PnL,
positive PnL in most chronological folds, and resistance to top-game
concentration. Ties favor the strongest worst chronological fold.

The frozen-policy expanding-window replay currently contains 1,071 fills over
1,440 scheduled games (0.744/game), +$201.83 net PnL, and 8.74% ROI at the
actual $2.50 order budget. All seven folds are positive; the July 18-22 final
holdout contains 41 fills, +$25.12, and 27.62% ROI. Removing its best four
games leaves +$2.80.

The two-position limit counts concurrently open positions, not lifetime trades
in a game. A fully exited reversal frees a slot. The 120-second cooldown is
scoped to settlement value in the shared live risk ledger, so a hit-reversion
order cannot suppress a settlement signal. Durable trigger keys prevent only
duplicate submission of the same pitch decision.

Opposite-side entries are not blocked unconditionally. Live and replay both
permit a reversal only when the new trade's expected PnL exceeds the realized
loss and fees required to unwind every conflicting position. Live unwinds use
immediate-or-cancel partial sells; if any conflicting contracts remain, the
opposite entry is skipped and the remainder stays tracked. Fill-or-kill is not
used for this path.

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
data/settlement_value/away_execution_trades.parquet
data/settlement_value/state_updates.parquet
```

Train the deployed latency model and evaluate its frozen live policy:

```bash
.venv/bin/python -m settlement_value_strategy.train_latency
.venv/bin/python -m settlement_value_strategy.research_latency
.venv/bin/python -m settlement_value_strategy.backtest_live_policy
```

`setup_data.py mispricing` already runs the preparation step. To rerun only
that derivation:

```bash
.venv/bin/python -m settlement_value_strategy.prepare_data
```

Research uses expanding-window chronological folds: every model is trained
strictly before the fold it scores. Policy selection excludes the final July
18-22 period. The final production model is then trained through July 22 for
future games, and `live_config.json` records and verifies the exact model-file
hash at startup.

The settlement-specific local state model is trained from MLB-only game states
from the 2023 pitch-clock season onward, using only games strictly before June
17, 2026. It does not require historical Kalshi data or a synthetic pregame
market prior. Its batting-team feature contract is identical in preprocessing
and live inference. Kalshi-linked rows remain exclusive to settlement-model
training and execution replay.

`train_latency` atomically rewrites the latency model and live configuration.
`research_latency` rewrites the chronological selection artifacts, and
`backtest_live_policy` rewrites the exact deployed-policy fold results. Always
retrain after regenerating shared data because a frozen model can be
incompatible with changed anchors or preprocessing even if schemas match.

## Tests

```bash
.venv/bin/python -m unittest \
  settlement_value_strategy.test_strategy \
  settlement_value_strategy.test_pipeline -v
```

## Early-exit research

The research-only early-exit overlay keeps the deployed entry set fixed, then
tests stop-loss exits against exact later executions:

```bash
.venv/bin/python -m settlement_value_strategy.research_early_exit
```

An exit requires a later scored MLB state and a strictly later opposite-taker
execution within five seconds. Each observed execution supplies only its
printed size; the simulator sells that partial quantity and keeps managing the
remainder. Exit fees are charged and any unsold contracts still settle. Results
are written to `results/early_exit_research_summary.json` and
`results/early_exit_research_trades.csv`.

The current frozen policy has early exits disabled. The previous 20-point stop
improved aggregate tuning PnL but reduced the earlier untouched holdout, and
the historical overlay uses trade prints while live execution observes an
order-book bid. It therefore remains a research experiment rather than part of
the parity-tested production policy.

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
polling gaps and signals older than the five-second fill window, requires
enough top-level size, and applies the same model-side eligibility rule as the
backtest. It logs all model features and execution timing to a versioned CSV.
Workers share cash through SQLite; startup reconciliation settles positions
whose original worker missed the final game state. Paper mode never submits
orders; guarded live mode requires the explicit real-money acknowledgement.
`ALLOW_UNVALIDATED_MISPRICING=1` permits paper observation when a future loaded
policy is disabled.

## Docker and reference result

The external strategy selector remains `mispricing` for command compatibility:

```bash
docker build -t mlb-kalshi-trader .
docker run --rm mlb-kalshi-trader mispricing backtest
```

The deployed policy is now the market-anchored latency-residual model in
`model/live_config.json`, not the legacy settlement classifier in
`model/config.json`. The saved expanding-window replay contains 1,071 fills
across 1,440 games, producing $201.83 net PnL and 8.74% ROI at the live $2.50
budget. Its July 18-22 final fold has 41 fills, $25.12 net PnL, and 27.62% ROI;
removing its four best games leaves $2.80. This short final period is promising
but not sufficient by itself to establish durable profitability. Additional
same-side positions are allowed only when both probability and expected return
improve.
