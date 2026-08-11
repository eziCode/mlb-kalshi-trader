# Event-reaction reversion strategy

This strategy trades delayed Kalshi reactions after configured completed MLB
events. It estimates a state-adjusted market target immediately after the
event, enters when the exact trade tape remains sufficiently far from that
target, and exits when the market reaches its configured target or hold limit.
Live exits use immediate-or-cancel sells: any available quantity is sold and
the remaining contracts stay open for later eligible exits.
Positions still open at game end settle normally.

The folder was formerly named `portable_trade_tape_strategy`. “Hit reversion”
describes the economic thesis; the trade tape is the execution mechanism.

## Strategy thesis

Completed events can produce abrupt changes in base occupancy, outs, score,
and home-team win probability. Kalshi may react incompletely or with a short delay. The
strategy compares the observed market against the change implied by a local
win-expectancy model while retaining the pre-event Kalshi price as its prior.

For pre-hit market probability `M0`, local fair probability before the hit
`F0`, and fair probability after it `F1`:

```text
target = logistic(logit(M0) + logit(F1) - logit(F0))
```

The target moves dynamically if later baseball state changes occur while a
candidate or position is active.

## Event and signal lifecycle

1. Observe a newly completed plate appearance from the authoritative MLB feed.
2. Continue only for an event type listed in the loaded configuration. The
   checked-in policy includes singles, triples, walks, intentional
   walks, hit-by-pitches, field errors, fielder's choices, and catcher
   interference; home runs are excluded.
3. Compute the directional fair-value move for the batting team.
4. Anchor to a fresh Kalshi execution observed before the event.
5. Compare the anchored target with exact subsequent Kalshi executions.
6. Start watching the side whose fee-adjusted residual exceeds the configured
   edge. The stale direct realized-value model is disabled.
7. Require that side to persist through the confirmation interval.
8. Expire the candidate after the entry deadline or invalidate it on the next
   pitch/material state transition.

Startup events are never traded: a worker first establishes a live baseline,
then considers only events observed afterward.

## Local win-expectancy model

The packaged CatBoost model estimates home-win probability from:

- pregame home probability;
- inning and top/bottom half;
- outs;
- home score differential;
- balls and strikes;
- runners on first, second, and third.

It is trained chronologically with inverse game-frequency weights so games
with many pitches do not dominate the loss. This state model produces the
incremental fair move; it is not itself the trading policy.

## Entry and execution assumptions

The backtest uses executed trades, not reconstructed quotes. Its fill contract
requires a later execution on the compatible taker side after the measured
0.68-second submission latency, with enough reported size. The simulator
remains a fill proxy rather than a historical order-book reconstruction.

The checked-in deployment policy currently uses:

- the configured event types listed above;
- both YES- and NO-side residuals, with paired away-YES execution for NO;
- no direct reversion-value model; its causal retraining failed the forward
  deployment gate;
- a five-point minimum fee-adjusted edge with no confirmation delay;
- fixed-budget sizing of $2.50 per entry;
- unlimited positions per game, with at least 60 seconds between entries;
- ten-second maximum pre-event anchor age;
- twenty-second event-to-entry deadline;
- next-pitch invalidation;
- completed-event alignment: the event's terminal pitch must still be the
  newest MLB state visible at the decision point;
- atomic hit state derived from the play's own runner movements, score, and
  outs; at-bat progression is logged for validation but never authorizes a
  late entry;
- no additional minimum local fair move;
- no entries after the ninth inning.

Fees use Kalshi’s rounded taker-fee formula.

The 60-second cooldown is scoped to hit reversion in the shared live risk
ledger. Settlement-value orders in the same game do not start or extend this
cooldown. There is no hidden live maximum-trades or maximum-positions limit;
each MLB event can produce at most one entry, and the durable event key only
prevents the same event from being submitted twice after a retry or restart.

## Exit behavior

For a home YES position, target reversion occurs when the observed home YES
price reaches or exceeds the configured target. A home-NO signal is executed
as paired away-team YES, so its exit is evaluated against the held away YES
contract reaching `1 - home_target`; it does not borrow the home market's
price, liquidity, or taker direction.

The checked-in policy has a 240-second maximum hold. Its exit target updates
with subsequent causal baseball states, preserving the original market anchor
while allowing a changed game state to move the price at which the reversion
thesis is considered complete. Optional momentum logic
can delay a reversion exit while the held-side price continues moving
favorably, then exit on velocity reversal, trailing giveback, or the momentum
hold limit. Momentum is disabled in the selected configuration. Any remaining
position settles at the final game outcome.

Live entries and exits are immediate-or-cancel. Every available contract up to
the $2.50 budget is accepted; an unfilled entry remainder is cancelled, while an
exit can sell partially and keeps the unsold position active for later eligible
trades or settlement. The shared account cap and duplicate-order ledger are
safety controls, not strategy activity limits.

## Data, tuning, and evaluation

Run data setup from the repository root:

```bash
.venv/bin/python setup_data.py trade-tape
```

Normalized executions are written to `data/shared/`. State probabilities use
the MLB-only batting-perspective win model and leakage-free updates under
`data/settlement_value/`, shared with the settlement-value strategy so live
and research calculations use the same feature contract and pregame prior.

Tune on pre-holdout dates, then evaluate the fixed outer holdout. The worker processes share the read-only tuning frames on macOS and Linux; use `--workers 1` for sequential debugging:

```bash
(cd hit_reversion_strategy && ../.venv/bin/python scripts/tune.py --workers 8)
(cd hit_reversion_strategy && ../.venv/bin/python scripts/backtest.py)
```

To audit a specific live window without overwriting the standard holdout
artifacts, provide inclusive game dates and a separate output prefix. The
default live fill proxy only counts a fill after a compatible post-signal
execution, avoiding the optimistic assumption that every observed trade was
buyable at its printed price. Pass `--no-live-fill-proxy` only to reproduce
the older optimistic research assumption:

The deployment replay also waits 0.68 seconds after both entry and exit
decisions before searching for fill evidence. This is the median submission
latency measured from the live risk ledger; it can be overridden with the
backtest latency flags.

```bash
(cd hit_reversion_strategy && ../.venv/bin/python scripts/backtest.py \
  --start-date 2026-07-24 --end-date 2026-08-09 \
  --output-prefix live_window_conservative)
```

The tuner rewrites `models/trade_tape_config.json`. The backtest rewrites the
holdout artifacts and refuses to enable deployment unless the loaded policy
was already enabled and remains profitable.

## Tests

```bash
(cd hit_reversion_strategy && \
  ../.venv/bin/python -m unittest discover -s tests -v)
```

Tests cover confirmation, candidate expiry, next-pitch invalidation, exact
later-trade fill timing, rejection of trades preceding live event observation,
state-model feature parity, independent home/away liquidity, partial reversion
exits, strategy-scoped cooldowns, and momentum-delayed exits.

## Live paper trading

Single game:

```bash
MLB_GAME_PK=... KALSHI_MARKET_TICKER=... \
ALLOW_UNVALIDATED_HYBRID=1 \
  .venv/bin/python hit_reversion_strategy/scripts/paper_trade.py
```

Discovery and all-game coordination:

```bash
.venv/bin/python hit_reversion_strategy/scripts/paper_trade.py \
  --discover-only --date YYYY-MM-DD

ALLOW_UNVALIDATED_HYBRID=1 \
  .venv/bin/python hit_reversion_strategy/scripts/paper_trade.py \
  --all-games --date YYYY-MM-DD
```

Each game runs in an isolated worker while all workers share SQLite-backed
cash and positions. The trader polls public MLB and Kalshi endpoints, validates
quote/feed freshness, recovers positions after restart, and never submits real
orders.

## Docker and reference result

The external strategy selector remains `trade-tape` for command compatibility:

```bash
docker build -t mlb-kalshi-trader .
docker run --rm mlb-kalshi-trader trade-tape tune
docker run --rm mlb-kalshi-trader trade-tape backtest
```

The current exact-policy research holdout contains 581 fills across 297 games,
$112.94 net PnL, and 7.78% ROI at the live $2.50 budget. Removing the best game
leaves $100.19 and removing the best four leaves $82.26. The replay uses the checked-in
shared-WebSocket observation-latency profile, paired away-team YES execution
using only actual away-market trade size and aggressor direction,
dynamic targets, partial exits, the 60-second entry cooldown, and the same
direct-value gate loaded by live trading.

The value-model metadata hashes the model binary, deployment configuration,
and latency profile. Both replay and live startup fail closed if those files do
not match, preventing a model trained under one policy from silently running
under another. This holdout has been reused during strategy development, so
its 7.78% ROI is a research diagnostic rather than an unbiased forward-return
estimate. Historical executions remain a fill proxy rather than a full
order-book reconstruction.
