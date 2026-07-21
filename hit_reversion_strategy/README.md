# Hit-reaction reversion strategy

This strategy trades delayed Kalshi reactions after completed MLB hits. It
estimates the state-adjusted market target immediately after the hit, enters
when the exact trade tape remains sufficiently far from that target, and exits
when the market reverts to the updated target. Positions still open at game end
settle normally.

The folder was formerly named `portable_trade_tape_strategy`. “Hit reversion”
describes the economic thesis; the trade tape is the execution mechanism.

## Strategy thesis

Hits can produce abrupt changes in base occupancy, outs, score, and home-team
win probability. Kalshi may react incompletely or with a short delay. The
strategy compares the observed market against the change implied by a local
win-expectancy model while retaining the pre-hit Kalshi price as its prior.

For pre-hit market probability `M0`, local fair probability before the hit
`F0`, and fair probability after it `F1`:

```text
target = logistic(logit(M0) + logit(F1) - logit(F0))
```

The target moves dynamically if later baseball state changes occur while a
candidate or position is active.

## Event and signal lifecycle

1. Observe a newly completed plate appearance from the authoritative MLB feed.
2. Continue for singles, doubles, and triples. Home runs are excluded because
   they are not part of the hit-reversion thesis.
3. Require a meaningful directional fair-value move for the batting team.
4. Anchor to a fresh Kalshi execution observed before the event.
5. Compare the anchored target with exact subsequent Kalshi executions.
6. Start watching the side whose residual exceeds the configured edge.
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

The backtest uses executed trades, not reconstructed quotes. An entry requires
a strictly later observed execution on the compatible taker outcome side and
enough reported size. Therefore the simulator is causal but remains a fill
proxy rather than a historical order-book reconstruction.

The selected segmented research policy currently uses:

- singles, doubles, and triples;
- both YES- and NO-side residuals;
- separate minimum-edge, confirmation, and reversion thresholds for every
  hit-type/side pair;
- a later move back toward the target after confirmation;
- fixed maximum-payout sizing of ten contracts per entry;
- unlimited positions per game, with at least 180 seconds between entries;
- five-second maximum pre-event anchor age;
- ten-second event-to-entry deadline;
- next-pitch invalidation;
- minimum local fair move of 0.5 points.

Fees use Kalshi’s rounded taker-fee formula.

## Exit behavior

For a YES position, target reversion occurs when the observed YES price reaches
or exceeds the dynamic target. For NO, it occurs when YES falls to or below the
target. The exit also requires a strictly later compatible execution with
sufficient size.

There is no unconditional time-based exit. Optional momentum logic can delay a
reversion exit while the held-side price continues moving favorably, then exit
on velocity reversal, trailing giveback, or the momentum hold limit. Momentum
is disabled in the selected configuration. Any remaining position settles at
the final game outcome.

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
state-model feature parity, reversion exits, and momentum-delayed exits.

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

The frozen holdout begins June 28, 2026: 136 fills across 220 games (0.62 per
game), $23.05 net PnL, and 2.90% ROI. Removing the best game leaves $18.12;
removing the best four leaves $5.09. The selected pre-holdout policy produced
719 fills across 918 games (0.78 per game), $81.38 net PnL, and 2.00% ROI.
Two of three chronological tuning folds were profitable and the losing fold
was -0.56% ROI. Deployment remains enabled by the frozen holdout gate.
