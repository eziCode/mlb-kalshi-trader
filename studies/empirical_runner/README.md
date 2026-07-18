# Partial-reversion runner study

This study tests whether keeping part of a successful empirical-reversion
position open improves the strategy. It uses only 2026 data.

## Policy

At the first executable reversion, the simulator sells the smallest number of
contracts whose proceeds, net of the partial-exit fee, recover the original
position cost and entry fee. It retains the remaining contracts as a runner.

The runner exits on the first causally fillable trigger:

- a second price target;
- an activated trailing giveback from its post-reversion high-water mark;
- an adverse locally calibrated win-probability move on a later completed
  pitch; or
- settlement.

Triggers based on a trade cannot fill on that same trade. Every exit requires
a strictly later trade on the compatible aggressor side with enough reported
size. There is no elapsed-time exit. Another position in the same game cannot
open until the runner exits.

## Chronology

- Runner policy and entry thresholds tuned: June 17–27, 2026.
- Outer holdout evaluated once: June 28–July 10, 2026.
- Selection required at least 30 positions and 15 scaled reversions.

## Result

The selected tuning policy was already negative:

- Runner: 80 trades, −$12.91, −1.59% ROI.
- Full exit at first reversion with the same entry thresholds: 80 trades,
  −$12.00, −1.48% ROI.

On the outer holdout:

- Runner: 101 trades, −$59.05, −5.77% ROI.
- Full-exit baseline: 105 trades, −$57.72, −5.43% ROI.

The runner therefore reduced holdout PnL by $1.33. It scaled 76 holdout
reversions; 69 runners exited through the trailing rule, five at the second
target, one on adverse state, and one at settlement. After recovering capital,
the retained runner averaged only about 2% of the original contracts. That was
too small to offset the eight full positions that never reverted and lost
$81.13 at settlement.

The saved configuration remains `enabled: false`. Partial exits do not repair
the strategy's main problem: rare full-position settlement losses dominate the
small profits earned after successful reversions.

See `summary.json`, `tuning_grid.csv`, and `holdout_trades.csv` for the exact
artifacts.
