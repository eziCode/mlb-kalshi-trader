# Event-agnostic state-reversion study

This study tests whether Kalshi overreacts to observed MLB state transitions.
It does not use event names.  Every completed pitch is eligible.

For each transition, the builder anchors to the last market trade strictly
before pitch start, applies the local model's fair log-odds change, and watches
strictly post-pitch trades for a persistent market residual.  A paper entry
requires a later compatible taker-side trade with enough reported size.  A
paper exit likewise requires residual reversion followed by a strictly later
compatible trade.  Candidates are evaluated independently; the final policy
simulation then enforces one chronological position per game.

## Chronology

- Model fit: before June 17, 2026
- Platt calibration: June 17-21
- Threshold tuning: June 22-27
- Outer holdout: June 28 onward

The two-stage policy predicts whether the complete executable trade has
positive net PnL, then separately estimates conditional winning and losing PnL.
Platt-calibrated profitability and the two magnitude estimates are combined
into expected PnL. This counts profitable stops and timeouts correctly instead
of treating every non-reversion exit as a classification failure.

## Current result

The rebuilt dataset contains 17,846 fit, 2,876 calibration, 3,578 tuning, and
7,570 holdout candidates. Tuning profitability AUC is 0.672 and predicted PnL
has 0.427 correlation with realized PnL. The activity frontier is profitable
only below 20 tuning trades. Its best configuration with at least 20 trades
returned 24 trades, -$0.37, and -0.15% ROI. On the untouched holdout it returned
58 trades, -$21.62, and -3.65% ROI. The configuration remains disabled.

The complete activity/return curve is in `trade_count_roi_frontier.csv`.

Reproduce with:

```bash
.venv/bin/python models/train_state_reversion_classifier.py
.venv/bin/python backtesting/evaluate_state_reversion.py
```

`overshoot_candidates.parquet` is generated from the ignored exact-timestamp
trade-tape inputs and cached locally for repeatable model experiments.
