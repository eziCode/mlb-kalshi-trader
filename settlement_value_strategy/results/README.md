# Settlement-value study results

The strategy predicts calibrated home-team settlement probability without
using event-name labels. Signals must retain their expected value at a
strictly later, compatible execution with sufficient reported size. Positions
are $10 and are held to settlement.

## Chronology

- Model fit: before June 17, 2026
- Calibration: June 17-21
- Policy tuning: June 22-27
- Outer development holdout: June 28 onward

## Reproduce results

From the repository root:

```bash
.venv/bin/python setup_data.py mispricing
.venv/bin/python -m settlement_value_strategy.prepare_data
.venv/bin/python -m settlement_value_strategy.train
.venv/bin/python -m settlement_value_strategy.backtest
```

`train` rewrites the model, calibration, policy configuration, tuning grid,
and training summary. `backtest` rewrites holdout summaries and trade logs.

## Frozen reference

The frozen policy is NO-only and requires at least ten percentage points of
edge and $2 of predicted net settlement value. Its recorded holdout contains
44 trades, $75.97 net PnL, 16.68% ROI, ROC AUC 0.870, and Brier score 0.148.

This is development validation rather than a pristine final test. Deployment
remains disabled pending forward paper performance on newly collected games.
