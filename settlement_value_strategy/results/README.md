# Settlement-value study results

The strategy predicts calibrated home-team settlement probability without
using event-name labels. Signals must retain their expected value at a
strictly later, compatible execution with sufficient reported size. Positions
are $10 and are held to settlement.

## Chronology

- Model fit: before June 17, 2026
- First policy-validation period: June 17-21
- Second policy-validation period: June 22-27
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

The frozen policy takes only an away-team settlement view and routes it to the
independently traded away-YES contract. It uses at most one $10 position per
game. Candidate thresholds must be profitable in both chronological policy
periods and remain profitable after removing their best game.

The selected development policy produced 115 fills, $111.07 net PnL, and
9.37% ROI across its two policy-validation periods. The later development
holdout produced 180 fills, $203.88 net PnL, and 10.99% ROI. Removing the four
best holdout games leaves $119.91, and 70.6% of holdout days were profitable.

This is development validation rather than a pristine final test. Deployment
remains disabled pending forward paper performance on newly collected games.
