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

The frozen policy trades both settlement directions: home signals buy home
YES, while away signals buy the independently traded paired away-YES contract.
It does not cap the number of $10 positions per game and requires at least 200
seconds between fills. Candidate thresholds must trade both directions, be profitable in both
chronological policy periods, and remain profitable after removing their best
game.

The selected development policy produced 30 fills, $48.44 net PnL, and 15.68%
ROI across its two policy-validation periods. The later development holdout
produced 53 fills, $66.66 net PnL, and 12.19% ROI. Removing the four best games
leaves a small positive result, but statistical uncertainty remains high, so
validation and deployment remain disabled.

This is development validation rather than a pristine final test. Deployment
remains disabled pending forward paper performance on newly collected games.
