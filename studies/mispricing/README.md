# Event-agnostic settlement-value strategy

This system treats Kalshi as a market prior and predicts calibrated home-team
settlement probability from the observed market price, local state-model
movement, raw post-transition state, state deltas, and recent trade flow. Event
names such as single, double, walk, and strikeout are excluded from the feature
contract.

At each safely observed state update, it compares calibrated fair probability
with the executable value of YES and NO. A signal must retain its edge at a
strictly later compatible taker-side trade with sufficient reported size. It
takes at most one $10 position per game and holds to settlement.

## Chronology

- Fit: before June 17, 2026
- Probability calibration: June 17-21
- Threshold and side selection: June 22-27
- Development holdout: June 28 onward

The selected policy is NO-only, requires at least ten percentage points of
probability edge and $2 of predicted net settlement value after the entry fee.

## Results

Tuning produced 20 trades, +$171.16 PnL, and 82.37% ROI. All three chronological
folds were profitable, with a worst-fold ROI of 56.10%.

The frozen policy produced 44 trades across 44 games on the development
holdout, +$75.97 PnL, and 16.68% ROI after $15.38 of fees. Win rate was 61.36%,
the largest winner contributed 23.84% of total PnL, and contract prices ranged
from 25 to 74 cents. The model's holdout ROC AUC was 0.870 and Brier score was
0.148.

This holdout has been inspected during earlier strategy development, so the
result is validation evidence rather than a pristine final test. Configuration
remains disabled until it succeeds in forward paper trading on newly collected
games.

```bash
.venv/bin/python models/train_mispricing_model.py
.venv/bin/python backtesting/evaluate_mispricing_model.py
```
