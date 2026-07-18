# Residual-path ML policy

This pipeline separates event-agnostic alpha, execution, and action selection.
It builds one example for every persistent same-direction market overshoot,
whether or not a simulated order fills. Event names are excluded from the
feature contract.

- Alpha models predict residual contraction at 2, 5, 10, 30, and 60 seconds.
- A separately calibrated classifier predicts maker-fill probability.
- Conditional-PnL models learn horizon outcomes only for observed maker fills.
- The policy chooses a horizon or no trade from predicted fill-adjusted value.

Chronology is unchanged: fit before June 17, calibration June 17-21, tuning
June 22-27, and development holdout June 28 onward.

## Result

The dataset contains 2,239 fit, 381 calibration, 475 tuning, and 1,008 holdout
signals. Maker-fill prediction generalizes reasonably (0.753 tuning AUC), but
alpha does not: contraction correlations range from -0.046 to 0.120. Average
residual contraction is slightly positive only at two seconds and becomes
negative at longer horizons.

The best tuning policy with at least 20 trades uses a ten-second horizon and
loses 1.61% across 70 fills; every chronological fold is negative. On the
development holdout it loses 1.39% across 165 fills. Deployment is disabled.

```bash
.venv/bin/python models/train_residual_path_policy.py
.venv/bin/python backtesting/evaluate_residual_path_policy.py
```
