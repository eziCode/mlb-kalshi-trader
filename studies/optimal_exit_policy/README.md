# Fitted optimal-stopping exit policy

This policy replaces fixed loss and elapsed-time stops with a learned
continuation-value decision:

```text
sell when executable liquidation value
          >= conservative continuation value + safety margin
```

The continuation model is a CatBoost quantile regressor trained through five
rounds of fitted policy iteration. Its 23 features include the model-adjusted
residual, liquidation return, price momentum and volatility, spread proxy,
elapsed time as a predictor, and current MLB game state. Elapsed time never
causes an exit by itself.

Every sell decision must remain valid for five seconds and then execute on a
later compatible trade-tape observation. The model was fitted on games before
June 17, tuned on June 17–27, and evaluated on games beginning June 28.

## Results

- Selected validation policy: 3% continuation-value margin, five-second
  confirmation, +39.21% ROI on 88 accepted entries with 20 learned exits.
- June 28–July 10 holdout: −5.32% ROI, −$55.55 PnL on 101 accepted entries.
- The 10 learned exits earned +$72.75, while the 91 continued positions lost
  −$128.29 at settlement.

The policy did not generalize, so `optimal_exit_config.json` contains
`"enabled": false`. It can only be exercised by the live paper trader through
the explicit unvalidated-policy override.

Machine-readable outputs are in `training_summary.json`,
`validation_grid.csv`, `holdout_summary.json`, and
`holdout_policy_trades.csv`.
