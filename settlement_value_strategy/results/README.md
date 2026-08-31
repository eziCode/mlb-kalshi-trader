# Settlement-value research results

This directory preserves the settlement-value research trail. The final
corrected result is negative, and the checked-in live policy is disabled.

## Final decision

| Evaluation | Games | Fills | Net PnL | Capital | ROI |
| --- | ---: | ---: | ---: | ---: | ---: |
| Seven-fold expanding-window replay | 1,440 | 213 | -$15.70 | $466.78 | -3.36% |
| Post-training slice, 2026-07-24 to 2026-08-09 | 234 | 41 | -$4.23 | $97.77 | -4.32% |

Only two of seven chronological folds were profitable. Removing the best four
games from the post-training slice produces -$16.95. The authoritative policy
file therefore records `enabled: false`, `tuning_passed: false`, and
`validation_passed: false`.

This corrected replay includes the measured 4.72-second submission latency,
requires a strictly later compatible execution within five seconds, caps a
fill by printed size, routes away exposure through the independent paired
away-YES market, charges Kalshi fees, and uses a $2.50 order budget. Execution
prints are still a fill proxy rather than a complete reconstruction of book
depth and queue position.

## Artifact guide

- `live_policy_backtest_summary.json` and
  `live_policy_backtest_trades.csv`: authoritative corrected replay.
- `latency_research_summary.json`: model and policy search results before the
  exact frozen-policy correction.
- `early_exit_research_summary.json`: research-only stop-loss overlay; early
  exits remain disabled.
- `holdout_summary.json` and related CSVs: earlier $10 settlement study,
  retained for reproducibility but superseded as a deployment claim.
- `training_summary.json` and `tuning_grid.csv`: earlier model-training audit
  trail.

## Reproduce

From the repository root:

```bash
.venv/bin/python setup_data.py mispricing
.venv/bin/python -m settlement_value_strategy.prepare_data
.venv/bin/python -m settlement_value_strategy.train_latency
.venv/bin/python -m settlement_value_strategy.research_latency
.venv/bin/python -m settlement_value_strategy.backtest_live_policy
```

These commands rewrite their corresponding model, configuration, summary, and
trade artifacts. Retrain after regenerating shared data because the frozen
model is tied to its causal feature and anchor contract.
