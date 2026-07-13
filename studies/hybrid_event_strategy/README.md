# Hybrid hit-event residual study

This study tests whether Kalshi's post-hit move exceeds the move implied by a
locally trained win-expectancy model.

For each isolated completed hit, the expected post-event market target is:

```text
logit(target) = logit(pre-event Kalshi midpoint)
              + logit(post-event local fair)
              - logit(pre-event local fair)
```

This uses the model's relative state update while retaining Kalshi's own
absolute pre-event calibration. A trade is considered only when the next
observed bid/ask still differs from the target by the configured threshold.
While a position is open, later game states move the target by their local
fair-odds change; they do not force an automatic exit. Fees are charged on
entry and exit.

Historical event outcomes are delayed until the next pitch start because the
available MLB timestamps do not include pitch completion. Candles containing
multiple completed plate appearances are excluded.

## Results

- Training-period selection: best eligible configuration was a 4% minimum
  edge and eight-minute maximum hold, returning −10.73% ROI on 407 trades.
- Outer holdout (June 28–July 10): −9.93% ROI, −$473.88 PnL on 461 trades,
  with $311.13 in fees.
- Because training-period performance was negative, `hybrid_config.json`
  contains `"enabled": false` and the paper trader fails closed by default.

See `tuning_grid.csv`, `tuning_summary.json`, and `holdout_summary.json` for
the machine-readable outputs.
