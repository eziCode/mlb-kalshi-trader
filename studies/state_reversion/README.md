# Event-agnostic state-reversion study

This study tests whether Kalshi overreacts to observed MLB state transitions.
Event names are never model features. Version three admits only a genuine
same-direction overshoot: the market and local fair log-odds must move in the
same direction, and the market move must exceed the fair move by a configured
margin. Underreactions and generic market/model disagreement are excluded.

The market anchor is the last trade before pitch end minus a two-second safety
buffer, must be no more than five seconds old, and is sensitivity-tested with
one-, two-, and three-second buffers. Each candidate requires a meaningful
fair move, one second of persistent excess movement, and at least $0.25 of
modeled target profit after configured fees.

The selected execution proxy uses maker entry and exit. A resting order fills
at its limit only after a strictly later opposite-taker trade reaches that
price with sufficient reported size. Queue position is unavailable, so these
remain optimistic fill proxies. The configured maker fee is zero; separate
holdout sensitivity reports show results under hypothetical nonzero rates.

Positions exit on executable residual reversion, a 120-second opportunity
timeout, settlement, or when subsequent local fair value moves materially
against the held contract. Residual expansion by itself is not a stop.

## Chronology

- Model fit: before June 17, 2026
- Platt and EV calibration: June 17-21
- Threshold tuning: June 22-27
- Outer holdout: June 28 onward

Selection requires at least 20 tuning trades, at least three trades in each of
three chronological tuning folds, and positive PnL in all folds. It maximizes
the worst fold ROI without using the outer holdout.

## Current result

After removing a leaked feature that had incorporated realized future PnL, no
configuration passed the three-fold stability rule. The best aggregate tuning
configuration produced 25 trades, +$4.49 PnL, and 1.80% ROI, but one of its
three folds lost 4.46%.

On the untouched holdout that frozen fallback produced 40 trades, -$5.21 PnL,
and -1.30% ROI before any maker fees. The strict strategy is a substantial
improvement over the earlier mixed-disagreement implementation, but it has not
validated positive returns. Deployment remains disabled.

The historical tape also cannot establish maker queue position or the actual
fee schedule for each traded series, so maker fills remain optimistic proxies.

## Reproduction and reports

```bash
.venv/bin/python models/analyze_state_reversion_execution.py
.venv/bin/python models/train_state_reversion_classifier.py
.venv/bin/python backtesting/evaluate_state_reversion.py
```

- `execution_latency_sensitivity.csv`
- `trade_count_roi_frontier.csv`
- `holdout_segments.csv`
- `holdout_maker_fee_sensitivity.csv`
- `holdout_trades.csv`
