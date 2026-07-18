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

## Mechanical audit and current result

The audit found and fixed a maker-price accounting error: the feature frame had
stored the observed trade price instead of the simulated resting limit for 58
of 170 candidates. Version five now has zero YES/NO entry, target, or PnL
arithmetic violations across 311 deterministic candidates.

Across 59,070 directionally comparable completed plate appearances, 894 local
fair moves (1.51%) oppose the event heuristic. There are 53 fair-transition
continuity gaps among 233,477 adjacent observed updates; these are retained in
the audit report for timestamp/cache investigation.

The transparent baseline tests 25%, 50%, 75%, and full residual contraction,
next-pitch and next-plate-appearance exits, entry-latency caps, inning caps,
residual thresholds, and separate YES/NO policies. Its selected tuning rule is
NO-only with 25% contraction or an exit at the next plate appearance. It earns
+$27.59 and 11.50% ROI across 24 trades; all three tuning folds are positive at
15.71%, 6.56%, and 11.08%. On the already-used development holdout, however, it
returns 30 trades, -$53.74, and -17.91% ROI.

After the deterministic audit, the leak-free ML policy is rebuilt on full
reversion outcomes. It returns 28 tuning trades, +$8.89, and 3.17% ROI, with
one fold at -4.43%. On holdout it returns 32 trades, -$9.53, and -2.98% ROI.
Neither policy validates, and deployment remains disabled.

The historical tape also cannot establish maker queue position or the actual
fee schedule for each traded series, so maker fills remain optimistic proxies.

## Reproduction and reports

```bash
.venv/bin/python models/analyze_state_reversion_execution.py
.venv/bin/python models/tune_state_reversion_baseline.py
.venv/bin/python backtesting/evaluate_state_reversion_baseline.py
.venv/bin/python backtesting/audit_state_reversion_mechanics.py
.venv/bin/python models/train_state_reversion_classifier.py
.venv/bin/python backtesting/evaluate_state_reversion.py
```

- `execution_latency_sensitivity.csv`
- `trade_count_roi_frontier.csv`
- `holdout_segments.csv`
- `holdout_maker_fee_sensitivity.csv`
- `holdout_trades.csv`
- `mechanics_audit_summary.json`
- `state_alignment_audit.csv`
- `arithmetic_violations.csv`
- `representative_tick_replays.csv`
- `entry_latency_diagnostics.csv`
- `deterministic_tuning_grid.csv`
- `deterministic_holdout_summary.json`
