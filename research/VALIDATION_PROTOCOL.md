# Reboot validation protocol — 2026-09-27

The existing policy was developed through August 11 using data through August 9.
Its reported +$24.68 is not an untouched holdout. Local live logs continue into
September, so the later period is described as a chronological diagnostic,
not automatically as a blind test.

Keep the packaged model weights, event filters, edge thresholds, and July 20
MLB rating state fixed. Evaluate August 12–September 26 after correcting the
state and execution contracts. Do not optimize parameters on this period.
The seven execution scenarios in `reboot.py` are a fixed sensitivity grid;
report every scenario, including zero trades and negative results.

The reference assumes 0.68 seconds of submission latency, a 250 ms evidence
window, a one-cent adverse price adjustment on each side, 10% participation in
the compatible printed quantity, and $100 starting cash. These are declared
stress assumptions, not measured historical quotes or guaranteed fills.
Alternative scenarios independently increase latency to 1.5 or 3 seconds,
increase the price adjustment to two cents, reduce participation to 1%, reduce
the evidence window to 50 ms, or add two seconds of publication delay.

Orders freeze price and quantity before later prints. An in-flight order cannot
be cancelled by future baseball state or a future model score. Each entry event
gets one submission in this diagnostic. Exit retries require a fresh decision
and another submission delay. Partial exits reserve the remaining position.
No entry or exit may borrow liquidity from the independently traded other-team
contract. Settlement cash is not recycled without exchange payout timestamps.

Write model, policy, source, and data hashes; report cash rejections, fees,
concentration, outcome-attributed drawdown, and bootstrap intervals resampling
whole calendar days including zero-trade days. The interval quantifies sample
variation under the assumptions; it does not measure fill-model error or undo
strategy selection. Outcome-day drawdown is not intraday marked-to-market risk.

Historical tape results never enable real-money deployment. Deployment requires
prospective observations of executable quotes/depth and measured reception/order
timing, followed by actual fill reconciliation. The checked-in reversion policy
is disabled because its earlier enablement was based on the invalidated research
contract. Paper research remains available; no running external service is changed.
