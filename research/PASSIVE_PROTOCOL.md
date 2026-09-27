# Passive inventory experiment

The continuous taker search failed its profit/frequency requirements. The
next implemented experiment quotes passively inside or at the edge of a
single game's home YES/NO book. It has **not** demonstrated a profitable
strategy or one entry per game. A 20-second pregame integration capture is
plumbing verification only.

Default rules in `passive.py`:

* Observe sequenced book snapshots, deltas, and public trades from `record.py`.
* Evaluate once per second; require a two-cent spread for opening exposure.
* Join the best bid on each side; improve it by one cent when the spread is at
  least three cents. Submit post-only; reject a quote that crosses at arrival.
* Quote one contract. After acquiring inventory, quote only its reducing side.
  Existing orders remain exposed until their delayed cancellations arrive.
* Use 680ms submission and cancellation delays and a ten-second quote lifetime.
  Timers run even between messages. Repricing loses queue priority. There is
  no instantaneous cancel/replace. Late-received trades executed before the
  modeled arrival cannot fill the new quote.
* Queue ahead starts at all displayed volume at the limit when the order
  arrives. Public trades of the opposite taker side consume that queue before
  filling the shadow order. Book cancellations never improve our queue estimate.
* Include adverse trades through the limit; a pending cancel cannot erase such
  a fill. Ignore duplicate IDs and block trades.
* Reserve pending-order cash, track partial fills and signed YES/NO inventory,
  and reconcile closing cash. Opening maker fees default to zero under the
  current standard series schedule; the CLI accepts a different maker fee rate.
* Report both realized PnL and the value of remaining inventory sold into
  displayed bids after taker fees. Missing bid depth contributes zero to that
  mark. Unpaired inventory is never excluded from the result.
* Stop at sequence/connection gaps; do not fabricate cancellation confirmation
  or resume an old queue. Exposed quotes remain reported at capture end.

This is a shadow simulation. Local reception is not exchange processing time;
the inserted order would change the real book; hidden queue and cancellation
semantics remain unverified. The inventory mark is an observed-book diagnostic,
not a completed liquidation. Maker fees and contract rules must match the
specific series/date when moving beyond this research.

Run the default policy unchanged on complete future slates before any tuning.
Report all markets, including zero fills and losing inventory, and compare
680ms/1.5s/3s delays, joining versus improving, and applicable maker fees.
Partition new observations chronologically before selecting variants. Measure
entries/game, fraction of games traded, queue waits, cancellation-race fills,
inventory duration, markouts after each fill, and inventory-inclusive PnL.
Only actual order/fill reconciliation can validate the shadow fill assumptions.

```bash
# Public discovery requires no credentials.
.venv/bin/python -m research.forward --date 2026-09-27 --discover-only

# Set the existing KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH environment
# variables. Capture both team books and MLB for the slate; submit no orders.
.venv/bin/python -m research.forward --date 2026-09-27 --duration-seconds 14400

# Re-evaluate one captured home market under another declared latency.
.venv/bin/python -m research.passive PATH_TO_CAPTURE.jsonl.gz \
  --ticker HOME_MARKET_TICKER --latency 1.5 --output NEW_REPORT.json
```

`forward` saves the slate mapping, capture, and per-game reports under the
ignored `data/raw/forward_observations/` directory. Each replay has its own
shadow account; its summary deliberately makes no combined portfolio-return
claim. No recorder or background service is left running by the integration
check. The live policies remain disabled.

References: [Kalshi book updates](https://docs.kalshi.com/websockets/orderbook-updates),
[public trades](https://docs.kalshi.com/websockets/public-trades),
[fee schedule](https://kalshi.com/docs/kalshi-fee-schedule.pdf).
