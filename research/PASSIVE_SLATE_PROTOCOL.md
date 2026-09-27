# Passive slate experiment, declared September 27, 2026

This experiment uses prospective observations. The initial 20/30-second
captures are integration checks only. A six-hour, read-only capture of all
15 discovered games began at approximately 16:59 UTC on September 27.

The next research question is whether passive entries retain enough value
after queue waiting, maker fees, adverse selection, and inventory exits.
No actual orders are placed. No backtest or shadow PnL is called live profit.

Before inspecting today's strategy PnL, declare twelve candidates:

* Opening spread: one cent, joining the bid; or two cents, improving by one
  cent only when the spread reaches three cents.
* Opening price filter: none; own-book microprice; or the other team's book.
  Filtered prices require at least half a cent of estimated value after the
  opening maker fee. This estimate is a hypothesis, not a learned probability.
* Entry regime: all observed live play; or the first 45 seconds of an observed
  middle/end-of-inning break. Pregame and final games cannot open positions.

These choices form 2 × 3 × 2 candidates. All use one contract, sixty-second
quote lifetime, a one-second decision clock, 680ms submission/cancel latency,
the correct KXMLBGAME maker fee rate 0.0175, and a thirty-second inventory
limit. Expired inventory first cancels outstanding quotes, waits through
cancellation, then submits bounded IOC liquidation against subsequently
observed depth with a one-cent adverse execution penalty and taker fees.
Failed/partial liquidations retry with fresh decisions and latency. Closing
orders may not silently increase inventory. Existing quotes remain exposed
through cancellation delay and adverse trade-throughs.

Microprice weights the best bid/ask by opposite-side displayed size. The peer
reference is one minus the other team's midpoint. Both use only observations
already received; exchange event timestamps also gate trade-fill evidence.
No signal may erase a previously submitted order or its adverse fill.

Queue ahead is all displayed size at modeled arrival. Decreasing displayed
size does not automatically improve queue position. Compatible trades consume
the queue; a trade through our price cannot be ignored. Partial fills,
pending cash, maker/taker fees, and open inventory must be retained.

Every result reports unique entry orders, fill messages, contracts, games
traded, queue wait, cancellation-race fills, 5/30/60-second subsequent marks,
forced-exit counts, fees, realized PnL, and inventory-inclusive liquidation
marks. Incomplete captures and sequence gaps are explicitly labeled and may
not be presented as completed-game performance.

Today's first complete slate is development evidence for these candidates.
Selection must be frozen before any subsequent validation game is examined.
No positive candidate is accepted solely because it is the best of twelve.
Require positive inventory-inclusive net results on later games and stress
submission/cancellation delays to 1.5/3 seconds, plus an extra cent of
liquidation cost. Report zero-trade and losing games. Track the user's
every-game activity target separately from average entries per game.

Initial individual-market replays are diagnostics, not a combined bankroll.
A selected policy must also pass a shared chronological portfolio replay
before reporting portfolio returns. Actual order reconciliation remains
necessary to validate modeled fills. No hidden queue, spread touch, terminal
outcome, or unobserved cancellation is assumed to make a trade profitable.
