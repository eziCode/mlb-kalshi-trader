# Selective maker v1 and independent sportsbook anchor

Declared September 27, 2026. This is a separate experiment. The earlier value,
maker-hold, and twelve passive candidates remain unchanged. Today's data are
development. Validation rejects captures preceding protocol freeze and games
on or before its Eastern calendar date. It does not automatically certify that
an otherwise eligible dataset has never been viewed by a researcher.

Freeze the implementation, configurations and optional baseball model before
evaluation. Report all three declared candidates at 680ms, 1.5s and 3s delays:

- `selective_live`: join bids when current game state is complete, both books
  are fresh, their probabilities agree, recent movement is limited, the queue
  is bounded and estimated value clears entry/exit maker fees plus 0.5 cents.
- `selective_break`: the same policy restricted to already observed inning
  breaks, using the source break clock rather than restarting it on receipt.
- `sportsbook_anchor`: shift the frozen baseball model's initial log odds by
  a three-book consensus known strictly before the observed first pitch.
  Missing pregame quotes or model disable entries; do not substitute Kalshi
  prices. Retain the same execution and book-quality filters.

Defaults: one contract per quote; one-second decisions; ten-second quote
lifetime; thirty-second maximum inventory age; five seconds before reentry
after flattening; twenty opening orders filled per game; stop new entries
after three dollars of realized game losses. Pending orders remain exposed
through cancellation even when they subsequently breach an entry target.
Inventory is limited by one quote per side and no new same-direction opening
quote while holding inventory. Partial cancel-race reversals still count.

Require book updates within thirty seconds, peer midpoint disagreement at
most three cents, midpoint range at most three cents over ten seconds, and
at most fifty displayed contracts at the chosen price. Decision prices stay
immutable through submission latency. Joining the queue starts behind all
displayed quantity; only compatible public trades consume it. Cancellations
of displayed depth give no queue credit. Adverse trade-throughs can fill.

Reducing quotes remain available when an opening filter fails. On inventory
timeout, cancel resting orders, wait for acknowledgement latency, then submit
a bounded taker liquidation with a one-cent adverse execution penalty. Charge
maker fees to passive fills and taker fees to liquidation. Do not reuse consumed
unchanged depth. Missing/stale books stop new quotes and request cancellation;
already submitted orders remain exposed. Baseball entries use innings 1–7.

Use a single $100 account per candidate/scenario. Positions in the home market's
YES and NO outcomes share that market's book; the other team's market supplies
reference information only. Fee-inclusive reserves, partial fills, exits,
and net inventory reconcile against an independent fill ledger. Later observed
exchange finalization may replace remaining inventory marks; settlement cash
is not recycled into earlier orders. MLB final scores do not create payouts.

The sportsbook adapter uses The Odds API's documented full-game h2h endpoint:
https://the-odds-api.com/liveapi/guides/v4/

Record request, receipt, journal and bookmaker/market update timestamps.
Fresh fetches do not refresh unchanged source quotes. Require both outcomes,
unique bookmakers, plausible margin, mutually unique team/date/start matching,
and at least three books at most 180 seconds old with probability range at
most three points. Normalize implied probabilities within each book, then use
the median home probability. A full new snapshot removes omitted events and
books. A raw JSON import is received now and cannot fabricate past availability.
No sportsbook execution is assumed. Request counts and remaining-quota floors
bound collection; credentials never appear in recorded errors.

Keep 5/30/60-second post-fill diagnostics separate from decision inputs. Report
spread capture, subsequent midpoint movement, and fee-adjusted depth-walk marks.
Insufficient depth contributes zero proceeds for that part of a conservative
mark, with missing quantity reported; it does not constitute an actual exit.
Unobserved horizons remain missing. Cluster descriptive uncertainty by game,
requiring at least ten games and retaining the selection/fill-model caveat.

Reports include all scheduled/mapped games, zero-entry games, attempts, unique
opening orders filled, contracts, fees, cash committed, open inventory, unsettled
marks, observed settlement PnL, and a drawdown sampled at sixty-second intervals.
That sampled drawdown is not the maximum over every intrasecond price movement.
Capture completeness, final-game coverage, and pending orders are separate.

The target is at least one entry in every eligible game with positive net
performance robust to execution assumptions. No candidate is promoted
automatically. These programs contain no exchange order-submission path.
