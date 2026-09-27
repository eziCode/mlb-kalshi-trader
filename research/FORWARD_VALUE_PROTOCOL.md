# Prospective frozen value policy, declared September 27, 2026

The market correction selected `corrected_01c` using development/selection
data. It made small positive historical profits in each period but lost under
one latency stress and does not meet the every-game activity target. Do not
change its model coefficients or threshold after viewing this forward test.

Replay September 27 observations with the same one-cent entry edge, one-contract
size, five-second decision clock, 680ms submission delay, one-cent adverse
execution penalty, innings 1–7, and maximum one filled entry order per game.
Use one $100 account across games. Submit bounded shadow IOC orders against
the actually observed home/away books. Walk displayed ask depth at modeled
arrival; honor partial fills and misses, preserve orders through intervening
adverse information, and return only unused reserved cash. These remain
modeled orders, never actual exchange fills.

The frozen probability model consumes fields already present together in a
received MLB linescore: inning/half, count, runs, and occupied bases. Require
consistent batting-team identity and valid counts. Incomplete balls in play
invalidate new signals; submitted orders remain exposed. A poll heartbeat
can refresh feed availability but cannot invent a changed state. The first
pitch must have been observed so the pregame anchor can be fixed from an
already received trade strictly earlier than that pitch, at most five minutes
old. Missing first-pitch history or an anchor disables entry for that game.

This prospective feed path and actual book liquidity differ from historical
pitch archives and trade-print proxies. Report that difference explicitly;
do not label agreement as measured execution. Models and source hashes are
captured before replay. The existing twelve passive variants remain a separate
development experiment; do not select this value policy using their results.

Hold acquired contracts. Until exchange settlement has actually been recorded,
inventory stays in the account and is marked against its own market's visible
bids after taker fees. Missing depth contributes zero to that conservative
mark. Report realized PnL, marked PnL, outstanding inventory, all slate games,
games observed live/final, entry orders, quantities, misses, fees, and coverage.
An ended recorder is not a settled portfolio. Sequence gaps invalidate the
subsequent path and do not silently erase existing positions.
