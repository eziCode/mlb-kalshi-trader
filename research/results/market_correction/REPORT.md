# Frozen market-correction candidate

`corrected_01c` has positive simulated net PnL in development, selection,
and the subsequent chronological check. It is a research candidate, not a
verified profitable bot. The every-game activity target remains unmet.

| Period | Mapped games | Entry orders filled | Net PnL |
|---|---:|---:|---:|
| Development, August 12–September 1 | 283 | 147 | $1.93 |
| Selection, September 2–15 | 186 | 89 | $1.95 |
| Chronological check, September 16–26 | 141 | 73 | $2.26 |
| Combined shared account | 610 | 309 | $6.13 |

One contract per entry. All 309 positions settled using the exchange's
recorded timestamp and payout, with sixty extra seconds before proceeds
became available. The combined account starts at $100 and ends at $106.134;
maximum reserved plus invested cost is $5.5566. There were 8,073 submitted
simulated attempts, of which 7,764 expired. Attempts are not counted as trades.
Fees total $3.776. Four ambiguous source doubleheader games remain excluded.

Coverage is 50.7%, or 0.51 filled entry orders per mapped game. The descriptive
95% day-resampling interval for combined PnL is **−$9.18 to +$20.56** and
does not include model-selection or execution-model error. These historical
dates were already examined in previous strategy research, so the later
check is not described as an untouched holdout.

The candidate adds a small logistic correction to market probability using
market confidence, disagreement with the updated baseball model, inning,
home-team indicator, and paired-market prices. Fit and feature scaling use
only development games. A fixed penalty controls model size. The one-cent
entry threshold was chosen from the declared candidates using development
and selection, then frozen before scoring the later check. All six candidates,
including the unsuccessful pregame-anchor variants, are retained in
[summary.json](summary.json).

| Later-check execution scenario | Filled entries | Net PnL |
|---|---:|---:|
| 680ms delay, 1¢ penalty | 73 | $2.26 |
| 1.5-second delay | 72 | **−$1.55** |
| 3-second delay | 66 | $3.31 |
| 2¢ penalty | 35 | $4.04 |
| Publication delayed another 2 seconds | 73 | $2.43 |
| 1% printed-volume participation | 53 | $0.46 |
| 50ms execution-evidence window | 42 | $2.24 |

Every stress uses the frozen coefficients and threshold. Changing execution
also changes which orders fill, so these rows are not the same trades with
costs subtracted. The 1.5-second loss and wide uncertainty prevent accepting
this as a robust edge. The source used for the stress run is retained with
its original manifest; a later guard additionally verifies that a supplied
model directory matches the frozen model hashes.

Historical fills use bounded, delayed, side-specific public trade evidence.
They do not establish that actual quotes or depth were executable. The
[prospective protocol](../../FORWARD_VALUE_PROTOCOL.md) therefore uses recorded
books and MLB observations, still as a shadow test with no actual orders.
It reports open inventory against the correct home or away market and
preserves partially filled or pending orders at capture boundaries. The
September 27 recorder and periodic replay are collecting that next evidence.

No live policy has been enabled. Positive historical PnL is not represented
as realized account profit or evidence that scaling trade size will work.
