# Repository audit — September 27, 2026

The project has useful MLB state, exact-trade acquisition, shared feeds, and
account-risk infrastructure. Its previous positive result did not establish
executable profitability. More season data is useful only after correcting the
information and execution contracts.

| Finding | Effect | Implemented change |
| --- | --- | --- |
| The shared builder shifted the next pitch's state backwards | Runner movement, score, or inning information could arrive early; research differed from atomic live play state | Score current pitch count and the completed play itself; share the terminal-play parser with live trading |
| Simulated IOC entries/exits could wait indefinitely for compatible prints | Later liquidity was treated like an immediately executable order | Bound the evidence window, freeze price and quantity, and require another decision/latency for exit retries |
| Pending orders rechecked future signals/model scores | An in-flight order could disappear after adverse information | Preserve submitted orders until fill/expiry; score at submission |
| Training used MLB pregame priors while both trading call sites used market priors | The state model saw a different prior distribution | Use the same frozen MLB rating state in the new evaluation and reversion runtime |
| Equal-time prints could be borrowed across markets | Cross-market ordering was assumed without evidence | Use strictly earlier paired observations; retain independent home/away liquidity |
| Market download exceptions could still publish a successful aggregate | Missing data could silently alter the sample | Fail incomplete acquisitions; atomically publish aggregates and final-market caches; stream bounded download batches |
| Live launcher supplied the paper override and tuners could enable live trading | Disabled policy state was bypassable through ordinary workflows | Honor disabled policies in workers/launcher and keep tuner outputs disabled |
| Positive historical reports reused development data | Reported returns were selected on the evaluation period | Preserve frozen weights/thresholds; publish all fixed scenarios and day-block uncertainty, with hashes |

Acquisition retrieved 12,962,838 non-block executions across 1,280 markets for
August 10–September 26 and 639 MLB feeds. The frozen evaluation uses August
12–September 26. It contains 610 mapped games. Four other games were excluded
because date-based doubleheader pairing was ambiguous: game IDs 823489,
823491, 824784, and 824785. Exclusions are recorded in the run manifest.

The reference result is +$6.73 across 24 simulated fills after $1.70 of fees.
Its day-block bootstrap interval is $1.20 to $13.86 under the declared proxy
assumptions. Removing the best four games leaves +$1.77. However, increasing
submission latency to 1.5 seconds gives −$0.66, increasing it to 3 seconds
gives −$6.10, and adding 2 seconds of publication delay gives −$2.45. The
positive reference result deserves prospective testing; it is not robust
evidence of live profitability. There were no cash-rejected orders in any
scenario. Maximum reference locked capital was $9.75.

Validation: 112 shared/runtime/data tests and 71 reversion tests passed;
all scenario trade CSVs reconcile to their reported PnL. Docker was not rebuilt.

See [the complete result](results/reboot/REPORT.md) and
[the fixed protocol](VALIDATION_PROTOCOL.md). Model binaries were not retrained,
and no threshold was selected from these new results. Both checked-in live
policies remain disabled; no external trading runtime was started or altered.

Remaining limits are substantive: final MLB archives can contain corrections,
publication latency is modeled from a July sample, and historical executions
cannot prove quotes/depth remained available to a new order. The existing paper
trader also uses trade-print heuristics; its paper PnL does not validate fills.
Cash admission excludes pending-order reservations and does not resimulate
replacement signals after cash rejection. These limits are reported rather
than converted into live readiness.

A new read-only recorder captures actual book snapshots/deltas, reception
clocks, connection gaps, and MLB observations. A 30-second live integration
check captured both books, 11 deltas, 3 trades, and 15 MLB observations without
errors. Its [integration summary](results/recording_smoke.json) contains no
credentials. The next empirical step is book-based shadow replay and observed
fill reconciliation before a capital-bearing deployment.
