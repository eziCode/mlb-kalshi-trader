# Continuous strategy search — 2026-09-27

Purpose: search for an edge with at least one entry per game on average, and
report the fraction of games actually traded separately. Forcing a losing
entry in every game is not a profitability criterion. All policies remain
research-only; historical prints cannot validate executable books or fills.

This period was already used for the old strategy diagnostic. These are
chronological development/selection/check partitions, **not a new untouched
holdout**. This search uses no final-partition outcomes to select its candidate.

* Development/training: August 12–September 1 (283 games).
* Selection: September 2–15 (186 games).
* Final check: September 16–26 (141 games).

Declare the candidate grid before running it: 30/120-second price reversion,
30/120-second momentum, cross-market relative value, absolute state-model
value, market-anchored responses to every completed play, and learned future
bid changes. Each has 60/300-second holds and 0/1/3-cent minimum expected net
edges: 48 candidates. Two forced-activity controls are reported and cannot be
selected. The learned models use only development data, fixed depth/iterations,
explicit causal numeric features, and game-balanced sample weights. No game
ID, date, winner, future liquidity, or future features enter a predictor.

Decide every five seconds. Use only strictly earlier observations. Same-side
last prints are **price proxies**, not quotes; both sides must be <=10 seconds
old and noncrossed, with <=8-cent implied spread. Enter only at prices 5–95¢,
through inning seven, with a <=120-second-old observed state. Terminal states
receive the existing publication-delay distribution plus two seconds;
nonterminal states also receive two seconds. Never roll back to an older pitch
when a delayed message arrives. State value uses the frozen rating prior;
all-play value uses only a completed play's pre-start market anchor and its
atomic model update. Both team-YES markets have independent price/liquidity.

Use one contract per position, at most one open position per game. Orders freeze
price and size at decision time. Reference submission latency is 0.68 seconds,
with a 250ms evidence window. Only a same-market, compatible-side print inside
that window, with ten contracts of observed volume per simulated contract,
can support a proxy fill. Add one adverse cent to each buy/subtract one cent
from each sale, within the frozen limit. No later favorable observation can
cancel an in-flight order. No resting/maker fills are inferred from prints.

Exit after the fixed hold, retrying only on fresh five-second decisions and
with another submission delay. Unliquidated positions settle; no settlement
cash is recycled without payment timestamps. Globally enforce $100 starting
cash and reserve cash for pending submissions, process all games in time order,
and do not reuse equal-time proceeds. There is no leverage or simultaneous
opposite position in one game. Retries and future entry decisions follow the
actual accepted portfolio path, including rejected orders.

Selection: require positive development and selection PnL, >=1 entry/game and
>=80% game coverage in each partition. Rank eligible candidates by selection
day-bootstrap lower bound. If none qualifies, freeze the best selection lower
bound among candidates meeting the activity requirements, or among all
candidates if none meets them; explicitly mark it as a failed research
candidate. Never reinterpret that fallback as a passing strategy.

Write the selection decision and all development/selection results before
accessing final-check outcomes. Evaluate only the frozen candidate in the
final period, including 1.5s and 3s submission latency, two-cent adverse costs,
1% participation, 50ms evidence windows, and two extra seconds of state delay.
Report every stress, frequency, coverage, fees, cash rejects, settlement share,
day-block uncertainty, concentration, and markout prediction errors. Intervals
are descriptive and do not correct the 48-way search or fill-proxy error.

A positive result does not enable deployment. A book-based prospective shadow
test and reconciliation against real executions remain separate requirements.
