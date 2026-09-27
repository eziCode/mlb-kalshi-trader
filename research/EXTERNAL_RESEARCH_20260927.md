# External strategy evidence and next experiments

Reviewed September 27, 2026. This is an evidence review and research plan,
not a change to the frozen value or maker experiments. Their source hashes,
parameters, and prospective results retain their original meaning.

Related approaches have made money in author-reported live deployments.
This search did not establish an independently audited, reproducible MLB
Kalshi bot that meets our every-game activity target after current costs.
Public results supply hypotheses; they do not validate our implementation.

## Evidence reviewed

| Source | What it establishes | Limits and relevance |
|---|---|---|
| [McDermott's MLB Kalshi bot](https://github.com/william-mcdermott/kalshi-trading-bot/blob/main/README.md) | Reports 201 Kelly-sized signals, +30.5% net backtest ROI; combines pregame odds, game state, and pitcher adjustments. | Closest implementation match. This is a signal backtest, not an audited live return. Its preferred 8–12 cent edge bucket is a selected subgroup, not a transferable rule. |
| [That bot's backtest source](https://raw.githubusercontent.com/william-mcdermott/kalshi-trading-bot/main/backend/scripts/mlb_backtest.py) | The Kelly PnL loop uses logged ask or one minus logged bid, stake/price payout, and fees. | It does not require delayed executable depth or demonstrated fills in that loop. The maker option explicitly assumes a limit fill. This cannot resolve our execution question. |
| [Gu et al., *When do prophets profit?*, v3](https://arxiv.org/html/2607.06166v3) | Reports a live Kalshi pilot: $200 to $360.67, 236 filled orders, 129 markets, 26 days, $26.31 exchange fees. | Author-reported short pilot, not an independently audited MLB result. Two-hour decisions and a three-hour pre-resolution stop differ greatly from in-game trading. The paper itself declines to establish reliable alpha; removing the three best and three worst trades reduces reported ROI to 24.2%. |
| [Kaunitz et al., *Beating the bookies with their own numbers*](https://arxiv.org/pdf/1710.02824) | Reports $957.50 live profit over 265 soccer bets in five months using bookmaker consensus to find mispriced offers. | Historical sportsbook evidence, not Kalshi/MLB. About 30% of displayed opportunities had already changed in their paper exercise. The reported 8.5% return does not reconcile with 265 flat $50 stakes: those imply 7.23%; retain the discrepancy rather than repeating the percentage as verified. |
| [Bürgi, Deng and Whelan, *Makers and Takers*](https://www.karlwhelan.com/Papers/Kalshi.pdf) | Finds higher returns for makers and a 2.6% average return for maker purchases priced at least 50 cents. | Aggregate historical transactions, not a specified executable bot. Sample ends before maker fees began; it cannot justify assuming free liquidity provision today. |
| [Gupta, maker/taker study](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6858200) | Abstract reports positive aggregate maker returns and only +0.18 percentage points of maker alpha for MLB. | Abstract inspected through indexed SSRN text; direct full-text access failed. Fee treatment and replicability were not audited. Weak support for a small potential maker advantage, not an accepted net-return estimate. |
| [Jain, Harvard NBA market-making thesis](https://dash.harvard.edu/entities/publication/8600d0df-2b18-41ac-b75c-5fc15c2495fa) | Models when queue clearing outweighs adverse selection; quotes depend on game state and inventory. | Academic simulation informed by NBA data, not live MLB profit. Supports measuring fill quality and restricting risky quoting regimes. |
| [Moulinier, *Mistiming is not Arbitrage*](https://doi.org/10.2139/ssrn.7175258) | Author reports no surviving fee/spread edge across nine preregistered hypotheses, including a matched MLB cross-venue study. | Indexed abstract inspected, replication not run. A useful falsification warning: asynchronous prices create apparent arbitrage, and passive fills can systematically select losses. |
| [Moshrefi, sports price calibration](https://arxiv.org/html/2607.14430v1) | Reports that calibration changes with time to market close across MLB, NBA, and NHL. | Descriptive miscalibration is not executable profit. Its realized time-to-close grouping must not become a feature that reveals the future game end. Use already observed inning/state in deployable hypotheses. |
| [Simon, MLB sportsbook line movement](https://pubsonline.informs.org/doi/10.1287/mnsc.2022.00456) | Peer-reviewed study of 3,681 MLB games reports pregame pricing inefficiencies and overreaction. | Pregame sportsbook results do not validate fading in-game scoring jumps. Our existing indiscriminate reversion experiments lost money. |
| [crollila's MLB calibration repository](https://github.com/crollila/pm-calibration) | Reports that apparent calibration significance disappears when uncertainty accounts for repeated observations of the same game. | Its historical tradeability test is blocked by missing executable data. Supports our game/day grouping and prospective depth collection; supplies no verified profitable strategy. |

## Priority 1: measure which passive fills are worth accepting

The next analysis priority is the existing maker experiments' 5/30/60-second
fill diagnostics, followed by a separately frozen conditional quoting policy.
Selective liquidity provision is our hypothesis, not a conclusion that
market making is profitable.

For every filled entry, retain decision, arrival, fill, and cancellation
timestamps; limit and fill price; initial queue; consumed trade volume; fee;
inventory; paired-market prices; inning; observed game state; and feed age.
Measure subsequent midpoint changes at the declared horizons as diagnostics;
any added horizon belongs in a new experiment. Independently calculate
fee-adjusted executable bid marks
using available depth. A future midpoint is not a liquidation fill. Missing
observations and truncated horizons remain missing, not zero-return winners.

Compare spread earned with adverse price movement after the fill. Analyze
pregame, observed inning breaks, and active play separately, with game/day
clustered uncertainty. Quote cancellation after a pitch observation cannot
erase a fill that occurs before cancellation arrives. Keep zero-fill games
and all attempted orders in coverage and fill-rate denominators.

Use today's diagnostic sample for development only. Any new queue/volatility
filter must have a new protocol, frozen source/model hashes and a cutoff
before its evaluation data are collected or inspected. The current frozen
hold strategy remains a baseline. The existing twelve passive variants are
development candidates, so selecting the best today would not create a new
holdout result.

If the diagnostics support quoting, implement one small family of policies:
quote only when estimated fill-conditioned value exceeds maker fees and
an adverse-selection buffer; widen or stop on stale feeds and rapid changes;
skew quotes to reduce inventory; permit additional entries after legitimate
exits within explicit account/game exposure caps. Do not increase turnover
by lowering an edge threshold until losses disappear from the report.

## Priority 2: an independent sportsbook probability reference

Collect timestamped two-sided moneylines from available authorized sources,
with provider update time and local receipt time. Normalize implied
probabilities to remove the sportsbook margin. Retain missing quotes,
cross-book disagreement, and an explicit age limit. Consensus is an estimate,
not an oracle. Do not substitute closing odds for an earlier observation.

First use pregame consensus as an independent anchor for our frozen baseball
model. Only extend to in-game consensus if feed latency can be measured and
the data are sufficiently current. Kalshi fills still require actual books
at delayed arrival. We need not place sportsbook bets to test their prices
as information. Data availability, licensing and cost require a separate
assessment; no paid feed or external account was opened in this review.

## Priority 3: paired-contract consistency

Screen both team markets using synchronized, already received books and
verified game rules. A pair is actionable only if available prices plus all
fees leave positive value at the quantities each leg can actually fill.
Simulate both legs, reserve both costs, and retain unhedged exposure if only
one fills. A midpoint sum below one, an unmatched doubleheader, or a
one-sided execution is not locked-in profit. This is a separate experiment,
with its own fixed parameters and subsequent evaluation window.

## Economics and acceptance

Current [Kalshi fees](https://kalshi.com/docs/kalshi-fee-schedule.pdf) list
KXMLBGAME maker and taker multipliers of one. At 50 cents, the base per-contract
fee is about 0.4375 cents for a maker versus 1.75 cents for a taker, before
applicable rounding. Both sides of a round trip and any forced taker exit
must be charged. [Balance rounding](https://docs.kalshi.com/getting_started/fee_rounding)
differs between direct and non-direct members; use the account's actual rule.
Fee savings alone do not pay for adverse selection.

Report net settled shadow PnL, conservative open-inventory marks, peak cash
committed, drawdown, attempts, fills, entries per game, and the fraction of
all scheduled/mapped games with an entry. Report every declared candidate
and latency stress, not just the winner. More quotes or more contracts do
not mean more independent trading opportunities.

The target remains at least one entry in every eligible game. It is an
activity constraint to measure alongside profitability, not an instruction
to buy at a negative expected return. Repeatable fresh-data gains, realistic
fills, and stable execution stresses are required before accepting a strategy.
Shadow performance still cannot establish actual account profit.
