# State probability refresh, declared September 27, 2026

The earlier continuous search used the packaged local state model. That model
and all its replay results are retained. This is a separate experiment.

Train a new baseball win-probability model on regular-season 2026 Statcast
states before July 21. Labels use final post-play scores, including walk-offs;
games tied in the available archive are excluded. Team strength priors use
only earlier calendar dates, never game-ID order within a day. Saved ratings
are frozen at the training cutoff. Training feature columns are explicit;
pitch outcomes, future dates, and final scores are labels or metadata only.

Fit one fixed CatBoost configuration (500 trees, depth 5, learning rate .03,
L2 20, equal aggregate weight per game normalized to mean row weight 1).
Evaluate probability calibration on July 21–August 11 before assessing
strategy returns. Compare with the retained old model on the same inputs.

The first trading experiment uses the new model on the existing causal
August 12–September 26 state observations. Holding to settlement avoids a
second trading fee but keeps capital committed. Quotes remain imperfect
trade-print proxies; entry fills require the existing delayed, bounded,
side-specific execution evidence. No historical passive fills are inferred.

Candidate entry edges are 0, 3, 5, and 8 cents after entry fees and a one-cent
execution penalty. One-contract entries, at most one position per game,
and a shared $100 bankroll are used. Development is August 12–September 1,
selection September 2–15, and a subsequent chronological check September
16–26. All of these dates have already been inspected during earlier
research; they are not untouched validation. Report all candidates, total
games, zero-entry games, game coverage, fees, locked cash, and PnL.

Settlement proceeds use the archived market's actual `settlement_ts` and
`settlement_value_dollars`, with an additional sixty-second cash-availability
delay. Close time or a final MLB score does not release cash. Missing settlement
metadata leaves capital locked and its unresolved value conservatively at zero.
These terminal fields are execution/accounting outcomes, never entry features.
The archived values follow the official [market response schema](https://docs.kalshi.com/api-reference/market/get-market).

Any favorable historical result remains a research hypothesis requiring
prospective games and sensitivity to latency, publication delay, execution
cost, and sample concentration. No real orders are authorized by this
experiment. No frozen production model or policy is replaced or enabled.
