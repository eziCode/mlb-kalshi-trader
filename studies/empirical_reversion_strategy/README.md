# Empirical hit reaction and reversion study

This study uses only the available 2026 Kalshi trade tape. It does not use the
empty/low-volume 2025 archive.

## Design

1. Build one row per completed MLB hit with exact pitch start/end timestamps.
2. Measure Kalshi's volume-weighted HOME price during the five seconds after
   the hit.
3. Fit a market-reaction model on May 15–31. Its target is the batting team's
   five-second log-odds price move, conditional on the pre-hit price, hit type,
   inning, score, outs, runners, and pre-hit trade activity.
4. Beginning June 1, compare the observed reaction with that empirical
   expectation. A contrarian entry is fillable only on a strictly later trade
   with the compatible aggressor side and enough reported size for a $10
   position.
5. Label an entry as a profitable reversion only if a later compatible trade
   crosses the expected price before settlement and the round trip is positive
   after entry and exit fees.
6. Fit the reversion classifier on June 1–16, tune the probability safety
   margin on June 17–27, and evaluate once on June 28–July 10.

There is no predetermined-time exit. A position exits at an executable
reversion or otherwise settles.

## Result

- Reaction events: 8,125 across 752 games.
- Fillable post-June candidates: 3,392.
- Reversion classifier AUC: 0.748 on tuning and 0.722 on holdout.
- A future executable crossing is common (77.2% on tuning; 74.1% on holdout),
  but the typical reversion gain is much smaller than a failed position's
  settlement loss.
- After requiring the predicted reversion probability to exceed each trade's
  fee-adjusted break-even probability, only 3 tuning trades passed. No
  configuration met the 20-trade tuning minimum, and zero trades passed on the
  outer holdout.

The strategy is therefore saved with `enabled: false`. The result does not
support deploying this strategy profitably. It says the market often revisits
the empirical target, but the classifier is not sufficiently confident or
well-calibrated to cover the asymmetric settlement risk.

Machine-readable details are in `reaction_model_summary.json`,
`training_summary.json`, and `tuning_grid.csv`.
