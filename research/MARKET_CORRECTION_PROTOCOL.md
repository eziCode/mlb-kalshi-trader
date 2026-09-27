# Market correction experiment, declared September 27, 2026

The direct refreshed-model experiment lost money in both development and
selection at every tested threshold. Its high-activity version entered
594/610 games and lost $18.49. All those results remain retained.

Test two specified corrections to that model, without changing execution:

1. Anchor its initial probability to the last home-market print strictly before
   the first pitch. Require that print to be at most five minutes old and
   available at least two seconds before a decision. Add the log-odds change
   from the model's initial state to the pregame market log-odds. This tests
   whether the pregame price supplies information absent from frozen Elo.
2. Fit a small, regularized logistic correction around current market midpoint
   using development games only. Explicit features are market log-odds,
   anchored-model disagreement in log-odds, inning times market log-odds,
   signed home-team indicator, and the other team's contemporaneous price
   discrepancy. Features are antisymmetric between team markets; no intercept,
   identifiers, future outcomes, or fill outcomes are allowed as predictors.
   Final outcomes are training labels. Each game receives equal aggregate
   weight. Standardization is fitted on development data only. Fixed L2
   penalty 20 and Newton optimization; no hyperparameter search.

Use the same August 12–September 1 development, September 2–15 selection,
September 16–26 chronological check. These periods were already inspected
and are research data, not untouched validation. Entry thresholds for both
corrections are 0, 1, and 3 cents after entry costs. Do not introduce more
thresholds after seeing results. Select a candidate using development and
selection only; prefer positive PnL in both, then the selection-period lower
day-block confidence bound. Persist selection before scoring the later check.
Report all candidates, including failures, and average entries and game coverage.

Trades remain one contract, at most one entry per game, with bounded delayed
side-specific trade evidence, entry fees, adverse-price penalty, and shared
cash. Settlement uses exchange timestamps plus sixty seconds. Better fitting
the observed history is not sufficient for deployment; any selected candidate
needs fresh prospective validation and execution stresses.
