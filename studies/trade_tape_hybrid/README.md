# Exact-timestamp Kalshi trade-tape hybrid

This study replaces one-minute Kalshi candles with 5.47 million observed
home-market trades and replaces next-pitch event timing with MLB's cached
pitch `endTime` values.

## Method

- A hit's pre-event anchor is the final observed trade before the terminal
  pitch starts, provided that trade is no more than five seconds old.
- The expected post-hit target applies the local win-expectancy model's
  log-odds update to that pre-event Kalshi trade price.
- A hit must move local fair value by at least 0.5 percentage points in the
  batting team's direction. Every candidate stores its exact at-bat, pitch,
  and event-end timestamp.
- An unfilled candidate is canceled by the next completed plate appearance or
  any intervening score, out, or base-state change.
- A residual must persist for the selected number of seconds.
- Entry occurs only on a strictly later trade whose reported taker outcome
  side matches the desired position and whose reported size covers the paper
  order. Exit uses the corresponding opposite-side trade evidence.
- Positions exit only after model-adjusted reversion. There is no time-based
  exit; unreverted positions settle with the market.
- The tape does not contain standing bid/ask quotes, so observed compatible
  trades are an execution proxy rather than proof that the same liquidity
  would have remained available to the strategy.

## Results

- Tuning period, June 17–27: 7.5% minimum residual and one-second persistence,
  +7.78% ROI on 114 fills.
- Fixed June 28–July 10 holdout: −7.77% ROI, −$104.50 PnL on 130 fills,
  including $80.89 in fees.
- The 113 positions that reverted earned $74.56 in aggregate. The 17 positions
  that never reverted lost $179.06 at settlement.

Because the fixed holdout failed, deployment is disabled in
`models/market_reaction_model/trade_tape_config.json`.

Machine-readable outputs are in `tuning_grid.csv`, `tuning_summary.json`,
`holdout_summary.json`, and `holdout_trades.csv`.
