# Processing scripts

These scripts are normally invoked by the root-level `setup_data.py` command.

## `build_event_state_features.py`

Combines downloaded Statcast pitches with authoritative MLB timestamps and
builds:

```text
data/processed/mlb_game_state/pitch_state_features.parquet
```

The output contains game identity, pitch start timestamps, inning,
half-inning, outs, score differential, count, runners, and pitch identity.
Unused hitter-form and pitcher-context features are deliberately excluded.

## `build_shared_data.py`

Combines pitch states, cached MLB feeds, exact Kalshi executions, and the
packaged local win-expectancy model. It maps home-team markets, handles
doubleheaders chronologically, derives the last pregame execution anchor, and
writes:

```text
data/shared/home_market_trades.parquet
data/shared/away_market_trades.parquet
data/shared/state_updates.parquet
```

In the previous local snapshot those outputs contained 12,984,711 home executions,
15,554,123 paired away executions, and 1,149,706 causal state updates. The
independent away tape is required for realistic home-NO execution; the
backtests never borrow home-market liquidity for that side.

Run processors directly only when debugging:

```bash
.venv/bin/python data/processing_scripts/build_event_state_features.py
.venv/bin/python data/processing_scripts/build_shared_data.py
```

The state builder now derives post-pitch count and terminal-play state from
the current feed object. It does not shift future pitch state backwards.
The optional `--pregame-prior-state` and `--prior-state-as-of` arguments
reproduce the frozen MLB prior used by the reversion runtime. The supplied
as-of date must precede every evaluated game. Metadata records model/prior
hashes, the state contract, and game coverage.
