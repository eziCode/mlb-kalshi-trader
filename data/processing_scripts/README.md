# Processing scripts

These scripts are normally invoked by the root-level `setup_data.py` command.

## `build_event_state_features.py`

Combines downloaded Statcast pitches with authoritative MLB timestamps and
builds:

```text
data/processed/mlb_game_state/pitch_state_features.parquet
```

The output contains game identity, completed-event availability, inning,
half-inning, outs, score differential, count, runners, and pitch identity.
Unused hitter-form and pitcher-context features are deliberately excluded.

## `build_shared_data.py`

Combines pitch states, cached MLB feeds, exact Kalshi executions, and the
packaged local win-expectancy model. It maps home-team markets, handles
doubleheaders chronologically, derives the last pregame execution anchor, and
writes:

```text
data/shared/home_market_trades.parquet
data/shared/state_updates.parquet
```

Run processors directly only when debugging:

```bash
.venv/bin/python data/processing_scripts/build_event_state_features.py
.venv/bin/python data/processing_scripts/build_shared_data.py
```

