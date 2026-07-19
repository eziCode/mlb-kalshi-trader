# Data pipeline

The `data/` directory separates acquisition, intermediate processing, and the
canonical inputs consumed by both strategies.

```text
data/
├─ download_scripts/       public API acquisition
├─ processing_scripts/     deterministic normalization and feature building
├─ raw/                    resumable downloads and API caches
├─ processed/              intermediate pitch-state features
├─ shared/                 canonical normalized trade/state inputs
├─ mispricing/             decisions and compact execution tape
└─ trade_tape/             local win-model training inputs
```

## Recommended command

From the repository root:

```bash
.venv/bin/python setup_data.py both
```

The required first argument is `mispricing`, `trade-tape`, or `both`.
Acquisition and shared normalization are common to both. Settlement-value
setup also
builds `data/settlement_value/decision_rows.parquet` and
`data/settlement_value/execution_trades.parquet`.

It runs, in order:

1. `download_mlb_statcast.py`
2. `download_mlb_pitch_timestamps.py`
3. `build_event_state_features.py`
4. `download_live_kalshi_market_logs.py`
5. `build_shared_data.py`

Each step has a visible banner, the invoked command, live downloader output,
and elapsed time. A failed step stops the pipeline immediately.

## Inputs downloaded

- Statcast pitch-by-pitch baseball state
- Full MLB live feeds with authoritative pitch start/end timestamps
- Final MLB scores used as settlement labels
- Exact Kalshi `KXMLBGAME` executions, including timestamp, price, size, and
  taker side

One-minute Kalshi candles are intentionally not used.

## Shared outputs

`shared/home_market_trades.parquet` contains the normalized home-team contract
execution tape. `shared/state_updates.parquet` contains causal completed-pitch
state transitions, outcomes, local fair values, and event timing.

The hit-reversion strategy consumes these files directly. The settlement-value
strategy turns them into model-specific decision rows and a compact causal
execution tape with:

```bash
.venv/bin/python -m settlement_value_strategy.prepare_data
```

## Reruns and smoke tests

Downloads and per-market/per-game caches are reusable. Common commands:

```bash
.venv/bin/python setup_data.py both --skip-downloads
.venv/bin/python setup_data.py both --max-games 5
.venv/bin/python setup_data.py both --refresh
.venv/bin/python setup_data.py both --dry-run
```

Do not treat a `--max-games` output as a training corpus; it is intended only
to test API access and schemas.
