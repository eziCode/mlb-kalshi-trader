# MLB Kalshi trader

This repository contains two MLB Kalshi paper-trading strategies:

- **Settlement value** (`mispricing` selector): predicts the home team's final
  settlement probability after any completed pitch and holds one qualifying
  position per game to settlement.
- **Hit reversion** (`trade-tape` selector): trades delayed market reactions
  after hits and normally exits at its state-adjusted target.

Neither paper trader submits real orders.

## Quick start

Run every command below from the repository root.

```bash
python3.12 -m venv .venv
.venv/bin/pip install \
  -r settlement_value_strategy/requirements.txt \
  -r hit_reversion_strategy/requirements.txt
```

Choose `mispricing`, `trade-tape`, or `both`. The command downloads Statcast,
MLB live feeds, and exact Kalshi executions, then builds the requested data:

```bash
.venv/bin/python setup_data.py both
```

Use `mispricing` or `trade-tape` instead of `both` when setting up one strategy.

The command prints each stage and streams downloader progress. It writes:

```text
data/shared/home_market_trades.parquet
data/shared/state_updates.parquet
```

Then prepare, train, and evaluate the settlement-value strategy:

```bash
.venv/bin/python -m settlement_value_strategy.prepare_data
.venv/bin/python -m settlement_value_strategy.train
.venv/bin/python -m settlement_value_strategy.backtest
```

Tune and evaluate the hit-reversion strategy from the same shared data:

```bash
(cd hit_reversion_strategy && ../.venv/bin/python scripts/tune.py)
(cd hit_reversion_strategy && ../.venv/bin/python scripts/backtest.py)
```

Retraining after data setup matters: frozen models are tied to the data
contract used to produce them.

## Data flow

```text
setup_data.py
  ├─ Statcast pitches ───────────────┐
  ├─ MLB live feeds/timestamps ──────┼─> data/processed pitch states
  └─ exact Kalshi executions ────────┘
                                           │
                                           v
                                    data/shared/
                                    ├─ home_market_trades.parquet
                                    └─ state_updates.parquet
                                           │
                     ┌─────────────────────┴────────────────────┐
                     v                                          v
       data/settlement_value + train                    trade-tape tune/backtest
```

Raw downloads are resumable under `data/raw/`. Intermediate pitch-state data,
shared inputs, and strategy-specific datasets all remain under root `data/`.
Model and study outputs stay inside their respective strategy folders.

## Data setup options

```bash
# Show every stage without running it
.venv/bin/python setup_data.py both --dry-run

# Reprocess existing raw downloads without network calls
.venv/bin/python setup_data.py both --skip-downloads

# Small acquisition/integration run
.venv/bin/python setup_data.py both --max-games 5

# Force refresh of reusable API caches
.venv/bin/python setup_data.py both --refresh
```

By default Kalshi data begins at the MLB market launch date, 2025-04-16, and
ends today. Override this with `--start-date` and `--end-date`.

## Tests

```bash
.venv/bin/python -m unittest \
  settlement_value_strategy.test_strategy settlement_value_strategy.test_pipeline -v

(cd hit_reversion_strategy && \
  ../.venv/bin/python -m unittest discover -s tests -v)
```

## Docker

Build one image from the repository root:

```bash
docker build -t mlb-kalshi-trader .
```

Select the strategy in the container command:

```bash
docker run --rm mlb-kalshi-trader mispricing backtest
docker run --rm mlb-kalshi-trader trade-tape backtest
```

The operation defaults to `backtest`. Run `docker run --rm
mlb-kalshi-trader help` to see every operation.

Data setup is also available in Docker. Mount `data/` so downloads persist:

```bash
docker run --rm \
  -v "$PWD/data:/app/data" \
  mlb-kalshi-trader setup-data both
```

## Paper trading

Both live paper traders require public network access. See the strategy
READMEs for single-game, all-game, discovery, logging, and safety options:

- [`settlement_value_strategy/README.md`](settlement_value_strategy/README.md)
- [`hit_reversion_strategy/README.md`](hit_reversion_strategy/README.md)
