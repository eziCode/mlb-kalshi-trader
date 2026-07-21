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
(cd hit_reversion_strategy && ../.venv/bin/python scripts/tune.py --workers 8)
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

The optional strategy-only images also use the repository root as their build
context so their frozen backtest data is included:

```bash
docker build -f settlement_value_strategy/Dockerfile -t settlement-value .
docker build -f hit_reversion_strategy/Dockerfile -t hit-reversion .
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

Production-style paper trading runs both strategies in one container behind a
single authenticated Kalshi WebSocket and one adaptive, cached MLB feed.
Put the Kalshi RSA private key in
`secrets/kalshi-private.key` (the directory is gitignored), then start with:

```bash
docker volume create mlb-paper-state
docker run -d \
  --name mlb-paper \
  --restart unless-stopped \
  -e KALSHI_API_KEY_ID=YOUR_KEY_ID \
  -e KALSHI_PRIVATE_KEY_PATH=/run/secrets/kalshi-private.key \
  -e SLATE_TIMEZONE=America/Chicago \
  -e PAPER_STARTING_CASH=1000 \
  -v "$PWD/secrets/kalshi-private.key:/run/secrets/kalshi-private.key:ro" \
  -v "$PWD/paper_logs:/app/paper_logs" \
  -v mlb-paper-state:/app/state \
  mlb-kalshi-trader paper-both --date YYYY-MM-DD
```

The Kalshi feed performs bounded REST bootstrap/recovery calls, then supplies
top-of-book snapshots and exact trades from one WebSocket to every isolated
worker. The MLB feed polls each game once, shares that snapshot across both
strategies, slows down before games, and applies jittered exponential backoff
while retaining the last valid state after upstream errors.
See the strategy READMEs for single-strategy diagnostic modes:

- [`settlement_value_strategy/README.md`](settlement_value_strategy/README.md)
- [`hit_reversion_strategy/README.md`](hit_reversion_strategy/README.md)

The combined runtime uses this container layout:

```text
/app/settlement_value_strategy
/app/hit_reversion_strategy
/app/paper_logs
/app/state/settlement-value
/app/state/hit-reversion
```

Query a running container without referring to its internal Python script:

```bash
docker exec \
  -e PAPER_PORTFOLIO_DB=/app/state/settlement-value/settlement_value_portfolio_YYYY-MM-DD.sqlite3 \
  mlb-paper /bin/sh /app/docker-entrypoint.sh mispricing portfolio-status

docker exec \
  -e PAPER_PORTFOLIO_DB=/app/state/hit-reversion/hit_reversion_portfolio_YYYY-MM-DD.sqlite3 \
  mlb-paper /bin/sh /app/docker-entrypoint.sh trade-tape portfolio-status
```
