# Research reboot

Run from the repository root with the existing `.venv`. This workflow uses
the packaged model weights without retraining and keeps generated data under
the ignored `data/` paths.

```bash
.venv/bin/python -m research.reboot prepare \
  --start-date 2026-08-12 --end-date 2026-09-26 --workers 4
.venv/bin/python -m research.reboot evaluate \
  --start-date 2026-08-12 --end-date 2026-09-26 --starting-cash 100
```

Independent weekly replays use four worker processes by default; set
`evaluate --workers 1` for a smaller memory footprint. Portfolio accounting
still runs once, chronologically over the whole period.

Use `prepare --skip-downloads` after a successful acquisition. It requires a
complete, non-smoke acquisition manifest covering the requested dates. The
downloaders retain per-market caches, merge incremental Statcast windows, and
publish aggregate files atomically. Failed or malformed market downloads do
not silently produce a successful aggregate. Data is processed in weekly
chunks to bound memory use.

`evaluate` writes a manifest before scoring, verifies model/data compatibility,
runs every scenario in the fixed [protocol](VALIDATION_PROTOCOL.md), and writes
`REPORT.md`, `summary.json`, and cash-admitted trade CSVs. Repeated evaluations
require a new `--output-dir`; later results are not an untouched holdout once
they have been consulted. Dates after previous development are a useful
chronological check, not an automatic guarantee of an unbiased test.

The state builder scores the current pitch and its completed play directly.
It never copies the next pitch's runners, score, or inning. Terminal outcomes
are available no earlier than the later of pitch end and play end, then receive
the declared publication delay. Incomplete runner actions cannot trigger a new
event entry. Final archives can contain later official corrections; only
prospectively recorded observations can resolve that remaining limitation.

Both live policies are disabled. The paper override cannot bypass disabled
reversion live execution, and the combined launcher starts no processes when
both policies are disabled. No external runtime is started by this workflow.

## Prospective evidence recorder

Set `KALSHI_API_KEY_ID` and `KALSHI_PRIVATE_KEY_PATH` in your environment. Pass
the actual game ID and **both** independently traded team-YES tickers:

```bash
.venv/bin/python -m research.record \
  --game-pk GAME_PK --ticker HOME_MARKET_TICKER --ticker AWAY_MARKET_TICKER \
  --duration-seconds 10800
```

The recorder subscribes only to public books and trades and polls MLB. It
contains no order-submission path. Compressed JSONL files preserve raw book
snapshots/deltas, connection boundaries, exchange sequences, local wall-clock
and monotonic timestamps, and changed MLB play observations. Sequence gaps
force a reconnect and a fresh book snapshot. Data before a missing sequence
must not be carried across the gap. It records gaps rather than pretending a
disconnected or incomplete book remained executable.

This builds the evidence needed for a subsequent book-based shadow replay and
fill reconciliation; the recorder itself does not establish profitability.

## Validation

```bash
.venv/bin/python -m unittest discover -v
(cd hit_reversion_strategy && ../.venv/bin/python -m unittest discover -s tests -v)
```

Tests include immutable entry price/quantity, adverse state during submission,
no future rescoring, expiry, side-specific partial fills, zero-trade results,
cash constraints, next-pitch leakage, incomplete doubleheaders, and disabled
live-policy enforcement.

Exchange semantics were checked against the official
[historical API](https://docs.kalshi.com/getting_started/historical_data),
[fee rounding](https://docs.kalshi.com/getting_started/fee_rounding),
[fee schedule](https://kalshi.com/docs/kalshi-fee-schedule.pdf), and
[order-book stream](https://docs.kalshi.com/websockets/orderbook-updates).
