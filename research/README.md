# Research reboot

## Current candidate and prospective test

The [frozen market correction](results/market_correction/REPORT.md) produces
positive simulated net PnL in development, selection, and a later chronological
check: $6.13 combined across 309 entries in 610 mapped games. It misses the
every-game target, loses under the 1.5-second latency stress, and has a
combined uncertainty interval spanning losses. It remains research evidence.

```bash
.venv/bin/python -m research.market_correction --model-dir MODEL_DIRECTORY \
  --output-dir NEW_CORRECTION_RESULTS
.venv/bin/python -m research.market_check --model-dir MODEL_DIRECTORY \
  --frozen-run CORRECTION_RESULTS --output-dir NEW_STRESS_RESULTS
.venv/bin/python -m research.forward_value FROZEN_CAPTURE --slate SLATE_JSON \
  --model-dir MODEL_DIRECTORY --frozen-run CORRECTION_RESULTS --output-dir NEW_FORWARD_RESULTS
```

`research.watch_forward` periodically snapshots a growing capture and replays
the frozen candidate at 680ms, 1.5s, and 3s delays. `--passive` also runs the
separate twelve-candidate passive study. It never retunes parameters or places
orders. Each prefix is retained; source changes or continuity errors stop the
watcher for review. See [the prospective protocol](FORWARD_VALUE_PROTOCOL.md).

Add `--maker` to run the separately declared [passive hold experiment](FORWARD_MAKER_PROTOCOL.md)
at the same three delays. The probability model stays frozen; fills must clear
the observed queue or trade through the quote, and cancellations take time.
Use `research.record_settlements` to collect public exchange-finalization
observations and pass the journal to the watcher with `--settlements`. Payout
reconciliation preserves partial quantities, actual market identity, missing
settlements, and pending-order uncertainty. It reports settled shadow PnL
separately from open-inventory marks; no actual order is placed.

September 27's [mapping audit](results/passive_slate/mapping_correction.json)
corrects the initial discovery: 14 games are mapped out of 15 scheduled. Boston's
September 27 game had been paired with a September 26 contract without schedule
evidence of postponement. The unverified pair is excluded. The daily matcher now
requires the original game date, uses explicit postponement fields when present,
and rejects ambiguous doubleheaders. All earlier prefixes are retained with
their original scope; subsequent scoring uses the audited slate. MLB's warmup
status also reports abstract `Live`, so passive entries now require a pitch.

## Continuous strategy search and passive experiment

The completed [48-candidate search](results/continuous/REPORT.md) tested
continuous price reversion, momentum, paired-market differences, state value,
all-play responses, and learned future bid changes. None passed both positive
development/selection PnL and the activity requirements. The frozen fallback
made 367 simulated entries in 141 mapped final-period games (2.60/game; 87.2% game
coverage) and lost $12.72 after fees and execution penalties. Every final
execution stress was negative. One contract per entry and $100 starting cash
were used; this is not evidence that every possible strategy fails.
Four ambiguous doubleheader games were excluded by the underlying dataset
builder, as documented in the original reboot audit; they are outside these
frequency denominators.

```bash
# Each new run needs an empty output directory; all trials are retained.
.venv/bin/python -m research.continuous develop --output-dir research/results/NEW_RUN
.venv/bin/python -m research.continuous check --output-dir research/results/NEW_RUN
```

The [protocol](CONTINUOUS_PROTOCOL.md) fixes the split, candidates, frequency
requirements, and fallback rule. Both game coverage and entries/game are
reported; contracts and order submissions are not counted as separate entries.
Features exclude future observations, winners, and simulated execution outcomes.
Cash reservations, failed attempts, exit retries, and all games' positions run
through one chronological portfolio. The later dates were previously inspected
for the old strategy and are not described as an untouched holdout.

The next implemented experiment is [passive inventory quoting](PASSIVE_PROTOCOL.md).
Its shadow replay tracks displayed queue, partial fills, post-only rejection,
cancellation races, and inventory-inclusive marks. It makes no historical maker
fill assumption. Use `python -m research.forward` to discover and record an
entire slate without sending orders, then automatically replay each home market.
An integration check received 30 snapshots, 123 deltas, and 19 trades across
15 games with no gaps; its 20-second pregame window produced zero shadow fills
and does not evaluate profitability.

The [prospective slate experiment](PASSIVE_SLATE_PROTOCOL.md) adds twelve
declared passive variants, observed inning-break windows, inventory timeouts,
bounded liquidation, and subsequent fill-price diagnostics. Its shared cash
replay processes every game's timers chronologically. Freeze an immutable
prefix before inspecting a recording that is still growing:

```bash
.venv/bin/python -m research.slate_study snapshot CAPTURE --output-dir NEW_SNAPSHOT
.venv/bin/python -m research.slate_study evaluate NEW_SNAPSHOT/capture.jsonl.gz \
  --slate SLATE_JSON --output-dir NEW_RESULTS --shared-cash 100
```

An ended recording and an observed final game are reported separately. No
settlement is invented for open inventory at the end of a recording. Initial
September 27 prefixes contain rain delays and pregame markets, with no entries.

The separate [state-model refresh](STATE_REFRESH_PROTOCOL.md) trained on 1,490
completed 2026 regular-season games using final MLB labels and team ratings
that exclude same-day and future results. On 281 later games, its log loss
was 0.475 versus 0.524 for the packaged model given the same inputs. Direct
hold-to-settlement trading remained negative across development and selection
for every tested edge threshold. The most active candidate entered 594 of
610 games and lost $18.49 overall; this is a failed trading experiment despite
the probability improvement. See [calibration](results/state_refresh/calibration.json)
and [trading results](results/state_refresh/settlement_summary.json).

```bash
.venv/bin/python -m research.state_refresh --output-dir NEW_MODEL_DIRECTORY
.venv/bin/python -m research.settlement_study --model-dir NEW_MODEL_DIRECTORY \
  --output-dir NEW_TRADING_RESULTS
```

An additional [old quote-log audit](results/paired_quote_audit.json) found nine
apparent paired-price dislocations in eight of 256 markets. These are sparse
old-policy observations, not synchronized executable opportunities or a count
of all opportunities during those games.

## Frozen original-policy diagnostic

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
