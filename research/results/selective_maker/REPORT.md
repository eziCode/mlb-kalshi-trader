# Selective maker implementation and integration check

The implementation passes 250 root-suite tests, including 36 new tests.
The earlier frozen value and maker-hold paths remain unchanged.

Three declared policies run at 680ms, 1.5s, and 3s submission/cancellation delays.
The first completed periodic replay covers 14 mapped games from a 15-game slate.
One game has an observed first pitch; none has an observed final result.
The reference selective-live scenario made four opening attempts and zero fills.
All nine replays retain valid book continuity. This is integration evidence,
not a profitability result. The capture is incomplete and scoring continues.

| Scenario | Opening attempts | Filled entries | Net shadow PnL |
|---|---:|---:|---:|
| selective_live_latency_0.68 | 4 | 0 | $+0.0000 |
| selective_live_latency_1.5 | 4 | 0 | $+0.0000 |
| selective_live_latency_3.0 | 3 | 0 | $+0.0000 |
| selective_break_latency_0.68 | 0 | 0 | $+0.0000 |
| selective_break_latency_1.5 | 0 | 0 | $+0.0000 |
| selective_break_latency_3.0 | 0 | 0 | $+0.0000 |
| sportsbook_anchor_latency_0.68 | 0 | 0 | $+0.0000 |
| sportsbook_anchor_latency_1.5 | 0 | 0 | $+0.0000 |
| sportsbook_anchor_latency_3.0 | 0 | 0 | $+0.0000 |

The independent sportsbook adapter and as-of consensus path are implemented,
but no ODDS_API_KEY or sportsbook journal is configured in this run. That
candidate remains inactive; no alternate feed or synthetic probability is substituted.

The local periodic run is under `data/reboot/selective_watch_20260927_v1`.
Each prefix retains reports, fills, order attempts, 5/30/60-second diagnostics,
and settlement-adjusted accounting. Rejection counters describe signal checks,
including checks outside eligible opening windows; they are not additive causes
of missed trades. Future marks never feed the strategy.

See [the protocol](../../NEXT_EXPERIMENT.md), [commands](../../README.md),
and [the immutable integration metadata](integration_check.json).
The full frozen source is archived locally with the protocol; its hashes are
also retained in [protocol.json](protocol.json).
