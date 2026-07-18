# Trade-tape hybrid

Architecture: `mlb_kalshi/trade_tape.py`. Its exact-timestamp data builder and
tuning/evaluation entrypoints live here. Study artifacts are in
[`studies/trade_tape_hybrid`](../../studies/trade_tape_hybrid).

This is the only approach with a true live paper adapter:

```bash
PAPER_MODE=live ALLOW_UNVALIDATED_HYBRID=1 \
python -m approaches.trade_tape_hybrid.paper_trader --all-games
docker build -f approaches/trade_tape_hybrid/Dockerfile -t trade-tape .
```

Validation failed, so live paper mode still requires the explicit override.
