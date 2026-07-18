# Hybrid event

Architecture: `mlb_kalshi/hybrid.py`. Training tunes the event-relative fair
target; custom data comes from the causal candle join. Study artifacts are in
[`studies/hybrid_event_strategy`](../../studies/hybrid_event_strategy).

```bash
python -m approaches.hybrid_event.build_data
python -m approaches.hybrid_event.train
python -m approaches.hybrid_event.evaluate
docker build -f approaches/hybrid_event/Dockerfile -t hybrid-event .
```

Paper mode is replay-only. Validation failed and deployment is disabled.
