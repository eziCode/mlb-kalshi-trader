# Empirical reversion

Architecture: `mlb_kalshi/empirical_reaction.py`. The custom builder creates
exact-timestamp reaction events; training also performs the frozen evaluation.
Study artifacts are in
[`studies/empirical_reversion_strategy`](../../studies/empirical_reversion_strategy).

```bash
python -m approaches.empirical_reversion.build_data
python -m approaches.empirical_reversion.train
docker build -f approaches/empirical_reversion/Dockerfile -t empirical-reversion .
```

Paper mode is replay-only. Validation failed and deployment is disabled.
