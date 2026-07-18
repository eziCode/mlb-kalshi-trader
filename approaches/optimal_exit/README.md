# Optimal exit

Architecture: `mlb_kalshi/optimal_exit.py`. It builds post-entry trajectories
and trains a continuation-value regressor. Study artifacts are in
[`studies/optimal_exit_policy`](../../studies/optimal_exit_policy).

```bash
python -m approaches.optimal_exit.build_data
python -m approaches.optimal_exit.train
python -m approaches.optimal_exit.evaluate
docker build -f approaches/optimal_exit/Dockerfile -t optimal-exit .
```

Paper mode is replay-only. Validation failed and deployment is disabled.
