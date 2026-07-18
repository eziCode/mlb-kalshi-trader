# Residual path

Architecture: `mlb_kalshi/residual_path.py`. Training separates contraction
alpha, maker-fill probability, and horizon PnL models. Its custom all-signal
dataset is built inside the trainer from the shared exact trade tape. Study
artifacts are in
[`studies/residual_path_policy`](../../studies/residual_path_policy).

```bash
python -m approaches.residual_path.build_data
python -m approaches.residual_path.train
python -m approaches.residual_path.evaluate
docker build -f approaches/residual_path/Dockerfile -t residual-path .
```

Paper mode is replay-only. Validation failed and deployment is disabled.
