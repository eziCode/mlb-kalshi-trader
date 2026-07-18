# Empirical runner

Architecture: `mlb_kalshi/runner.py`. This is an extension of empirical
reversion and consumes its generated candidates. Study artifacts are in
[`studies/empirical_runner`](../../studies/empirical_runner).

```bash
python -m approaches.empirical_runner.build_data
python -m approaches.empirical_runner.train
docker build -f approaches/empirical_runner/Dockerfile -t empirical-runner .
```

Paper mode is replay-only. Validation failed and deployment is disabled.
