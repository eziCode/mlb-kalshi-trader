# State reversion

Architectures: `mlb_kalshi/state_overshoot.py` and
`mlb_kalshi/state_reversion.py`. The folder covers the deterministic baseline,
two-stage classifier, execution audit, and latency study. Study artifacts are
in [`studies/state_reversion`](../../studies/state_reversion).

Additional commands:

```bash
python models/tune_state_reversion_baseline.py
python models/analyze_state_reversion_execution.py
python backtesting/audit_state_reversion_mechanics.py
docker build -f approaches/state_reversion/Dockerfile -t state-reversion .
```

Paper mode is replay-only. Validation failed and deployment is disabled.
