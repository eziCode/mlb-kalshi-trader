# Strategy approaches

Each folder is a runnable strategy boundary containing training, custom data
preparation, evaluation, paper execution, Docker instructions, and a study
manifest. Shared probability and execution contracts remain in `mlb_kalshi/`.

| Approach | Study | Paper mode | Validation |
|---|---|---|---|
| `absolute_probability` | none; root README only | replay | failed |
| `hybrid_event` | `studies/hybrid_event_strategy` | replay | failed |
| `trade_tape_hybrid` | `studies/trade_tape_hybrid` | live/replay | failed |
| `optimal_exit` | `studies/optimal_exit_policy` | replay | failed |
| `empirical_reversion` | `studies/empirical_reversion_strategy` | replay | failed |
| `empirical_runner` | `studies/empirical_runner` | replay | failed |
| `state_reversion` | `studies/state_reversion` | replay | failed |
| `residual_path` | `studies/residual_path_policy` | replay | failed |
| `mispricing` | `studies/mispricing` | replay | development holdout passed |

Build any container from the repository root:

```bash
docker build -f approaches/Dockerfile \
  --build-arg APPROACH=mispricing \
  -t mlb-kalshi-mispricing .
docker run --rm -e PAPER_MODE=replay mlb-kalshi-mispricing
```

Only `trade_tape_hybrid` currently has a real-time adapter. Other containers
default to deterministic paper replay and fail explicitly if `PAPER_MODE=live`
is requested.

Compose profiles provide the same interface:

```bash
docker compose -f approaches/docker-compose.yml \
  --profile mispricing up --build
```
