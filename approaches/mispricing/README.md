# Mispricing

Architecture: `mlb_kalshi/mispricing.py`. Training builds its event-agnostic
decision rows, fits/calibrates settlement probability, and tunes fee-aware
YES/NO thresholds. Study artifacts are in
[`studies/mispricing`](../../studies/mispricing).

```bash
python -m approaches.mispricing.build_data
python -m approaches.mispricing.train
python -m approaches.mispricing.evaluate
PAPER_MODE=replay python -m approaches.mispricing.paper_trader
docker build -f approaches/mispricing/Dockerfile -t mispricing .
```

Paper mode is replay-only. Development holdout passed, but live mode remains
disabled pending a fresh forward-paper adapter and validation period.
