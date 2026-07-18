# Absolute probability

This foundational approach trains the local win-expectancy and reaction models
and trades their absolute probability disagreement with Kalshi. Architecture:
`mlb_kalshi/strategy.py`. There is no dedicated `studies/` directory; its
corrected causal result is documented in the root README.

```bash
python -m approaches.absolute_probability.build_data
python -m approaches.absolute_probability.train
python -m approaches.absolute_probability.evaluate
docker build -f approaches/absolute_probability/Dockerfile -t absolute-probability .
```

Paper mode is replay-only. Validation failed and deployment is disabled.
