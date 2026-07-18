"""Compare causal execution and timestamp assumptions on tuning dates only."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlb_kalshi.state_overshoot import (  # noqa: E402
    OvershootConfig, build_state_overshoot_candidates, simulate_state_reversion,
)


DATA_DIR = PROJECT_ROOT / "data/processed/trade_tape"
STUDY_DIR = PROJECT_ROOT / "studies/state_reversion"
TUNE_START = pd.Timestamp("2026-06-22").date()
HOLDOUT_START = pd.Timestamp("2026-06-28").date()


def main() -> None:
    trades = pd.read_parquet(DATA_DIR / "home_market_trades.parquet")
    updates = pd.read_parquet(DATA_DIR / "state_updates.parquet")
    for frame in (trades, updates):
        frame["game_date"] = pd.to_datetime(frame["game_date"]).dt.date
    games = set(updates.loc[
        (updates.game_date >= TUNE_START) & (updates.game_date < HOLDOUT_START),
        "game_pk",
    ])
    trades = trades[trades.game_pk.isin(games)].copy()
    updates = updates[updates.game_pk.isin(games)].copy()

    rows = []
    variants = [("taker", "taker"), ("maker", "taker"), ("maker", "maker")]
    for latency in [1.0, 2.0, 3.0]:
        for entry_execution, exit_execution in variants:
            config = OvershootConfig(
                minimum_logit_residual=0.08,
                observation_latency_buffer_seconds=latency,
                minimum_fair_logit_move=0.02,
                minimum_target_profit=0.25,
                entry_execution=entry_execution,
                exit_execution=exit_execution,
            )
            candidates = build_state_overshoot_candidates(trades, updates, config)
            if candidates.empty:
                result = simulate_state_reversion(
                    candidates, [], config, expected_pnls=[]
                )
            else:
                result = simulate_state_reversion(
                    candidates, np.ones(len(candidates)), config,
                    expected_pnls=np.ones(len(candidates)),
                )
            rows.append({
                "latency_buffer_seconds": latency,
                "entry_execution": entry_execution,
                "exit_execution": exit_execution,
                "maker_fee_rate": config.maker_fee_rate,
                "independent_candidates": len(candidates),
                "portfolio_trades": result.accepted,
                "reversion_exits": result.reversion_exits,
                "thesis_invalidations": result.thesis_invalidations,
                "timeout_exits": result.timeout_exits,
                "settlements": result.settlements,
                "fees": result.fees, "capital": result.capital,
                "pnl": result.pnl, "roi": result.roi,
            })
    report = pd.DataFrame(rows).sort_values(
        ["roi", "pnl"], ascending=False
    )
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(STUDY_DIR / "execution_latency_sensitivity.csv", index=False)
    summary = {
        "dates": "2026-06-22 through 2026-06-27",
        "outer_holdout_used": False,
        "maker_fill_assumption": (
            "resting order fills at its limit only after a strictly later "
            "opposite-taker trade reaches the price; queue position unknown"
        ),
        "maker_fee_rate": 0.0,
        "results": report.to_dict(orient="records"),
    }
    (STUDY_DIR / "execution_latency_sensitivity.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(report.to_string(index=False, formatters={"roi": "{:.2%}".format}))


if __name__ == "__main__":
    main()
