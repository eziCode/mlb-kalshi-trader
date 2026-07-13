"""Build post-entry decision trajectories for fitted optimal stopping."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlb_kalshi.optimal_exit import build_exit_trajectories  # noqa: E402
from mlb_kalshi.trade_tape import (  # noqa: E402
    TradeTapeConfig,
    simulate_trade_tape,
)


DATA_DIR = PROJECT_ROOT / "data/processed/trade_tape"
MODEL_DIR = PROJECT_ROOT / "models/market_reaction_model"
OUTPUT_PATH = DATA_DIR / "optimal_exit_trajectories.parquet"


def main() -> None:
    trades = pd.read_parquet(DATA_DIR / "home_market_trades.parquet")
    updates = pd.read_parquet(DATA_DIR / "state_updates.parquet")
    selected = json.loads((MODEL_DIR / "trade_tape_config.json").read_text())
    entry_config = TradeTapeConfig(**selected)
    baseline = simulate_trade_tape(trades, updates, entry_config)
    print(
        f"Baseline entries: {len(baseline.records):,} "
        f"({baseline.reversion_exits:,} reversions, "
        f"{baseline.settlements:,} settlements)"
    )
    trajectories = build_exit_trajectories(
        trades, updates, baseline.records
    )
    if trajectories.empty:
        raise RuntimeError("No optimal-exit trajectories were generated")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    trajectories.to_parquet(OUTPUT_PATH, index=False)
    print(
        f"Saved {len(trajectories):,} decision snapshots across "
        f"{trajectories['trajectory_id'].nunique():,} trajectories to "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
