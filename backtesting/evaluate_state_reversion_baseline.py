"""Evaluate the frozen deterministic overreaction policy on outer holdout."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlb_kalshi.state_overshoot import OvershootConfig, simulate_state_reversion  # noqa: E402


MODEL_DIR = PROJECT_ROOT / "models/market_reaction_model"
STUDY_DIR = PROJECT_ROOT / "studies/state_reversion"
HOLDOUT_START = pd.Timestamp("2026-06-28").date()


def main() -> None:
    raw = json.loads(
        (MODEL_DIR / "state_reversion_baseline_config.json").read_text()
    )
    config = OvershootConfig(**{
        key: value for key, value in raw.items()
        if key in OvershootConfig.__dataclass_fields__
    })
    frame = pd.read_parquet(STUDY_DIR / "deterministic_candidates.parquet")
    frame["game_date"] = pd.to_datetime(frame.game_date).dt.date
    frame = frame[
        (frame.game_date >= HOLDOUT_START)
        & (frame.entry_latency_seconds <= raw["maximum_selected_entry_latency"])
        & (frame.inning_after <= raw["maximum_inning"])
    ].copy()
    if raw["side_filter"] != "both":
        frame = frame[frame.side.eq(raw["side_filter"])]
    result = simulate_state_reversion(
        frame, np.ones(len(frame)), config, expected_pnls=np.ones(len(frame))
    )
    passed = bool(result.accepted >= 20 and result.pnl > 0 and result.roi > 0)
    raw["validation_passed"] = passed
    raw["enabled"] = False
    (MODEL_DIR / "state_reversion_baseline_config.json").write_text(
        json.dumps(raw, indent=2)
    )
    summary = {
        "holdout_start": str(HOLDOUT_START), "config": raw,
        "candidates": len(frame), "trades": result.accepted,
        "reversion_exits": result.reversion_exits,
        "thesis_invalidations": result.thesis_invalidations,
        "timeout_exits": result.timeout_exits,
        "settlements": result.settlements,
        "fees": result.fees, "capital": result.capital,
        "pnl": result.pnl, "roi": result.roi,
        "validation_passed": passed,
    }
    (STUDY_DIR / "deterministic_holdout_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    pd.DataFrame(result.records).to_csv(
        STUDY_DIR / "deterministic_holdout_trades.csv", index=False
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
