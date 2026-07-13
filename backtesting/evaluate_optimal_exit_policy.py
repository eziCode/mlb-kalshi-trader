"""Final chronological holdout evaluation of the learned sell policy."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

from catboost import CatBoostRegressor
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlb_kalshi.optimal_exit import (  # noqa: E402
    ExitPolicyConfig,
    evaluate_exit_policy,
)


DATA_PATH = (
    PROJECT_ROOT / "data/processed/trade_tape/optimal_exit_trajectories.parquet"
)
MODEL_DIR = PROJECT_ROOT / "models/market_reaction_model"
MODEL_PATH = MODEL_DIR / "optimal_exit_continuation.cbm"
CONFIG_PATH = MODEL_DIR / "optimal_exit_config.json"
STUDY_DIR = PROJECT_ROOT / "studies/optimal_exit_policy"
HOLDOUT_START = pd.Timestamp("2026-06-28").date()


def main() -> None:
    training_summary = json.loads(
        (STUDY_DIR / "training_summary.json").read_text()
    )
    selected = ExitPolicyConfig(**training_summary["selected_config"])
    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)
    frame = pd.read_parquet(DATA_PATH)
    dates = pd.to_datetime(frame["game_date"]).dt.date
    holdout = frame[dates >= HOLDOUT_START].copy()
    result = evaluate_exit_policy(holdout, model, selected)

    deployment_enabled = bool(
        selected.enabled and result.pnl > 0 and result.roi > 0
    )
    deployment = ExitPolicyConfig(
        enabled=deployment_enabled,
        continuation_margin=selected.continuation_margin,
        confirmation_seconds=selected.confirmation_seconds,
    )
    CONFIG_PATH.write_text(json.dumps(asdict(deployment), indent=2))
    summary = {
        "selected_config": asdict(selected),
        "deployment_config": asdict(deployment),
        "holdout_start": str(HOLDOUT_START),
        "snapshots": len(holdout),
        "trajectories": int(holdout["trajectory_id"].nunique()),
        "games": int(holdout["game_pk"].nunique()),
        "trades": result.trades,
        "model_exits": result.model_exits,
        "settlements": result.settlements,
        "fees": result.fees,
        "capital": result.capital,
        "pnl": result.pnl,
        "roi": result.roi,
        "time_based_exit": False,
    }
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    (STUDY_DIR / "holdout_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    pd.DataFrame(asdict(record) for record in result.records).to_csv(
        STUDY_DIR / "holdout_policy_trades.csv", index=False
    )

    print("FITTED OPTIMAL-STOPPING HOLDOUT")
    print(f"Deployment enabled:    {deployment_enabled}")
    print(f"Continuation margin:   {selected.continuation_margin:.1%}")
    print(f"Confirmation:          {selected.confirmation_seconds:g} seconds")
    print("Time-based exit:       none")
    print(f"Games:                 {holdout['game_pk'].nunique():,}")
    print(f"Position trajectories: {holdout['trajectory_id'].nunique():,}")
    print(f"Accepted trades:       {result.trades:,}")
    print(f"Learned exits:         {result.model_exits:,}")
    print(f"Settlements:           {result.settlements:,}")
    print(f"Fees:                  ${result.fees:,.2f}")
    print(f"Capital:               ${result.capital:,.2f}")
    print(f"Net PnL:               ${result.pnl:,.2f}")
    print(f"ROI:                   {result.roi:.2%}")


if __name__ == "__main__":
    main()
