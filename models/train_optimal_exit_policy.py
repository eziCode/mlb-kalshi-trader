"""Train and chronologically tune the fitted continuation-value exit policy."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlb_kalshi.optimal_exit import (  # noqa: E402
    EXIT_FEATURES,
    ExitPolicyConfig,
    evaluate_exit_policy,
    train_continuation_model,
)


DATA_PATH = (
    PROJECT_ROOT / "data/processed/trade_tape/optimal_exit_trajectories.parquet"
)
MODEL_DIR = PROJECT_ROOT / "models/market_reaction_model"
MODEL_PATH = MODEL_DIR / "optimal_exit_continuation.cbm"
CONFIG_PATH = MODEL_DIR / "optimal_exit_config.json"
STUDY_DIR = PROJECT_ROOT / "studies/optimal_exit_policy"
FIT_END = pd.Timestamp("2026-06-17").date()
HOLDOUT_START = pd.Timestamp("2026-06-28").date()


def main() -> None:
    frame = pd.read_parquet(DATA_PATH)
    dates = pd.to_datetime(frame["game_date"]).dt.date
    fit = frame[dates < FIT_END].copy()
    validation = frame[(dates >= FIT_END) & (dates < HOLDOUT_START)].copy()
    if fit.empty or validation.empty:
        raise RuntimeError("Chronological fit/validation split is empty")
    print(
        f"Fit snapshots/trajectories: {len(fit):,} / "
        f"{fit['trajectory_id'].nunique():,}"
    )
    print(
        f"Validation snapshots/trajectories: {len(validation):,} / "
        f"{validation['trajectory_id'].nunique():,}"
    )

    model = train_continuation_model(fit, policy_iterations=5)
    rows = []
    for margin in [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15]:
        for confirmation in [1.0, 2.0, 3.0, 5.0]:
            config = ExitPolicyConfig(
                continuation_margin=margin,
                confirmation_seconds=confirmation,
            )
            result = evaluate_exit_policy(validation, model, config)
            rows.append({
                "continuation_margin": margin,
                "confirmation_seconds": confirmation,
                "trades": result.trades,
                "model_exits": result.model_exits,
                "settlements": result.settlements,
                "fees": result.fees,
                "capital": result.capital,
                "pnl": result.pnl,
                "roi": result.roi,
            })
    grid = pd.DataFrame(rows).sort_values(
        ["roi", "pnl", "trades"], ascending=False
    )
    eligible = grid[
        (grid["trades"] >= 30) & (grid["model_exits"] >= 20)
    ]
    selection = eligible.iloc[0] if not eligible.empty else grid.iloc[0]
    enabled = bool(selection["pnl"] > 0 and selection["roi"] > 0)
    config = ExitPolicyConfig(
        enabled=enabled,
        continuation_margin=float(selection["continuation_margin"]),
        confirmation_seconds=float(selection["confirmation_seconds"]),
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    CONFIG_PATH.write_text(json.dumps(asdict(config), indent=2))
    grid.to_csv(STUDY_DIR / "validation_grid.csv", index=False)
    summary = {
        "fit_end_exclusive": str(FIT_END),
        "validation_start": str(FIT_END),
        "validation_end": str(HOLDOUT_START - pd.Timedelta(days=1)),
        "fit_snapshots": len(fit),
        "fit_trajectories": int(fit["trajectory_id"].nunique()),
        "validation_snapshots": len(validation),
        "validation_trajectories": int(validation["trajectory_id"].nunique()),
        "features": list(EXIT_FEATURES),
        "selection_rule": (
            "maximum ROI with >=30 trades and >=20 learned model exits"
        ),
        "selected_config": asdict(config),
        "selected_validation_result": selection.to_dict(),
        "time_based_exit": False,
        "outer_holdout_used": False,
    }
    (STUDY_DIR / "training_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print("Top validation configurations:")
    print(grid.head(12).to_string(index=False, formatters={
        "roi": "{:.2%}".format,
        "pnl": "${:,.2f}".format,
        "fees": "${:,.2f}".format,
    }))
    print(f"Saved {MODEL_PATH}")
    print(f"Saved {CONFIG_PATH} (validation enabled={enabled})")


if __name__ == "__main__":
    main()
