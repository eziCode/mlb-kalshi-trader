"""Tune hybrid execution parameters without looking at the outer holdout."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

from catboost import CatBoostClassifier
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlb_kalshi.hybrid import HybridConfig, simulate_hybrid  # noqa: E402
from mlb_kalshi.strategy import state_feature_frame, validate_market_prices  # noqa: E402
from models.train_market_reaction_model import params, state_pool  # noqa: E402


TRAIN_DATA = PROJECT_ROOT / "data/processed/train/training_dataset.parquet"
MODEL_DIR = PROJECT_ROOT / "models/market_reaction_model"
CONFIG_PATH = MODEL_DIR / "hybrid_config.json"
STUDY_DIR = PROJECT_ROOT / "studies/hybrid_event_strategy"


def chronological_calibration_split(frame: pd.DataFrame, fraction: float = 0.75):
    dates = pd.to_datetime(frame["game_date"]).dt.date
    unique_dates = sorted(dates.unique())
    cutoff = unique_dates[int(len(unique_dates) * fraction)]
    return frame[dates < cutoff].copy(), frame[dates >= cutoff].copy(), cutoff


def main() -> None:
    frame = pd.read_parquet(TRAIN_DATA).sort_values("decision_time")
    validate_market_prices(frame)
    state_fit, tune, cutoff = chronological_calibration_split(frame)

    metadata = json.loads((MODEL_DIR / "metadata.json").read_text())
    iterations = int(metadata["state_iterations"])
    state_model = CatBoostClassifier(**params(iterations))
    state_model.fit(state_pool(state_fit))
    tune["fair_prob"] = state_model.predict_proba(
        state_feature_frame(tune)
    )[:, 1]

    rows = []
    for minimum_edge in [
        0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.10,
        0.125, 0.15, 0.175, 0.20, 0.25,
    ]:
        for max_hold_minutes in [2.0, 3.0, 5.0, 8.0, 10.0]:
            config = HybridConfig(
                minimum_edge=minimum_edge,
                max_hold_minutes=max_hold_minutes,
            )
            result = simulate_hybrid(tune, config)
            rows.append({
                "minimum_edge": minimum_edge,
                "max_hold_minutes": max_hold_minutes,
                "trades": result.trades,
                "yes_trades": result.yes_trades,
                "no_trades": result.no_trades,
                "fees": result.fees,
                "capital": result.capital,
                "pnl": result.pnl,
                "roi": result.roi,
            })

    grid = pd.DataFrame(rows).sort_values(
        ["roi", "pnl", "trades"], ascending=False
    )
    eligible = grid[grid["trades"] >= 30]
    selection = eligible.iloc[0] if not eligible.empty else grid.iloc[0]
    enabled = bool(selection["pnl"] > 0 and selection["roi"] > 0)
    selected = HybridConfig(
        enabled=enabled,
        minimum_edge=float(selection["minimum_edge"]),
        max_hold_minutes=float(selection["max_hold_minutes"]),
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    selected.to_json(CONFIG_PATH)
    grid.to_csv(STUDY_DIR / "tuning_grid.csv", index=False)
    report = {
        "calibration_cutoff": str(cutoff),
        "state_fit_rows": len(state_fit),
        "tuning_rows": len(tune),
        "selection_rule": "maximum net ROI among configurations with >=30 trades",
        "selected_config": asdict(selected),
        "selected_tuning_result": selection.to_dict(),
        "outer_test_used": False,
    }
    (STUDY_DIR / "tuning_summary.json").write_text(
        json.dumps(report, indent=2)
    )

    print(f"State calibration rows: {len(state_fit):,}")
    print(f"Hybrid tuning rows:     {len(tune):,}")
    print("Top tuning configurations:")
    print(grid.head(10).to_string(index=False, formatters={
        "roi": "{:.2%}".format,
        "pnl": "${:,.2f}".format,
        "fees": "${:,.2f}".format,
    }))
    print(f"Saved {CONFIG_PATH} (live enabled={enabled})")


if __name__ == "__main__":
    main()
