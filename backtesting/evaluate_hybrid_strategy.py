"""One-shot outer-holdout evaluation of the event-conditioned hybrid."""

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

from mlb_kalshi.hybrid import (  # noqa: E402
    HybridConfig,
    add_event_targets,
    simulate_hybrid,
)
from mlb_kalshi.strategy import state_feature_frame, validate_market_prices  # noqa: E402


TEST_DATA = PROJECT_ROOT / "data/processed/test/test_dataset.parquet"
MODEL_DIR = PROJECT_ROOT / "models/market_reaction_model"
CONFIG_PATH = MODEL_DIR / "hybrid_config.json"
STUDY_DIR = PROJECT_ROOT / "studies/hybrid_event_strategy"


def add_fair_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    model = CatBoostClassifier()
    model.load_model(MODEL_DIR / "local_win_expectancy.cbm")
    result = frame.copy()
    result["fair_prob"] = model.predict_proba(
        state_feature_frame(result)
    )[:, 1]
    return result


def main() -> None:
    config = HybridConfig.from_json(CONFIG_PATH)
    frame = pd.read_parquet(TEST_DATA)
    validate_market_prices(frame)
    frame = add_fair_probabilities(frame)
    prepared = add_event_targets(frame, config)
    result = simulate_hybrid(frame, config)

    event_rows = prepared[prepared["hybrid_event"]]
    summary = {
        "config": asdict(config),
        "decision_rows": len(frame),
        "games": int(frame["game_pk"].nunique()),
        "isolated_hit_events": len(event_rows),
        "trades": result.trades,
        "yes_trades": result.yes_trades,
        "no_trades": result.no_trades,
        "reversion_exits": result.early_exits,
        "timed_exits": result.timed_exits,
        "invalidated_exits": result.invalidated_exits,
        "settlements": result.settlements,
        "fees": result.fees,
        "capital": result.capital,
        "pnl": result.pnl,
        "roi": result.roi,
    }
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    (STUDY_DIR / "holdout_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print("HYBRID EVENT-RESIDUAL HOLDOUT")
    print(f"Live enabled:         {config.enabled}")
    print(f"Minimum edge:         {config.minimum_edge:.1%}")
    print(f"Maximum hold:         {config.max_hold_minutes:.0f} minutes")
    print(f"Decision rows:        {len(frame):,}")
    print(f"Games:                {frame['game_pk'].nunique():,}")
    print(f"Isolated hit events:  {len(event_rows):,}")
    print(f"Trades:               {result.trades:,}")
    print(f"YES / NO:             {result.yes_trades:,} / {result.no_trades:,}")
    print(f"Reversion exits:      {result.early_exits:,}")
    print(f"Timed exits:          {result.timed_exits:,}")
    print(f"Event-change exits:   {result.invalidated_exits:,}")
    print(f"Settlements:          {result.settlements:,}")
    print(f"Fees:                 ${result.fees:,.2f}")
    print(f"Capital:              ${result.capital:,.2f}")
    print(f"Net PnL:              ${result.pnl:,.2f}")
    print(f"ROI:                  {result.roi:.2%}")


if __name__ == "__main__":
    main()
