"""One-time chronological outer-holdout evaluation of state reversion."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

from catboost import CatBoostClassifier, CatBoostRegressor
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlb_kalshi.state_overshoot import OvershootConfig, simulate_state_reversion  # noqa: E402
from mlb_kalshi.state_reversion import overshoot_reversion_feature_frame  # noqa: E402


MODEL_DIR = PROJECT_ROOT / "models/market_reaction_model"
STUDY_DIR = PROJECT_ROOT / "studies/state_reversion"
HOLDOUT_START = pd.Timestamp("2026-06-28").date()


def main() -> None:
    raw_config = json.loads((MODEL_DIR / "state_reversion_config.json").read_text())
    config = OvershootConfig(**{
        key: value for key, value in raw_config.items()
        if key in OvershootConfig.__dataclass_fields__
    })
    calibration = json.loads(
        (MODEL_DIR / raw_config["calibration_path"]).read_text()
    )
    examples = pd.read_parquet(STUDY_DIR / "overshoot_candidates.parquet")
    examples["game_date"] = pd.to_datetime(examples["game_date"]).dt.date
    holdout = examples[examples["game_date"] >= HOLDOUT_START].copy()
    model = CatBoostClassifier()
    model.load_model(MODEL_DIR / config.model_path)
    raw = np.clip(
        model.predict_proba(overshoot_reversion_feature_frame(holdout))[:, 1],
        1e-6, 1 - 1e-6,
    )
    logits = np.log(raw / (1 - raw))
    probabilities = 1 / (1 + np.exp(-(
        calibration["intercept"] + calibration["coefficient"] * logits
    )))
    features = overshoot_reversion_feature_frame(holdout)
    win_model = CatBoostRegressor()
    win_model.load_model(MODEL_DIR / raw_config["win_model_path"])
    loss_model = CatBoostRegressor()
    loss_model.load_model(MODEL_DIR / raw_config["loss_model_path"])
    predicted_win = np.maximum(win_model.predict(features), 0.0)
    predicted_loss = np.minimum(loss_model.predict(features), 0.0)
    raw_expected_pnl = (
        probabilities * predicted_win + (1 - probabilities) * predicted_loss
    )
    expected_pnl = (
        calibration.get("ev_intercept", 0.0)
        + calibration.get("ev_coefficient", 1.0) * raw_expected_pnl
    )
    result = simulate_state_reversion(
        holdout, probabilities, config, expected_pnls=expected_pnl
    )
    validation_passed = bool(result.accepted >= 20 and result.pnl > 0 and result.roi > 0)
    deployment = {**asdict(config), "enabled": bool(config.enabled and validation_passed)}
    raw_config.update(deployment)
    raw_config["validation_passed"] = validation_passed
    (MODEL_DIR / "state_reversion_config.json").write_text(json.dumps(raw_config, indent=2))
    summary = {
        "holdout_start": str(HOLDOUT_START), "games": int(holdout.game_pk.nunique()),
        "candidates": len(holdout), "selected_config": asdict(config),
        "deployment_config": deployment, "trades": result.accepted,
        "rejected": result.rejected, "reversion_exits": result.reversion_exits,
        "adverse_stop_exits": result.adverse_stop_exits,
        "timeout_exits": result.timeout_exits,
        "settlements": result.settlements, "fees": result.fees,
        "capital": result.capital, "pnl": result.pnl, "roi": result.roi,
        "mean_predicted_pnl": float(np.mean(expected_pnl)),
        "validation_passed": validation_passed,
    }
    (STUDY_DIR / "holdout_summary.json").write_text(json.dumps(summary, indent=2))
    # Keep the conventional study entry point synchronized; this replaces the
    # obsolete expected-PnL-regression summary from the earlier experiment.
    (STUDY_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(result.records).to_csv(STUDY_DIR / "holdout_trades.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
