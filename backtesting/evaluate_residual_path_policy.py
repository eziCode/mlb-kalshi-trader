"""Evaluate the frozen residual-path ML policy on development holdout."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from catboost import CatBoostClassifier, CatBoostRegressor
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlb_kalshi.residual_path import (  # noqa: E402
    evaluate_path_policy, residual_path_feature_frame,
)


MODEL_DIR = PROJECT_ROOT / "models/market_reaction_model/residual_path"
STUDY_DIR = PROJECT_ROOT / "studies/residual_path_policy"
HOLDOUT_START = pd.Timestamp("2026-06-28").date()


def main() -> None:
    config = json.loads((MODEL_DIR / "policy_config.json").read_text())
    calibration = json.loads(
        (MODEL_DIR / "maker_fill_calibration.json").read_text()
    )
    frame = pd.read_parquet(STUDY_DIR / "residual_paths.parquet")
    frame["game_date"] = pd.to_datetime(frame.game_date).dt.date
    frame = frame[frame.game_date >= HOLDOUT_START].copy()
    features = residual_path_feature_frame(frame)
    fill_model = CatBoostClassifier()
    fill_model.load_model(MODEL_DIR / "maker_fill.cbm")
    raw = np.clip(fill_model.predict_proba(features)[:, 1], 1e-6, 1 - 1e-6)
    logit = np.log(raw / (1 - raw))
    fill_probability = 1 / (1 + np.exp(-(
        calibration["intercept"] + calibration["coefficient"] * logit
    )))
    horizon = int(config["horizon_seconds"])
    pnl_model = CatBoostRegressor()
    pnl_model.load_model(MODEL_DIR / f"conditional_pnl_{horizon}s.cbm")
    predicted_pnl = pnl_model.predict(features)
    result = evaluate_path_policy(
        frame, fill_probability, predicted_pnl, horizon,
        config["minimum_fill_probability"], config["minimum_expected_pnl"],
    )
    passed = bool(result["trades"] >= 20 and result["pnl"] > 0 and result["roi"] > .10)
    config["validation_passed"] = passed
    config["enabled"] = False
    (MODEL_DIR / "policy_config.json").write_text(json.dumps(config, indent=2))
    summary = {
        "holdout_start": str(HOLDOUT_START), "signals": len(frame),
        "config": config,
        **{key: result[key] for key in ["attempts", "trades", "pnl", "capital", "roi"]},
        "validation_passed": passed,
    }
    (STUDY_DIR / "holdout_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(result["records"]).to_csv(
        STUDY_DIR / "holdout_trades.csv", index=False
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
