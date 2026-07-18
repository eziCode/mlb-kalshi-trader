"""One-time chronological outer-holdout evaluation of state reversion."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
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
        "thesis_invalidations": result.thesis_invalidations,
        "timeout_exits": result.timeout_exits,
        "settlements": result.settlements, "fees": result.fees,
        "capital": result.capital, "pnl": result.pnl, "roi": result.roi,
        "mean_predicted_pnl": float(np.mean(expected_pnl)),
        "segment_report": "holdout_segments.csv",
        "maker_fee_sensitivity": "holdout_maker_fee_sensitivity.csv",
        "validation_passed": validation_passed,
    }
    (STUDY_DIR / "holdout_summary.json").write_text(json.dumps(summary, indent=2))
    # Keep the conventional study entry point synchronized; this replaces the
    # obsolete expected-PnL-regression summary from the earlier experiment.
    (STUDY_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    records = pd.DataFrame(result.records)
    records.to_csv(STUDY_DIR / "holdout_trades.csv", index=False)
    segment_rows = []
    if not records.empty:
        records["inning_group"] = pd.cut(
            records["inning_after"], [0, 3, 6, 9, np.inf],
            labels=["1-3", "4-6", "7-9", "extras"],
        )
        records["entry_price_group"] = pd.cut(
            records["entry_price"], [0, .25, .50, .75, 1.0],
            labels=["0-.25", ".25-.50", ".50-.75", ".75-1"],
        )
        for dimension in [
            "side", "completed_plate_appearance", "exit_reason",
            "inning_group", "entry_price_group",
        ]:
            for value, group in records.groupby(dimension, observed=True):
                segment_rows.append({
                    "dimension": dimension, "value": str(value),
                    "trades": len(group), "pnl": float(group.pnl.sum()),
                    "roi": float(group.pnl.sum() / (10.0 * len(group))),
                    "win_rate": float(group.pnl.gt(0).mean()),
                })
    pd.DataFrame(segment_rows).to_csv(
        STUDY_DIR / "holdout_segments.csv", index=False
    )

    fee_rows = []
    for rate in [0.0, 0.01, 0.02, 0.035, 0.07]:
        total_fee = 0.0
        if not records.empty:
            for row in records.itertuples(index=False):
                entry_raw = rate * row.contracts * row.entry_price * (1 - row.entry_price)
                entry_fee = math.ceil(entry_raw * 100 - 1e-12) / 100
                exit_fee = 0.0
                if row.exit_reason != "settlement":
                    exit_raw = rate * row.contracts * row.exit_price * (1 - row.exit_price)
                    exit_fee = math.ceil(exit_raw * 100 - 1e-12) / 100
                total_fee += entry_fee + exit_fee
        net = result.pnl - total_fee
        capital = 10.0 * len(records) + total_fee
        fee_rows.append({
            "maker_fee_rate": rate, "trades": len(records),
            "fees": total_fee, "pnl": net,
            "roi": net / capital if capital else 0.0,
        })
    pd.DataFrame(fee_rows).to_csv(
        STUDY_DIR / "holdout_maker_fee_sensitivity.csv", index=False
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
