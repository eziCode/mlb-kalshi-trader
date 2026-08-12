"""Load the frozen model and score prepared decision rows."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

from catboost import CatBoostClassifier, CatBoostRegressor
import numpy as np
import pandas as pd

from settlement_value_strategy.strategy import (
    MispricingConfig, _expit, _logit, market_adjusted_probability,
    entry_state_allowed, mispricing_feature_frame, model_signal,
)


ROOT = Path(__file__).resolve().parent
PREDICTION_THREAD_COUNT = 1


class MispricingPredictor:
    def __init__(self, root: Path = ROOT):
        self.root = Path(root)
        live_config = self.root / "model/live_config.json"
        config_path = live_config if live_config.exists() else self.root / "model/config.json"
        raw = json.loads(config_path.read_text())
        self.model_kind = raw.get("model_kind", "settlement_classifier")
        self.residual_shrinkage = float(raw.get("residual_shrinkage", 1.0))
        self.maximum_logit_move = float(raw.get("maximum_logit_move", .5))
        if self.model_kind in {
            "local_state_probability", "anchored_state_probability",
        }:
            self.model = None
            model_name = None
        elif self.model_kind == "latency_residual":
            self.model = CatBoostRegressor()
            model_name = raw.get("model_file", "latency_value.cbm")
        else:
            self.model = CatBoostClassifier()
            model_name = raw.get("model_file", "settlement_value.cbm")
        if model_name is not None:
            model_path = self.root / "model" / model_name
            expected_model_hash = raw.get("model_sha256")
            actual_model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
            if expected_model_hash and actual_model_hash != expected_model_hash:
                raise RuntimeError(
                    "Settlement-value model hash does not match live_config.json"
                )
            self.model.load_model(model_path)
        self.calibration = json.loads(
            (self.root / "model/calibration.json").read_text()
        )
        self.config = MispricingConfig(**{
            key: value for key, value in raw.items()
            if key in MispricingConfig.__dataclass_fields__
        })

    def probability(self, rows: pd.DataFrame) -> np.ndarray:
        if self.model_kind == "anchored_state_probability":
            return np.clip(
                rows["anchored_state_target"].to_numpy(float),
                1e-6, 1 - 1e-6,
            )
        if self.model_kind == "local_state_probability":
            return np.clip(
                rows["local_fair_after"].to_numpy(float), 1e-6, 1 - 1e-6
            )
        if self.model_kind == "latency_residual":
            residual = np.clip(
                self.model.predict(
                    mispricing_feature_frame(rows),
                    thread_count=PREDICTION_THREAD_COUNT,
                ),
                -self.maximum_logit_move,
                self.maximum_logit_move,
            )
            return _expit(
                _logit(rows["market_home_price"].to_numpy(float))
                + self.residual_shrinkage * residual
            )
        raw = np.clip(
            self.model.predict_proba(
                mispricing_feature_frame(rows),
                thread_count=PREDICTION_THREAD_COUNT,
            )[:, 1],
            1e-6, 1 - 1e-6,
        )
        return market_adjusted_probability(
            raw, rows["market_home_price"].to_numpy(float), self.calibration
        )

    def decision(self, row: dict) -> dict:
        frame = pd.DataFrame([row])
        probability = float(self.probability(frame)[0])
        market = float(row["market_home_price"])
        side, expected_pnl, edge, eligible = model_signal(
            probability, market, self.config
        )
        eligible = eligible and entry_state_allowed(row, self.config)
        return {
            "settlement_probability": probability,
            "side": side,
            "expected_pnl": float(expected_pnl),
            "probability_edge": float(edge),
            "eligible": bool(eligible),
        }
