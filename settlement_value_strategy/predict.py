"""Load the frozen model and score prepared decision rows."""

from __future__ import annotations

import json
from pathlib import Path

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd

from settlement_value_strategy.strategy import (
    MispricingConfig, market_adjusted_probability, mispricing_feature_frame,
    model_signal,
)


ROOT = Path(__file__).resolve().parent
PREDICTION_THREAD_COUNT = 1


class MispricingPredictor:
    def __init__(self, root: Path = ROOT):
        self.root = Path(root)
        self.model = CatBoostClassifier()
        self.model.load_model(self.root / "model/settlement_value.cbm")
        self.calibration = json.loads(
            (self.root / "model/calibration.json").read_text()
        )
        raw = json.loads((self.root / "model/config.json").read_text())
        self.config = MispricingConfig(**{
            key: value for key, value in raw.items()
            if key in MispricingConfig.__dataclass_fields__
        })

    def probability(self, rows: pd.DataFrame) -> np.ndarray:
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
        return {
            "settlement_probability": probability,
            "side": side,
            "expected_pnl": float(expected_pnl),
            "probability_edge": float(edge),
            "eligible": bool(eligible),
        }
