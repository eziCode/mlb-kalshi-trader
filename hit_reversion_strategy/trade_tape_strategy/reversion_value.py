"""Causal feature contract and scorer for incremental reversion entries."""

from __future__ import annotations

import json
from pathlib import Path

from catboost import CatBoostRegressor
import pandas as pd

from .strategy import estimated_round_trip_fee_per_contract


CATEGORICAL_FEATURES = ("event_type", "side")
NUMERIC_FEATURES = (
    "inning", "inning_topbot", "outs", "score_diff",
    "runner_on_first", "runner_on_second", "runner_on_third",
    "fair_before", "fair_after", "batting_fair_move", "side_fair_move",
    "entry_price", "target_price", "entry_net_edge",
    "market_move_since_pitch", "event_detection_latency_seconds",
    "entry_lag_seconds",
)
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def reversion_feature_row(
    *, event_type: str, side: str, inning: float, inning_topbot: float,
    outs: float, score_diff: float, runner_on_first: float,
    runner_on_second: float, runner_on_third: float, fair_before: float,
    fair_after: float, batting_home: bool, entry_price: float,
    target_price: float, pre_market_price: float,
    event_detection_latency_seconds: float, entry_lag_seconds: float,
) -> dict[str, object]:
    fair_move = float(fair_after) - float(fair_before)
    return {
        "event_type": str(event_type), "side": str(side),
        "inning": float(inning), "inning_topbot": float(inning_topbot),
        "outs": float(outs), "score_diff": float(score_diff),
        "runner_on_first": float(runner_on_first),
        "runner_on_second": float(runner_on_second),
        "runner_on_third": float(runner_on_third),
        "fair_before": float(fair_before), "fair_after": float(fair_after),
        "batting_fair_move": fair_move * (1.0 if batting_home else -1.0),
        "side_fair_move": fair_move * (1.0 if side == "yes" else -1.0),
        "entry_price": float(entry_price), "target_price": float(target_price),
        "entry_net_edge": (
            float(target_price) - float(entry_price)
            - estimated_round_trip_fee_per_contract(float(entry_price))
        ),
        "market_move_since_pitch": (
            float(entry_price) - float(pre_market_price)
        ),
        "event_detection_latency_seconds": float(
            event_detection_latency_seconds
        ),
        "entry_lag_seconds": float(entry_lag_seconds),
    }


class ReversionValueModel:
    def __init__(self, model_path: Path, metadata_path: Path):
        self.model = CatBoostRegressor()
        self.model.load_model(str(model_path))
        self.metadata = json.loads(metadata_path.read_text())
        if tuple(self.metadata["model_features"]) != MODEL_FEATURES:
            raise RuntimeError("Reversion-value model feature contract mismatch")
        self.threshold = float(self.metadata["prediction_threshold"])
        self.proven_edge = float(self.metadata["proven_edge"])

    def predict(self, features: dict[str, object]) -> float:
        frame = pd.DataFrame([features], columns=MODEL_FEATURES)
        return float(self.model.predict(frame)[0])

    def accepts(self, features: dict[str, object]) -> tuple[bool, float]:
        if float(features["entry_net_edge"]) >= self.proven_edge:
            return True, float("nan")
        prediction = self.predict(features)
        return prediction >= self.threshold, prediction
