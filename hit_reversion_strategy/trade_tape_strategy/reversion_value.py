"""Causal feature contract and scorer for incremental reversion entries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from catboost import CatBoostClassifier, CatBoostRegressor
import numpy as np
import pandas as pd

from .strategy import estimated_round_trip_fee_per_contract


CATEGORICAL_FEATURES = ("event_type",)
NUMERIC_FEATURES = (
    "inning", "outs", "contract_score_diff", "batting_is_contract_side",
    "runner_on_first", "runner_on_second", "runner_on_third",
    "contract_fair_before", "contract_fair_after",
    "batting_fair_move", "side_fair_move",
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
    side_sign = 1.0 if side == "yes" else -1.0
    batting_is_contract_side = bool(batting_home) == (side == "yes")
    return {
        "event_type": str(event_type), "inning": float(inning),
        "outs": float(outs),
        "contract_score_diff": float(score_diff) * side_sign,
        "batting_is_contract_side": float(batting_is_contract_side),
        "runner_on_first": float(runner_on_first),
        "runner_on_second": float(runner_on_second),
        "runner_on_third": float(runner_on_third),
        "contract_fair_before": (
            float(fair_before) if side == "yes" else 1.0 - float(fair_before)
        ),
        "contract_fair_after": (
            float(fair_after) if side == "yes" else 1.0 - float(fair_after)
        ),
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
    def __init__(
        self, model_path: Path, metadata_path: Path,
        config_path: Path | None = None,
        latency_profile_path: Path | None = None,
    ):
        self.model = CatBoostRegressor()
        self.model.load_model(str(model_path))
        self.metadata = json.loads(metadata_path.read_text())
        model_hash = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
        if model_hash != self.metadata.get("model_sha256"):
            raise RuntimeError("Reversion-value model/metadata hash mismatch")
        if tuple(self.metadata["model_features"]) != MODEL_FEATURES:
            raise RuntimeError("Reversion-value model feature contract mismatch")
        if self.metadata.get("entry_gate") != (
            "direct_value_model_required_for_every_entry"
        ):
            raise RuntimeError("Reversion-value entry gate is stale or unsafe")
        for path, metadata_key in (
            (config_path, "policy_config_sha256"),
            (latency_profile_path, "latency_profile_sha256"),
        ):
            if path is None:
                continue
            actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if actual != self.metadata.get(metadata_key):
                raise RuntimeError(
                    f"Reversion-value policy mismatch: {metadata_key}"
                )
        self.threshold = float(self.metadata["prediction_threshold"])

    def predict(self, features: dict[str, object]) -> float:
        frame = pd.DataFrame([features], columns=MODEL_FEATURES)
        return float(self.model.predict(frame)[0])

    def accepts(self, features: dict[str, object]) -> tuple[bool, float]:
        prediction = self.predict(features)
        metadata = getattr(self, "metadata", {})
        minimum_edge = float(metadata.get("minimum_entry_net_edge", -1.0))
        excluded = set(metadata.get("excluded_event_types", []))
        accepted = (
            prediction >= self.threshold
            and float(features["entry_net_edge"]) >= minimum_edge
            and str(features["event_type"]) not in excluded
        )
        return accepted, prediction


class CompetingRisksModel:
    """Versioned causal ensemble with an unconditional legacy-core bypass."""

    OUTCOMES = ("fast_reversion", "slow_reversion", "timeout", "settlement")

    def __init__(self, model_dir: Path, metadata_path: Path):
        self.metadata = json.loads(metadata_path.read_text())
        if tuple(self.metadata["model_features"]) != MODEL_FEATURES:
            raise RuntimeError("Competing-risks feature contract mismatch")
        if self.metadata.get("entry_gate") != "core_bypass_then_competing_risks":
            raise RuntimeError("Competing-risks entry gate is stale or unsafe")
        self.core_edge = float(self.metadata["core_edge"])
        self.threshold = 0.0
        self.models = []
        for seed in self.metadata["seeds"]:
            paths = {
                "risk": model_dir / f"competing_risks_{seed}_risk.cbm",
                "profit": model_dir / f"competing_risks_{seed}_profit.cbm",
                "downside": model_dir / f"competing_risks_{seed}_downside.cbm",
                "severe_loss": model_dir / f"competing_risks_{seed}_severe_loss.cbm",
                **{
                    outcome: model_dir / f"competing_risks_{seed}_{outcome}.cbm"
                    for outcome in self.OUTCOMES
                },
            }
            for key, path in paths.items():
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                if actual != self.metadata["model_sha256"][str(seed)][key]:
                    raise RuntimeError(f"Competing-risks model hash mismatch: {path.name}")
            risk, profit = CatBoostClassifier(), CatBoostClassifier()
            severe_loss = CatBoostClassifier()
            downside = CatBoostRegressor()
            risk.load_model(str(paths["risk"]))
            profit.load_model(str(paths["profit"]))
            downside.load_model(str(paths["downside"]))
            severe_loss.load_model(str(paths["severe_loss"]))
            conditional = {}
            for outcome in self.OUTCOMES:
                conditional[outcome] = CatBoostRegressor()
                conditional[outcome].load_model(str(paths[outcome]))
            self.models.append((risk, profit, downside, severe_loss, conditional))

    def accepts(self, features: dict[str, object]) -> tuple[bool, float]:
        edge = float(features["entry_net_edge"])
        if edge >= self.core_edge:
            return True, edge
        frame = pd.DataFrame([features], columns=MODEL_FEATURES)
        expected, reversion, profitable, downside, severe = [], [], [], [], []
        for risk, profit, quantile, severe_loss, conditional in self.models:
            probabilities = risk.predict_proba(frame)
            indices = {str(name): i for i, name in enumerate(risk.classes_)}
            expected.append(sum(
                probabilities[0, indices[outcome]]
                * float(conditional[outcome].predict(frame)[0])
                for outcome in self.OUTCOMES
            ))
            reversion.append(
                probabilities[0, indices["fast_reversion"]]
                + probabilities[0, indices["slow_reversion"]]
            )
            profitable.append(float(profit.predict_proba(frame)[0, 1]))
            downside.append(float(quantile.predict(frame)[0]))
            severe.append(float(severe_loss.predict_proba(frame)[0, 1]))
        lower_bound = float(np.mean(expected) - np.std(expected))
        accepted = bool(
            lower_bound > 0.0 and np.mean(reversion) > 0.5
            and np.mean(profitable) > 0.5 and np.mean(severe) < 0.25
        )
        return accepted, lower_bound
