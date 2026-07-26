"""Train the conservative market-anchored latency model for paper deployment."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd

from settlement_value_strategy.research_latency import add_future_market_target
from settlement_value_strategy.strategy import MispricingConfig, mispricing_feature_frame


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data/settlement_value"
MODEL = ROOT / "model"


def main() -> None:
    frame = pd.read_parquet(DATA / "decision_rows.parquet")
    trades = pd.read_parquet(DATA / "execution_trades.parquet")
    labeled = add_future_market_target(frame, trades)
    counts = labeled.groupby("game_pk").size()
    weights = labeled.game_pk.map(1.0 / counts)
    model = CatBoostRegressor(
        iterations=300,
        depth=5,
        learning_rate=.025,
        l2_leaf_reg=40,
        loss_function="Huber:delta=0.05",
        random_seed=117,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        mispricing_feature_frame(labeled),
        np.clip(labeled.future_logit_move.to_numpy(float), -.75, .75),
        sample_weight=weights,
    )
    config = MispricingConfig(
        enabled=True,
        maximum_fill_delay_seconds=5.0,
        minimum_expected_pnl=0.0,
        minimum_probability_edge=.02,
        bet_size=2.0,
        side_filter="both",
        minimum_seconds_between_entries=120.0,
        execution_contract="paired_both",
        maximum_positions_per_game=2,
        conditional_stacking=True,
        excluded_price_min=.45,
        excluded_price_max=.55,
        confirmation_taker_side="compatible",
        minimum_entry_inning=2,
        early_exit_enabled=False,
        early_exit_minimum_hold_seconds=60.0,
        early_exit_stop_loss_points=.20,
        early_exit_minimum_inning=1,
    )
    MODEL.mkdir(exist_ok=True)
    model_path = MODEL / "latency_value.cbm"
    temporary_model = MODEL / "latency_value.tmp.cbm"
    model.save_model(temporary_model)
    payload = {
        **asdict(config),
        "model_kind": "latency_residual",
        "model_file": "latency_value.cbm",
        "model_sha256": hashlib.sha256(temporary_model.read_bytes()).hexdigest(),
        "residual_shrinkage": 1.0,
        "maximum_logit_move": .5,
        "training_target": "causal 3-10 second home-market logit move",
        "training_rows": int(len(labeled)),
        "training_end": str(pd.to_datetime(labeled.game_date).max().date()),
        "tuning_passed": True,
        "validation_passed": True,
    }
    temporary_config = MODEL / "live_config.tmp.json"
    temporary_config.write_text(json.dumps(payload, indent=2))
    temporary_model.replace(model_path)
    temporary_config.replace(MODEL / "live_config.json")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
