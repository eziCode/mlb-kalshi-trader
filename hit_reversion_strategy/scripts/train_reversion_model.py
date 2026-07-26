"""Train a causal model of realized event-reversion PnL per contract.

The opportunity set is intentionally generated with a permissive edge floor.
The model then learns which of those executable opportunities historically
paid after entry/exit fees.  The outer holdout is never used for fitting or
threshold selection.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool


PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.backtest import (  # noqa: E402
    AWAY_TRADE_COLUMNS,
    OUTER_HOLDOUT_START,
    STATE_COLUMNS,
    TRADE_COLUMNS,
    apply_live_paired_execution_prices,
    apply_publication_latency,
)
from trade_tape_strategy.core import (  # noqa: E402
    TradeTapeConfig,
    simulate_trade_tape,
)
from trade_tape_strategy.strategy import (  # noqa: E402
    estimated_round_trip_fee_per_contract,
)


DATA = REPOSITORY / "data/shared"
CONFIG_PATH = PROJECT / "models/trade_tape_config.json"
MODEL_PATH = PROJECT / "models/reversion_value.cbm"
METADATA_PATH = PROJECT / "models/reversion_value.metadata.json"
LATENCY_PROFILE_PATH = PROJECT / "models/event_observation_latency.json"
OPPORTUNITY_EDGE = 0.01
MINIMUM_DEPLOYMENT_ROI = 0.015

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


def _logit(value: pd.Series) -> np.ndarray:
    clipped = np.clip(value.astype(float).to_numpy(), 1e-4, 1 - 1e-4)
    return np.log(clipped / (1.0 - clipped))


def build_opportunities(
    trades: pd.DataFrame,
    away_trades: pd.DataFrame,
    updates: pd.DataFrame,
    config: TradeTapeConfig,
) -> pd.DataFrame:
    """Replay permissive entries and attach only entry-time information."""
    games = set(trades.game_pk.unique())
    paired = apply_live_paired_execution_prices(
        trades.copy(), away_trades[away_trades.game_pk.isin(games)].copy(),
    )
    delayed_updates = apply_publication_latency(
        updates[updates.game_pk.isin(games)].copy()
    )
    permissive = replace(
        config,
        minimum_edge=OPPORTUNITY_EDGE,
        minimum_seconds_between_entries=0.0,
        direct_value_model_enabled=False,
    )
    result = simulate_trade_tape(paired, delayed_updates, permissive)
    records = pd.DataFrame(asdict(record) for record in result.records)
    if records.empty:
        return records

    event_columns = [
        "game_pk", "at_bat_number", "pitch_number", "game_date",
        "pitch_start_time", "pitch_end_time", "event_available_time",
        "completed_event_batting_home", "fair_before", "fair_after",
        "inning_after", "inning_topbot_after", "outs_when_up_after",
        "score_diff_after", "runner_on_first_after",
        "runner_on_second_after", "runner_on_third_after",
    ]
    events = delayed_updates[event_columns].drop_duplicates(
        ["game_pk", "at_bat_number", "pitch_number"], keep="last"
    )
    frame = records.merge(
        events,
        left_on=["game_pk", "trigger_at_bat", "trigger_pitch"],
        right_on=["game_pk", "at_bat_number", "pitch_number"],
        how="inner", validate="many_to_one",
    )
    fair_move = frame.fair_after.astype(float) - frame.fair_before.astype(float)
    batting_sign = np.where(frame.completed_event_batting_home, 1.0, -1.0)
    side_sign = np.where(frame.side == "yes", 1.0, -1.0)
    bounded_move = np.clip(
        _logit(frame.fair_after) - _logit(frame.fair_before),
        -float(config.maximum_event_log_odds_move),
        float(config.maximum_event_log_odds_move),
    )
    pre_market = 1.0 / (
        1.0 + np.exp(-(_logit(frame.anchor_target) - bounded_move))
    )
    pre_contract = np.where(frame.side == "yes", pre_market, 1.0 - pre_market)
    target_contract = np.where(
        frame.side == "yes", frame.anchor_target, 1.0 - frame.anchor_target
    )

    frame = frame.assign(
        event_type=frame.event_type.astype(str),
        side=frame.side.astype(str),
        inning=frame.inning_after.astype(float),
        inning_topbot=frame.inning_topbot_after.astype(float),
        outs=frame.outs_when_up_after.astype(float),
        score_diff=frame.score_diff_after.astype(float),
        runner_on_first=frame.runner_on_first_after.astype(float),
        runner_on_second=frame.runner_on_second_after.astype(float),
        runner_on_third=frame.runner_on_third_after.astype(float),
        batting_fair_move=fair_move * batting_sign,
        side_fair_move=fair_move * side_sign,
        target_price=target_contract,
        entry_net_edge=(
            target_contract - frame.entry_price.astype(float)
            - frame.entry_price.astype(float).map(
                estimated_round_trip_fee_per_contract
            )
        ),
        market_move_since_pitch=frame.entry_price.astype(float) - pre_contract,
        event_detection_latency_seconds=(
            pd.to_datetime(frame.event_available_time, utc=True)
            - pd.to_datetime(frame.pitch_end_time, utc=True)
        ).dt.total_seconds(),
        entry_lag_seconds=(
            pd.to_datetime(frame.entry_time, utc=True)
            - pd.to_datetime(frame.pitch_end_time, utc=True)
        ).dt.total_seconds(),
        pnl_per_contract=frame.pnl.astype(float) / frame.contracts.astype(float),
    )
    frame["game_date"] = pd.to_datetime(frame.game_date).dt.date
    return frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=list(MODEL_FEATURES) + ["pnl_per_contract"]
    )


def pool(frame: pd.DataFrame, label: bool = True) -> Pool:
    return Pool(
        frame.loc[:, MODEL_FEATURES],
        label=frame.pnl_per_contract if label else None,
        cat_features=list(CATEGORICAL_FEATURES),
    )


def metrics(frame: pd.DataFrame) -> dict:
    capital = float((frame.contracts * frame.entry_price).sum())
    game_pnl = frame.groupby("game_pk").pnl.sum()
    return {
        "games": int(frame.game_pk.nunique()),
        "trades": int(len(frame)),
        "pnl": float(frame.pnl.sum()),
        "capital": capital,
        "roi": float(frame.pnl.sum() / capital) if capital else 0.0,
        "pnl_without_best_game": (
            float(frame.pnl.sum() - game_pnl.max()) if len(game_pnl) else 0.0
        ),
    }


def main() -> None:
    config = TradeTapeConfig(**json.loads(CONFIG_PATH.read_text()))
    trades = pd.read_parquet(
        DATA / "home_market_trades.parquet", columns=TRADE_COLUMNS
    )
    away = pd.read_parquet(
        DATA / "away_market_trades.parquet", columns=AWAY_TRADE_COLUMNS
    )
    updates = pd.read_parquet(
        DATA / "state_updates.parquet", columns=STATE_COLUMNS
    )
    for frame in (trades, away, updates):
        frame["game_date"] = pd.to_datetime(frame.game_date).dt.date

    pre = trades[trades.game_date < OUTER_HOLDOUT_START].copy()
    holdout = trades[trades.game_date >= OUTER_HOLDOUT_START].copy()
    pre_rows = build_opportunities(pre, away, updates, config)
    holdout_rows = build_opportunities(holdout, away, updates, config)
    del trades, away, updates, pre, holdout
    gc.collect()
    dates = sorted(pre_rows.game_date.unique())
    validation_start = dates[int(len(dates) * 0.75)]
    fit = pre_rows[pre_rows.game_date < validation_start].copy()
    validation = pre_rows[pre_rows.game_date >= validation_start].copy()

    model = CatBoostRegressor(
        iterations=500, depth=5, learning_rate=0.03,
        loss_function="RMSE", l2_leaf_reg=20.0,
        random_seed=42, allow_writing_files=False, verbose=False,
    )
    model.fit(pool(fit), eval_set=pool(validation), early_stopping_rounds=75)
    validation["prediction"] = model.predict(pool(validation, label=False))
    thresholds = sorted(set(
        np.quantile(validation.prediction, np.linspace(0.0, 1.0, 41))
    ))
    thresholds.append(float(validation.prediction.max()) + 1e-9)
    candidates = []
    validation_games = int(pre[pre.game_date >= validation_start].game_pk.nunique())
    for threshold in thresholds:
        selected = validation[
            validation.prediction >= threshold
        ]
        item = metrics(selected)
        item["threshold"] = float(threshold)
        item["trades_per_game"] = item["trades"] / validation_games
        if (
            item["trades"] >= 30
            and item["roi"] >= MINIMUM_DEPLOYMENT_ROI
            and item["pnl_without_best_game"] > 0
        ):
            candidates.append(item)
    if not candidates:
        raise RuntimeError("No direct-model threshold passed validation gates")
    # The production objective is dollars per scheduled game, not activity.
    # With a fixed $2 live budget and a fixed validation slate, maximizing net
    # PnL is exactly maximizing EV/game.  Use ROI and concentration only as
    # safety gates, not as the ranking objective.
    selected = max(candidates, key=lambda row: (row["pnl"], row["roi"]))

    holdout_rows["prediction"] = model.predict(pool(holdout_rows, label=False))
    holdout_selected = holdout_rows[
        holdout_rows.prediction >= selected["threshold"]
    ]
    holdout_metrics = metrics(holdout_selected)
    holdout_games = int(holdout.game_pk.nunique())
    holdout_metrics["trades_per_game"] = (
        holdout_metrics["trades"] / holdout_games
    )
    deployable = bool(
        holdout_metrics["trades"] >= 30
        and holdout_metrics["roi"] >= MINIMUM_DEPLOYMENT_ROI
        and holdout_metrics["pnl_without_best_game"] > 0
    )
    metadata = {
        "model_features": list(MODEL_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "opportunity_edge": OPPORTUNITY_EDGE,
        "entry_gate": "direct_value_model_required_for_every_entry",
        "policy_config_sha256": hashlib.sha256(
            CONFIG_PATH.read_bytes()
        ).hexdigest(),
        "latency_profile_sha256": hashlib.sha256(
            LATENCY_PROFILE_PATH.read_bytes()
        ).hexdigest(),
        "minimum_deployment_roi": MINIMUM_DEPLOYMENT_ROI,
        "selection_objective": "maximum_net_pnl_per_scheduled_game",
        "validation_start": str(validation_start),
        "fit_rows": len(fit),
        "validation_rows": len(validation),
        "holdout_rows": len(holdout_rows),
        "prediction_threshold": selected["threshold"],
        "validation": selected,
        "holdout": holdout_metrics,
        "deployable": deployable,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = MODEL_PATH.with_name("reversion_value.tmp.cbm")
    temporary_metadata = METADATA_PATH.with_name(
        "reversion_value.metadata.tmp.json"
    )
    model.save_model(temporary_model)
    metadata["model_sha256"] = hashlib.sha256(
        temporary_model.read_bytes()
    ).hexdigest()
    temporary_metadata.write_text(json.dumps(metadata, indent=2))
    temporary_model.replace(MODEL_PATH)
    temporary_metadata.replace(METADATA_PATH)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
