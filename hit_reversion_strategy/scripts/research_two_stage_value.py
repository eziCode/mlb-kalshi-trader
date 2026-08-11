"""Evaluate a fixed two-stage causal model for reversion versus timeout value."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

from catboost import CatBoostClassifier, CatBoostRegressor, Pool
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.backtest import (  # noqa: E402
    AWAY_TRADE_COLUMNS, STATE_COLUMNS, TRADE_COLUMNS,
)
from scripts.train_reversion_model import (  # noqa: E402
    CATEGORICAL_FEATURES, CONFIG_PATH, DATA, MODEL_FEATURES,
    build_opportunities,
)
from trade_tape_strategy.core import TradeTapeConfig  # noqa: E402


FIT_END = pd.Timestamp("2026-06-28").date()
FORWARD_START = pd.Timestamp("2026-07-24").date()
RESULT = PROJECT / "artifacts/two_stage_value_summary.json"
MICROSTRUCTURE_FEATURES = (
    "venue_trade_count_2s", "venue_volume_2s",
    "venue_flow_imbalance_2s", "venue_price_volatility_2s",
)
RESEARCH_FEATURES = (*MODEL_FEATURES, *MICROSTRUCTURE_FEATURES)


def add_microstructure(
    rows: pd.DataFrame, home: pd.DataFrame, away: pd.DataFrame,
) -> pd.DataFrame:
    result = rows.copy()
    for name in MICROSTRUCTURE_FEATURES:
        result[name] = 0.0
    venues = {"yes": home, "no": away}
    for side, side_rows in result.groupby("side", sort=False):
        tape_source = venues[str(side)]
        tapes = {
            int(game): tape.sort_values(["created_time", "trade_id"])
            for game, tape in tape_source[
                tape_source.game_pk.isin(side_rows.game_pk.unique())
            ].groupby("game_pk", sort=False)
        }
        for game_pk, game_rows in side_rows.groupby("game_pk", sort=False):
            tape = tapes.get(int(game_pk))
            if tape is None or tape.empty:
                continue
            times = pd.to_datetime(tape.created_time, utc=True).array.as_unit("ns").asi8
            prices = tape.yes_price_dollars.to_numpy(float)
            sizes = tape.count_fp.to_numpy(float)
            takers = tape.taker_outcome_side.astype(str).to_numpy()
            for index, row in game_rows.iterrows():
                end = pd.Timestamp(row.entry_time).value
                start = end - 2_000_000_000
                left = int(np.searchsorted(times, start, side="left"))
                right = int(np.searchsorted(times, end, side="left"))
                if right <= left:
                    continue
                window_sizes = sizes[left:right]
                volume = float(window_sizes.sum())
                signs = np.where(takers[left:right] == "yes", 1.0, -1.0)
                result.loc[index, "venue_trade_count_2s"] = right - left
                result.loc[index, "venue_volume_2s"] = volume
                result.loc[index, "venue_flow_imbalance_2s"] = (
                    float((window_sizes * signs).sum() / volume) if volume else 0.0
                )
                result.loc[index, "venue_price_volatility_2s"] = float(
                    np.std(prices[left:right])
                )
    return result


def pool(frame: pd.DataFrame, label) -> Pool:
    return Pool(
        frame.loc[:, RESEARCH_FEATURES], label=label,
        cat_features=list(CATEGORICAL_FEATURES),
    )


def apply_live_cooldown(frame: pd.DataFrame, seconds: float) -> pd.DataFrame:
    kept = []
    for _, game in frame.sort_values("entry_time").groupby("game_pk", sort=False):
        last = None
        for index, row in game.iterrows():
            when = pd.Timestamp(row.entry_time)
            if last is None or (when - last).total_seconds() >= seconds:
                kept.append(index)
                last = when
    return frame.loc[kept].sort_values(["game_date", "game_pk", "entry_time"])


def apply_priority_overlay(
    frame: pd.DataFrame, baseline_mask: pd.Series, seconds: float,
) -> pd.DataFrame:
    """Let baseline entries bypass the subordinate sleeve's cooldown."""
    kept = []
    for _, game in frame.sort_values("entry_time").groupby("game_pk", sort=False):
        last_baseline = None
        last_overlay = None
        for index, row in game.iterrows():
            when = pd.Timestamp(row.entry_time)
            if bool(baseline_mask.loc[index]):
                if (
                    last_baseline is None
                    or (when - last_baseline).total_seconds() >= seconds
                ):
                    kept.append(index)
                    last_baseline = when
                continue
            last_any = max(
                value for value in (last_baseline, last_overlay)
                if value is not None
            ) if any(value is not None for value in (last_baseline, last_overlay)) else None
            if last_any is None or (when - last_any).total_seconds() >= seconds:
                kept.append(index)
                last_overlay = when
    return frame.loc[kept].sort_values(["game_date", "game_pk", "entry_time"])


def metrics(frame: pd.DataFrame, scheduled_games: int) -> dict:
    capital = float((frame.contracts * frame.entry_price + frame.fees).sum())
    games = frame.groupby("game_pk").pnl.sum()
    return {
        "trades": int(len(frame)),
        "trades_per_game": len(frame) / scheduled_games,
        "pnl": float(frame.pnl.sum()),
        "capital": capital,
        "roi": float(frame.pnl.sum() / capital) if capital else 0.0,
        "pnl_without_top_four_games": float(
            frame.pnl.sum() - games.nlargest(min(4, len(games))).sum()
        ) if len(games) else 0.0,
    }


def main() -> None:
    config = TradeTapeConfig(**json.loads(CONFIG_PATH.read_text()))
    home = pd.read_parquet(DATA / "home_market_trades.parquet", columns=TRADE_COLUMNS)
    away = pd.read_parquet(DATA / "away_market_trades.parquet", columns=AWAY_TRADE_COLUMNS)
    states = pd.read_parquet(DATA / "state_updates.parquet", columns=STATE_COLUMNS)
    for frame in (home, away, states):
        frame["game_date"] = pd.to_datetime(frame.game_date).dt.date
    rows = build_opportunities(home, away, states, config)
    rows = add_microstructure(rows, home, away)
    rows["reverted"] = rows.exit_reason.eq("reversion").astype(int)
    fit = rows[rows.game_date < FIT_END].copy()
    development = rows[
        (rows.game_date >= FIT_END) & (rows.game_date < FORWARD_START)
    ].copy()
    forward = rows[rows.game_date >= FORWARD_START].copy()

    classifier = CatBoostClassifier(
        iterations=300, depth=5, learning_rate=.03, l2_leaf_reg=30,
        loss_function="Logloss", random_seed=2026,
        allow_writing_files=False, verbose=False,
    )
    success = CatBoostRegressor(
        iterations=250, depth=4, learning_rate=.03, l2_leaf_reg=30,
        loss_function="Huber:delta=0.05", random_seed=2027,
        allow_writing_files=False, verbose=False,
    )
    failure = CatBoostRegressor(
        iterations=250, depth=4, learning_rate=.03, l2_leaf_reg=30,
        loss_function="Huber:delta=0.05", random_seed=2028,
        allow_writing_files=False, verbose=False,
    )
    classifier.fit(pool(fit, fit.reverted))
    success_fit = fit[fit.reverted.eq(1)]
    failure_fit = fit[fit.reverted.eq(0)]
    success.fit(pool(success_fit, success_fit.pnl_per_contract))
    failure.fit(pool(failure_fit, failure_fit.pnl_per_contract))
    downside = CatBoostRegressor(
        iterations=300, depth=5, learning_rate=.03, l2_leaf_reg=30,
        loss_function="Quantile:alpha=0.25", random_seed=2029,
        allow_writing_files=False, verbose=False,
    )
    downside.fit(pool(fit, fit.pnl_per_contract))

    selected = {}
    downside_selected = {}
    overlay_selected = {}
    priority_overlay_selected = {}
    scheduled = {}
    game_dates = home[["game_pk", "game_date"]].drop_duplicates()
    for name, frame, start, end in (
        ("development", development, FIT_END, FORWARD_START),
        ("forward", forward, FORWARD_START, None),
    ):
        probability = classifier.predict_proba(
            frame.loc[:, RESEARCH_FEATURES]
        )[:, 1]
        expected = (
            probability * success.predict(frame.loc[:, RESEARCH_FEATURES])
            + (1.0 - probability) * failure.predict(frame.loc[:, RESEARCH_FEATURES])
        )
        chosen = frame[expected > 0.0].copy()
        chosen["two_stage_expected_pnl_per_contract"] = expected[expected > 0.0]
        chosen = apply_live_cooldown(
            chosen, config.minimum_seconds_between_entries
        )
        downside_prediction = downside.predict(frame.loc[:, RESEARCH_FEATURES])
        downside_chosen = frame[downside_prediction > 0.0].copy()
        downside_chosen["downside_pnl_per_contract"] = downside_prediction[
            downside_prediction > 0.0
        ]
        downside_chosen = apply_live_cooldown(
            downside_chosen, config.minimum_seconds_between_entries
        )
        overlay_mask = (
            frame.entry_net_edge.ge(config.minimum_edge)
            | (downside_prediction > 0.0)
        )
        overlay_chosen = frame[overlay_mask].copy()
        overlay_chosen["downside_pnl_per_contract"] = downside_prediction[
            overlay_mask
        ]
        overlay_chosen = apply_live_cooldown(
            overlay_chosen, config.minimum_seconds_between_entries
        )
        priority_chosen = frame[overlay_mask].copy()
        priority_chosen["downside_pnl_per_contract"] = downside_prediction[
            overlay_mask
        ]
        priority_chosen = apply_priority_overlay(
            priority_chosen,
            priority_chosen.entry_net_edge.ge(config.minimum_edge),
            config.minimum_seconds_between_entries,
        )
        game_mask = game_dates.game_date >= start
        if end is not None:
            game_mask &= game_dates.game_date < end
        scheduled[name] = int(game_dates.loc[game_mask, "game_pk"].nunique())
        selected[name] = chosen
        downside_selected[name] = downside_chosen
        overlay_selected[name] = overlay_chosen
        priority_overlay_selected[name] = priority_chosen

    combined = pd.concat(selected.values(), ignore_index=True)
    downside_combined = pd.concat(downside_selected.values(), ignore_index=True)
    overlay_combined = pd.concat(overlay_selected.values(), ignore_index=True)
    priority_overlay_combined = pd.concat(
        priority_overlay_selected.values(), ignore_index=True
    )
    total_games = sum(scheduled.values())
    summary = {
        "method": "fixed two-stage causal EV rule; no threshold search",
        "fit_end_exclusive": str(FIT_END),
        "forward_start": str(FORWARD_START),
        "fit_rows": len(fit),
        "fit_reversions": int(fit.reverted.sum()),
        "development": metrics(selected["development"], scheduled["development"]),
        "forward": metrics(selected["forward"], scheduled["forward"]),
        "combined": metrics(combined, total_games),
        "downside_rule": {
            "method": "predicted 25th-percentile PnL per contract > 0",
            "development": metrics(
                downside_selected["development"], scheduled["development"]
            ),
            "forward": metrics(
                downside_selected["forward"], scheduled["forward"]
            ),
            "combined": metrics(downside_combined, total_games),
        },
        "baseline_plus_downside_overlay": {
            "method": "5% baseline OR positive predicted 25th-percentile PnL",
            "development": metrics(
                overlay_selected["development"], scheduled["development"]
            ),
            "forward": metrics(
                overlay_selected["forward"], scheduled["forward"]
            ),
            "combined": metrics(overlay_combined, total_games),
        },
        "priority_overlay": {
            "method": "baseline-priority sleeve plus downside-positive additions",
            "development": metrics(
                priority_overlay_selected["development"], scheduled["development"]
            ),
            "forward": metrics(
                priority_overlay_selected["forward"], scheduled["forward"]
            ),
            "combined": metrics(priority_overlay_combined, total_games),
        },
    }
    evaluated = summary["priority_overlay"]
    gates = {
        "trades_per_game_above_baseline": evaluated["combined"]["trades_per_game"] > .10,
        "pnl_above_baseline": evaluated["combined"]["pnl"] > 13.4734,
        "roi_at_least_five_percent": evaluated["combined"]["roi"] >= .05,
        "both_periods_positive": (
            evaluated["development"]["pnl"] > 0
            and evaluated["forward"]["pnl"] > 0
        ),
        "concentration_positive": (
            evaluated["combined"]["pnl_without_top_four_games"] > 0
        ),
    }
    summary["gates"] = gates
    summary["deployable"] = all(gates.values())
    RESULT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
