"""All-signal residual paths for event-agnostic alpha/execution learning."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlb_kalshi.hybrid import anchored_event_target
from mlb_kalshi.state_overshoot import (
    OvershootConfig, _entry_fill_compatible, _exit_fill_compatible,
    _logit, signed_logit_residual,
)
from mlb_kalshi.state_reversion import add_before_and_delta_state
from mlb_kalshi.strategy import CONFIG, taker_fee


PATH_HORIZONS = (2, 5, 10, 30, 60)

RESIDUAL_PATH_FEATURES = (
    "signed_logit_residual", "absolute_logit_residual",
    "fair_logit_move", "market_logit_move", "signal_home_price",
    "signal_contract_price", "anchor_age_seconds", "signal_latency_seconds",
    "inning_after", "inning_topbot_after", "outs_when_up_after",
    "score_diff_after", "balls_after", "strikes_after",
    "runner_on_first_after", "runner_on_second_after", "runner_on_third_after",
    "delta_inning", "delta_outs", "delta_score_diff", "delta_balls",
    "delta_strikes", "delta_runner_on_first", "delta_runner_on_second",
    "delta_runner_on_third", "pre_trade_count_2s", "pre_volume_2s",
    "pre_flow_imbalance_2s", "pre_price_volatility_2s", "entry_side_yes",
)


def residual_path_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(RESIDUAL_PATH_FEATURES) - set(frame)
    if missing:
        raise ValueError(f"Missing residual-path features: {sorted(missing)}")
    result = frame.loc[:, RESIDUAL_PATH_FEATURES].apply(pd.to_numeric, errors="coerce")
    if result.isna().any().any():
        bad = result.columns[result.isna().any()].tolist()
        raise ValueError(f"Residual-path features contain nulls: {bad}")
    return result.astype(float)


def evaluate_path_policy(
    frame: pd.DataFrame,
    fill_probability,
    predicted_conditional_pnl,
    horizon_seconds: int,
    minimum_fill_probability: float,
    minimum_expected_pnl: float,
) -> dict:
    """Chronological one-position policy evaluation using realized tape labels."""
    work = frame.copy()
    work["predicted_fill_probability"] = np.asarray(fill_probability, float)
    work["predicted_conditional_pnl"] = np.asarray(predicted_conditional_pnl, float)
    work["predicted_expected_pnl"] = (
        work.predicted_fill_probability * work.predicted_conditional_pnl
    )
    label = f"net_pnl_{horizon_seconds}s"
    records = []
    attempts = fills = 0
    for _, game in work.groupby("game_pk", sort=False):
        occupied_until = None
        for row in game.sort_values("signal_time").itertuples(index=False):
            signal_time = pd.Timestamp(row.signal_time)
            if occupied_until is not None and signal_time <= occupied_until:
                continue
            if (
                row.predicted_fill_probability < minimum_fill_probability
                or row.predicted_expected_pnl < minimum_expected_pnl
            ):
                continue
            attempts += 1
            actual_pnl = getattr(row, label)
            if not bool(row.maker_filled) or pd.isna(actual_pnl):
                continue
            fills += 1
            occupied_until = signal_time + pd.Timedelta(seconds=horizon_seconds)
            records.append({**row._asdict(), "realized_pnl": float(actual_pnl)})
    pnl = float(sum(row["realized_pnl"] for row in records))
    capital = float(CONFIG.bet_size * fills)
    return {
        "attempts": attempts, "trades": fills, "pnl": pnl,
        "capital": capital, "roi": pnl / capital if capital else 0.0,
        "records": records,
    }


def _pre_flow(times, prices, sizes, sides, index: int) -> dict:
    start = int(np.searchsorted(times, times[index] - 2_000_000_000, side="left"))
    selected = sizes[start:index]
    volume = float(selected.sum())
    signs = np.where(sides[start:index] == "yes", 1.0, -1.0)
    return {
        "pre_trade_count_2s": float(index - start),
        "pre_volume_2s": volume,
        "pre_flow_imbalance_2s": (
            float((selected * signs).sum() / volume) if volume else 0.0
        ),
        "pre_price_volatility_2s": (
            float(np.std(prices[start:index])) if index > start else 0.0
        ),
    }


def _held_price(side: str, yes: float, no: float) -> float:
    return float(yes if side == "yes" else no)


def build_residual_path_dataset(
    trades: pd.DataFrame,
    updates: pd.DataFrame,
    config: OvershootConfig | None = None,
) -> pd.DataFrame:
    """Build one row for every persistent strict overshoot, filled or not."""
    config = config or OvershootConfig(
        minimum_logit_residual=0.04, minimum_fair_logit_move=0.02,
        observation_latency_buffer_seconds=2.0,
    )
    prepared = add_before_and_delta_state(updates).dropna(subset=[
        "inning_before", "outs_when_up_before", "score_diff_before",
        "balls_before", "strikes_before", "runner_on_first_before",
        "runner_on_second_before", "runner_on_third_before",
    ])
    confirmation_ns = int(config.confirmation_seconds * 1e9)
    anchor_age_ns = int(config.maximum_pre_event_trade_age_seconds * 1e9)
    latency_ns = int(config.observation_latency_buffer_seconds * 1e9)
    entry_window_ns = int(config.maximum_entry_latency_seconds * 1e9)
    rows = []

    for game_pk, game_updates in prepared.groupby("game_pk", sort=False):
        tape = trades[trades.game_pk.eq(game_pk)].sort_values(
            ["created_time", "trade_id"]
        )
        if tape.empty:
            continue
        update_rows = list(game_updates.sort_values("pitch_end_time").itertuples(index=False))
        update_times = np.array([
            pd.Timestamp(row.pitch_end_time).value for row in update_rows
        ], dtype=np.int64)
        times = pd.to_datetime(tape.created_time, utc=True).array.as_unit("ns").asi8
        yes = tape.yes_price_dollars.to_numpy(float)
        no = tape.no_price_dollars.to_numpy(float)
        sizes = tape.count_fp.to_numpy(float)
        sides = tape.taker_outcome_side.astype(str).to_numpy()

        for update_pos, update in enumerate(update_rows):
            event_ns = int(update_times[update_pos])
            cutoff_ns = event_ns - latency_ns
            anchor_i = int(np.searchsorted(times, cutoff_ns, side="left") - 1)
            if anchor_i < 0 or cutoff_ns - times[anchor_i] > anchor_age_ns:
                continue
            fair_move = _logit(update.fair_after) - _logit(update.fair_before)
            if abs(fair_move) < config.minimum_fair_logit_move:
                continue
            anchor_market = float(yes[anchor_i])
            fair_target = float(anchored_event_target(
                anchor_market, update.fair_before, update.fair_after
            ))
            first = int(np.searchsorted(times, event_ns, side="right"))
            stop = int(np.searchsorted(times, event_ns + entry_window_ns, side="right"))
            watch_side = None
            watch_ns = None
            signal_i = None
            for i in range(first, stop):
                market_move = _logit(yes[i]) - _logit(anchor_market)
                residual = signed_logit_residual(yes[i], fair_target)
                genuine = (
                    np.sign(market_move) == np.sign(fair_move)
                    and abs(market_move) >= abs(fair_move) + config.minimum_logit_residual
                )
                side = "no" if market_move > 0 else "yes" if genuine else None
                if not genuine:
                    side = None
                if side is None:
                    watch_side, watch_ns = None, None
                elif side != watch_side:
                    watch_side, watch_ns = side, int(times[i])
                elif times[i] - int(watch_ns) >= confirmation_ns:
                    signal_i = i
                    break
            if signal_i is None:
                continue
            side = str(watch_side)
            signal_home = float(yes[signal_i])
            signal_contract = _held_price(side, yes[signal_i], no[signal_i])
            entry_residual = signed_logit_residual(signal_home, fair_target)
            directional_entry = abs(entry_residual)
            contracts = CONFIG.bet_size / signal_contract

            maker_fill_i = None
            fill_stop = int(np.searchsorted(
                times, times[signal_i] + entry_window_ns, side="right"
            ))
            for i in range(signal_i + 1, fill_stop):
                held = _held_price(side, yes[i], no[i])
                if (
                    _entry_fill_compatible(side, sides[i], "maker")
                    and held <= signal_contract and sizes[i] >= contracts
                ):
                    maker_fill_i = i
                    break

            def state_at(ns: int):
                pos = int(np.searchsorted(update_times, ns, side="right") - 1)
                return update_rows[max(update_pos, pos)]

            def path_at(ns: int, label: str, row: dict):
                i = int(np.searchsorted(times, ns, side="left"))
                if i >= len(times):
                    row[f"contraction_{label}"] = np.nan
                    return
                current = state_at(int(times[i]))
                dynamic_target = float(anchored_event_target(
                    fair_target, update.fair_after, current.fair_after
                ))
                current_residual = signed_logit_residual(yes[i], dynamic_target)
                directional = current_residual if side == "no" else -current_residual
                row[f"contraction_{label}"] = (
                    directional_entry - directional
                ) / directional_entry
                row[f"seconds_to_{label}"] = (times[i] - times[signal_i]) / 1e9

            row = {
                "dataset_version": 1,
                "game_pk": int(game_pk), "game_date": update.game_date,
                "trigger_at_bat": int(update.at_bat_number),
                "trigger_pitch": int(update.pitch_number),
                "trigger_time": pd.Timestamp(event_ns, tz="UTC"),
                "signal_time": pd.Timestamp(times[signal_i], tz="UTC"),
                "side": side, "entry_side_yes": float(side == "yes"),
                "anchor_market": anchor_market, "fair_before": float(update.fair_before),
                "fair_after": float(update.fair_after), "fair_target": fair_target,
                "fair_logit_move": fair_move,
                "market_logit_move": _logit(signal_home) - _logit(anchor_market),
                "signed_logit_residual": entry_residual,
                "absolute_logit_residual": directional_entry,
                "signal_home_price": signal_home,
                "signal_contract_price": signal_contract,
                "anchor_age_seconds": (cutoff_ns - times[anchor_i]) / 1e9,
                "signal_latency_seconds": (times[signal_i] - event_ns) / 1e9,
                "maker_filled": int(maker_fill_i is not None),
                "maker_fill_delay_seconds": (
                    (times[maker_fill_i] - times[signal_i]) / 1e9
                    if maker_fill_i is not None else np.nan
                ),
                "maker_fill_time": (
                    pd.Timestamp(times[maker_fill_i], tz="UTC")
                    if maker_fill_i is not None else pd.NaT
                ),
                **{name: float(getattr(update, name)) for name in [
                    "inning_after", "inning_topbot_after", "outs_when_up_after",
                    "score_diff_after", "balls_after", "strikes_after",
                    "runner_on_first_after", "runner_on_second_after",
                    "runner_on_third_after", "delta_inning", "delta_outs",
                    "delta_score_diff", "delta_balls", "delta_strikes",
                    "delta_runner_on_first", "delta_runner_on_second",
                    "delta_runner_on_third",
                ]},
                **_pre_flow(times, yes, sizes, sides, signal_i),
            }
            for seconds in PATH_HORIZONS:
                path_at(times[signal_i] + int(seconds * 1e9), f"{seconds}s", row)
            if update_pos + 1 < len(update_rows):
                path_at(update_times[update_pos + 1], "next_pitch", row)
            else:
                row["contraction_next_pitch"] = np.nan
            next_pa = next((candidate for candidate in range(update_pos + 1, len(update_rows))
                            if pd.notna(getattr(update_rows[candidate], "completed_event", None))), None)
            if next_pa is not None:
                path_at(update_times[next_pa], "next_pa", row)
            else:
                row["contraction_next_pa"] = np.nan

            if maker_fill_i is not None:
                fill_price = signal_contract
                for seconds in PATH_HORIZONS:
                    target_ns = times[maker_fill_i] + int(seconds * 1e9)
                    start_i = int(np.searchsorted(times, target_ns, side="left"))
                    exit_stop = int(np.searchsorted(
                        times, target_ns + 10_000_000_000, side="right"
                    ))
                    exit_i = next((i for i in range(start_i, exit_stop)
                                   if _exit_fill_compatible(side, sides[i], "taker")
                                   and sizes[i] >= contracts), None)
                    if exit_i is None:
                        row[f"net_pnl_{seconds}s"] = np.nan
                    else:
                        exit_price = _held_price(side, yes[exit_i], no[exit_i])
                        row[f"net_pnl_{seconds}s"] = (
                            contracts * (exit_price - fill_price)
                            - taker_fee(contracts, exit_price)
                        )
            rows.append(row)
    return pd.DataFrame(rows)
