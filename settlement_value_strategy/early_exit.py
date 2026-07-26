"""Causal research-only exits for settlement-value backtest records."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from settlement_value_strategy.strategy import taker_fee


@dataclass(frozen=True)
class EarlyExitConfig:
    minimum_hold_seconds: float = 30.0
    minimum_exit_inning: int = 1
    maximum_fill_delay_seconds: float = 5.0
    stop_loss_points: float | None = 0.10
    exit_edge_threshold: float | None = 0.0
    require_opposite_taker: bool = True


def _prepared_tape(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    tape = frame.sort_values(["created_time", "trade_id"]).reset_index(drop=True)
    times = pd.to_datetime(
        tape.created_time, utc=True
    ).array.as_unit("ns").asi8
    return tape, times


def prepare_exit_tapes(
    home_trades: pd.DataFrame, away_trades: pd.DataFrame,
) -> tuple[dict[int, tuple[pd.DataFrame, np.ndarray]], ...]:
    return tuple({
        int(game): _prepared_tape(group)
        for game, group in source.groupby("game_pk", sort=False)
    } for source in (home_trades, away_trades))


def apply_early_exits(
    records: pd.DataFrame,
    scored_decisions: pd.DataFrame,
    home_trades: pd.DataFrame,
    away_trades: pd.DataFrame,
    config: EarlyExitConfig,
    prepared_tapes: tuple[dict[int, tuple[pd.DataFrame, np.ndarray]], ...]
    | None = None,
) -> pd.DataFrame:
    """Apply an exit overlay without changing the strategy's entry set.

    A state can trigger an exit only after it was observable. The simulated
    sale then requires a strictly later execution before the next state update
    (and within the configured fill window), enough printed size, and normally
    an opposite-side taker. Positions without such a fill still settle.
    """
    if records.empty:
        return records.copy()
    decisions = scored_decisions.copy()
    decisions["signal_time"] = pd.to_datetime(decisions.signal_time, utc=True)
    decisions["next_update_time"] = pd.to_datetime(
        decisions.next_update_time, utc=True
    )
    decisions_by_game = {
        int(game): group.sort_values("signal_time")
        for game, group in decisions.groupby("game_pk", sort=False)
    }
    tape_maps = prepared_tapes or prepare_exit_tapes(home_trades, away_trades)

    output = records.copy()
    output["exit_reason"] = output.get("exit_reason", "settlement")
    output["exit_time"] = output.get("exit_time", pd.NaT)
    output["exit_time"] = pd.to_datetime(output["exit_time"], utc=True)
    output["exit_price"] = output.get("exit_price", np.nan)
    output["exit_fee"] = output.get("exit_fee", 0.0)
    output["exit_model_probability"] = np.nan
    output["exit_observed_price"] = np.nan
    output["exit_model_edge"] = np.nan

    hold_ns = int(config.minimum_hold_seconds * 1e9)
    fill_ns = int(config.maximum_fill_delay_seconds * 1e9)
    for index, entry in output.iterrows():
        # Preserve the simulator's existing profitable-reversal exits.
        if pd.notna(entry.get("exit_time")):
            continue
        game_pk = int(entry.game_pk)
        game_decisions = decisions_by_game.get(game_pk)
        tape_map = tape_maps[0 if entry.execution_contract == "home_yes" else 1]
        prepared = tape_map.get(game_pk)
        if game_decisions is None or prepared is None:
            continue
        tape, times = prepared
        prices = tape.yes_price_dollars.to_numpy(float)
        sizes = tape.count_fp.to_numpy(float)
        takers = tape.taker_outcome_side.astype(str).str.lower().to_numpy()
        fill_time_ns = pd.Timestamp(entry.fill_time).value
        earliest_exit_signal = pd.Timestamp(
            fill_time_ns + hold_ns, tz="UTC"
        )
        future = game_decisions[
            game_decisions.signal_time >= earliest_exit_signal
        ]
        for decision in future.itertuples(index=False):
            if (
                int(getattr(decision, "inning_after", 1))
                < config.minimum_exit_inning
            ):
                continue
            signal_ns = pd.Timestamp(decision.signal_time).value
            observed_index = int(np.searchsorted(times, signal_ns, side="right") - 1)
            if observed_index < 0:
                continue
            observed_price = float(prices[observed_index])
            held_probability = (
                float(decision.fair_probability)
                if entry.execution_contract == "home_yes"
                else 1.0 - float(decision.fair_probability)
            )
            edge = held_probability - observed_price
            thesis_broken = (
                config.exit_edge_threshold is not None
                and edge <= config.exit_edge_threshold
            )
            stop_hit = (
                config.stop_loss_points is not None
                and observed_price
                <= float(entry.fill_price) - config.stop_loss_points
            )
            if not thesis_broken and not stop_hit:
                continue
            deadline = signal_ns + fill_ns
            if pd.notna(decision.next_update_time):
                deadline = min(
                    deadline, pd.Timestamp(decision.next_update_time).value
                )
            start = int(np.searchsorted(times, signal_ns, side="right"))
            stop = int(np.searchsorted(times, deadline, side="left"))
            for trade_index in range(start, stop):
                if config.require_opposite_taker and takers[trade_index] != "no":
                    continue
                if sizes[trade_index] + 1e-9 < float(entry.contracts):
                    continue
                exit_price = float(prices[trade_index])
                exit_fee = taker_fee(float(entry.contracts), exit_price)
                realized = (
                    float(entry.contracts) * (exit_price - float(entry.fill_price))
                    - float(entry.entry_fee) - exit_fee
                )
                reason = (
                    "stop_loss_and_thesis_break"
                    if stop_hit and thesis_broken
                    else "stop_loss" if stop_hit else "thesis_break"
                )
                output.at[index, "pnl"] = realized
                output.at[index, "exit_reason"] = reason
                output.at[index, "exit_time"] = pd.Timestamp(
                    times[trade_index], tz="UTC"
                )
                output.at[index, "exit_price"] = exit_price
                output.at[index, "exit_fee"] = exit_fee
                output.at[index, "exit_model_probability"] = held_probability
                output.at[index, "exit_observed_price"] = observed_price
                output.at[index, "exit_model_edge"] = edge
                break
            if pd.notna(output.at[index, "exit_time"]):
                break
    return output
