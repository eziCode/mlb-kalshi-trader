"""Causal partial-scale-out and runner simulation for empirical reversions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mlb_kalshi.strategy import taker_fee


@dataclass(frozen=True)
class RunnerPolicyConfig:
    enabled: bool = False
    minimum_probability_residual: float = 0.05
    minimum_reversion_probability: float = 0.80
    trailing_giveback_fraction: float = 0.50
    trailing_activation_multiple: float = 0.25
    second_target_multiple: float = 2.0
    adverse_state_probability_move: float = 0.05
    minimum_runner_contracts: float = 0.10


@dataclass
class RunnerResult:
    trades: int = 0
    scaled_reversions: int = 0
    full_reversion_exits: int = 0
    runner_target_exits: int = 0
    runner_trailing_exits: int = 0
    runner_state_exits: int = 0
    runner_settlements: int = 0
    full_settlements: int = 0
    fees: float = 0.0
    capital: float = 0.0
    pnl: float = 0.0
    accepted_ids: list[int] = field(default_factory=list)

    @property
    def roi(self) -> float:
        return self.pnl / self.capital if self.capital else 0.0


def recovery_contracts(
    capital_to_recover: float,
    price: float,
    maximum_contracts: float,
) -> float:
    """Smallest fractional sale whose net proceeds recover the capital."""
    if capital_to_recover <= 0 or price <= 0 or maximum_contracts <= 0:
        return 0.0
    full_proceeds = (
        maximum_contracts * price - taker_fee(maximum_contracts, price)
    )
    if full_proceeds < capital_to_recover:
        return maximum_contracts
    low, high = 0.0, maximum_contracts
    for _ in range(60):
        middle = (low + high) / 2.0
        proceeds = middle * price - taker_fee(middle, price)
        if proceeds >= capital_to_recover:
            high = middle
        else:
            low = middle
    return high


def _trade_arrays(trades: pd.DataFrame) -> dict[int, tuple]:
    result = {}
    for game_pk, game in trades.groupby("game_pk", sort=False):
        game = game.sort_values(["created_time", "trade_id"])
        result[int(game_pk)] = (
            pd.to_datetime(game["created_time"], utc=True)
            .array.as_unit("ns").asi8,
            game["yes_price_dollars"].to_numpy(dtype=float),
            game["no_price_dollars"].to_numpy(dtype=float),
            game["count_fp"].to_numpy(dtype=float),
            game["taker_outcome_side"].astype(str).to_numpy(),
        )
    return result


def _update_arrays(updates: pd.DataFrame) -> dict[int, tuple]:
    result = {}
    for game_pk, game in updates.groupby("game_pk", sort=False):
        game = game.sort_values("pitch_end_time")
        result[int(game_pk)] = (
            pd.to_datetime(game["pitch_end_time"], utc=True)
            .array.as_unit("ns").asi8,
            game["fair_after"].to_numpy(dtype=float),
        )
    return result


def _settlement_won(side: str, home_win: int) -> bool:
    return (side == "yes" and home_win == 1) or (
        side == "no" and home_win == 0
    )


def _simulate_candidate(
    candidate,
    trade_arrays: tuple,
    update_arrays: tuple | None,
    config: RunnerPolicyConfig,
) -> dict:
    contracts = float(candidate.contracts)
    entry_price = float(candidate.entry_price)
    entry_fee = float(candidate.entry_fee)
    initial_capital = contracts * entry_price + entry_fee
    side = str(candidate.entry_side)
    home_win = int(candidate.home_win)
    base = {
        "candidate_id": int(candidate.candidate_id),
        "game_pk": int(candidate.game_pk),
        "game_date": candidate.game_date,
        "entry_time": pd.Timestamp(candidate.entry_time),
        "entry_side": side,
        "entry_price": entry_price,
        "contracts": contracts,
        "capital": initial_capital,
        "recovery_contracts": 0.0,
        "runner_contracts": 0.0,
        "recovery_price": np.nan,
        "runner_exit_price": np.nan,
        "runner_exit_time": pd.NaT,
    }
    if not bool(candidate.profitable_reversion):
        proceeds = contracts if _settlement_won(side, home_win) else 0.0
        return {
            **base,
            "exit_reason": "full_settlement",
            "occupied_until": pd.NaT,
            "fees": entry_fee,
            "pnl": proceeds - initial_capital,
        }

    recovery_time = pd.Timestamp(candidate.reversion_exit_time)
    recovery_price = float(candidate.reversion_exit_price)
    sold = recovery_contracts(initial_capital, recovery_price, contracts)
    recovery_fee = taker_fee(sold, recovery_price)
    recovery_proceeds = sold * recovery_price - recovery_fee
    runner_contracts = max(0.0, contracts - sold)
    base.update({
        "recovery_contracts": sold,
        "runner_contracts": runner_contracts,
        "recovery_price": recovery_price,
    })
    if runner_contracts < config.minimum_runner_contracts:
        full_fee = taker_fee(contracts, recovery_price)
        pnl = contracts * recovery_price - full_fee - initial_capital
        return {
            **base,
            "recovery_contracts": contracts,
            "runner_contracts": 0.0,
            "exit_reason": "full_reversion",
            "occupied_until": recovery_time,
            "fees": entry_fee + full_fee,
            "pnl": pnl,
        }

    times, yes_prices, no_prices, sizes, taker_sides = trade_arrays
    recovery_ns = recovery_time.value
    start = int(np.searchsorted(times, recovery_ns, side="right"))
    game_end_ns = pd.Timestamp(candidate.game_end_time).value
    stop = int(np.searchsorted(times, game_end_ns, side="right"))
    exit_taker_side = "no" if side == "yes" else "yes"
    original_move = max(recovery_price - entry_price, 1e-9)
    activation_price = (
        recovery_price + config.trailing_activation_multiple * original_move
    )
    second_target = min(
        1.0,
        entry_price + config.second_target_multiple * original_move,
    )
    high_water = recovery_price
    pending_ns = None
    pending_reason = None

    update_times = np.array([], dtype=np.int64)
    held_fairs = np.array([], dtype=float)
    update_index = 0
    prior_held_fair = None
    if update_arrays is not None:
        update_times, home_fairs = update_arrays
        held_fairs = home_fairs if side == "yes" else 1.0 - home_fairs
        update_index = int(np.searchsorted(
            update_times, recovery_ns, side="right"
        ))
        if update_index > 0:
            prior_held_fair = float(held_fairs[update_index - 1])

    for index in range(start, stop):
        trade_ns = int(times[index])
        while update_index < len(update_times) and update_times[update_index] <= trade_ns:
            current_fair = float(held_fairs[update_index])
            update_ns = int(update_times[update_index])
            if (
                pending_ns is None
                and prior_held_fair is not None
                and prior_held_fair - current_fair
                >= config.adverse_state_probability_move
            ):
                pending_ns = update_ns
                pending_reason = "runner_state"
            prior_held_fair = current_fair
            update_index += 1

        position_price = float(
            yes_prices[index] if side == "yes" else no_prices[index]
        )
        if (
            pending_ns is not None
            and trade_ns > pending_ns
            and taker_sides[index] == exit_taker_side
            and sizes[index] >= runner_contracts
        ):
            runner_fee = taker_fee(runner_contracts, position_price)
            runner_proceeds = runner_contracts * position_price - runner_fee
            return {
                **base,
                "exit_reason": pending_reason,
                "occupied_until": pd.Timestamp(trade_ns, tz="UTC"),
                "runner_exit_time": pd.Timestamp(trade_ns, tz="UTC"),
                "runner_exit_price": position_price,
                "fees": entry_fee + recovery_fee + runner_fee,
                "pnl": recovery_proceeds + runner_proceeds - initial_capital,
            }

        high_water = max(high_water, position_price)
        trailing_floor = recovery_price + (
            1.0 - config.trailing_giveback_fraction
        ) * (high_water - recovery_price)
        if pending_ns is None and position_price >= second_target:
            pending_ns = trade_ns
            pending_reason = "runner_target"
        elif (
            pending_ns is None
            and high_water >= activation_price
            and position_price <= trailing_floor
        ):
            pending_ns = trade_ns
            pending_reason = "runner_trailing"

    runner_proceeds = runner_contracts if _settlement_won(side, home_win) else 0.0
    return {
        **base,
        "exit_reason": "runner_settlement",
        "occupied_until": pd.NaT,
        "fees": entry_fee + recovery_fee,
        "pnl": recovery_proceeds + runner_proceeds - initial_capital,
    }


def build_runner_outcomes(
    candidates: pd.DataFrame,
    trades: pd.DataFrame,
    updates: pd.DataFrame,
    config: RunnerPolicyConfig,
    prepared_data: tuple[dict[int, tuple], dict[int, tuple]] | None = None,
) -> pd.DataFrame:
    """Precompute each candidate's outcome under one runner exit policy."""
    required_candidates = {
        "candidate_id", "game_pk", "game_date", "game_end_time", "home_win",
        "entry_time", "entry_side", "entry_price", "entry_fee", "contracts",
        "profitable_reversion", "reversion_exit_time", "reversion_exit_price",
    }
    if missing := required_candidates - set(candidates.columns):
        raise ValueError(f"Runner candidates are missing: {sorted(missing)}")
    if prepared_data is None:
        trade_map = _trade_arrays(trades)
        update_map = _update_arrays(updates)
    else:
        trade_map, update_map = prepared_data
    rows = []
    for candidate in candidates.itertuples(index=False):
        game_pk = int(candidate.game_pk)
        if game_pk not in trade_map:
            continue
        rows.append(_simulate_candidate(
            candidate,
            trade_map[game_pk],
            update_map.get(game_pk),
            config,
        ))
    return pd.DataFrame(rows)


def prepare_runner_data(
    trades: pd.DataFrame,
    updates: pd.DataFrame,
) -> tuple[dict[int, tuple], dict[int, tuple]]:
    """Prepare immutable per-game arrays for tuning multiple exit policies."""
    return _trade_arrays(trades), _update_arrays(updates)


def evaluate_runner_strategy(
    candidates: pd.DataFrame,
    outcomes: pd.DataFrame,
    config: RunnerPolicyConfig,
) -> RunnerResult:
    """Apply entry filters and enforce one open position per game."""
    merged = candidates.merge(
        outcomes,
        on=["candidate_id", "game_pk", "game_date", "entry_time", "entry_side",
            "entry_price", "contracts"],
        how="inner",
        suffixes=("", "_outcome"),
    )
    eligible = merged[
        (merged["probability_residual"] >= config.minimum_probability_residual)
        & (
            merged["predicted_reversion_probability"]
            >= config.minimum_reversion_probability
        )
    ].sort_values(["game_pk", "entry_time", "candidate_id"])
    result = RunnerResult()
    for _, game in eligible.groupby("game_pk", sort=False):
        occupied_until = None
        settled = False
        for row in game.itertuples(index=False):
            if settled:
                break
            entry_time = pd.Timestamp(row.entry_time)
            if occupied_until is not None and entry_time <= occupied_until:
                continue
            result.trades += 1
            result.capital += float(row.capital)
            result.fees += float(row.fees)
            result.pnl += float(row.pnl)
            result.accepted_ids.append(int(row.candidate_id))
            reason = str(row.exit_reason)
            result.scaled_reversions += int(float(row.runner_contracts) > 0)
            result.full_reversion_exits += int(reason == "full_reversion")
            result.runner_target_exits += int(reason == "runner_target")
            result.runner_trailing_exits += int(reason == "runner_trailing")
            result.runner_state_exits += int(reason == "runner_state")
            result.runner_settlements += int(reason == "runner_settlement")
            result.full_settlements += int(reason == "full_settlement")
            if reason in {"runner_settlement", "full_settlement"}:
                settled = True
            else:
                occupied_until = pd.Timestamp(row.occupied_until)
    return result
