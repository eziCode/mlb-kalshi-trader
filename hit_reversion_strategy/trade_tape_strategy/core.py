"""Sub-second trade-tape simulation for the hybrid hit residual strategy."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .hybrid import anchored_event_target
from .strategy import CONFIG, estimated_round_trip_fee_per_contract, taker_fee


@dataclass(frozen=True)
class TradeTapeConfig:
    enabled: bool = True
    minimum_edge: float = 0.04
    confirmation_seconds: float = 2.0
    maximum_pre_event_trade_age_seconds: float = 5.0
    maximum_event_to_entry_seconds: float = 10.0
    invalidate_on_next_pitch: bool = True
    minimum_fair_move: float = 0.005
    momentum_exit_enabled: bool = False
    momentum_window_seconds: float = 2.0
    minimum_favorable_velocity: float = 0.01
    momentum_trailing_giveback: float = 0.01
    momentum_max_hold_seconds: float = 2.0
    minimum_momentum_trades: int = 3
    minimum_seconds_between_entries: float = 180.0
    allowed_event_types: tuple[str, ...] = ("single", "double", "triple")
    maximum_hold_seconds: float = 0.0
    exit_target_mode: str = "dynamic"
    latch_reversion_exit: bool = False
    minimum_reversion_move: float = 0.0
    side_filter: str = "both"
    position_sizing: str = "fixed_payout"
    require_compatible_taker: bool = True
    minimum_edges_by_segment: dict[str, float] = field(default_factory=dict)
    confirmation_seconds_by_segment: dict[str, float] = field(
        default_factory=dict
    )
    minimum_reversion_moves_by_segment: dict[str, float] = field(
        default_factory=dict
    )


@dataclass
class Candidate:
    anchor_target: float
    anchor_fair: float
    event_time_ns: int
    event_type: str
    trigger_at_bat: int
    trigger_pitch: int
    material_state: tuple
    watch_side: str | None = None
    watch_started_ns: int | None = None


@dataclass
class PendingEntry:
    candidate: Candidate
    side: str
    created_ns: int
    confirmation_price: float


@dataclass
class TapePosition:
    side: str
    contracts: float
    entry_price: float
    entry_fee: float
    entry_ns: int
    anchor_target: float
    anchor_fair: float
    event_type: str
    trigger_at_bat: int
    trigger_pitch: int
    trigger_event_time_ns: int
    pending_exit_ns: int | None = None
    pending_exit_reason: str | None = None
    held_price_history: list[tuple[int, float]] = field(default_factory=list)
    momentum_hold_started_ns: int | None = None
    momentum_high_water: float | None = None


@dataclass
class TapeTradeRecord:
    game_pk: int
    side: str
    event_type: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp | None
    exit_reason: str
    entry_price: float
    exit_price: float | None
    contracts: float
    anchor_target: float
    anchor_fair: float
    trigger_at_bat: int
    trigger_pitch: int
    trigger_event_time: pd.Timestamp
    pnl: float
    fees: float


@dataclass
class TradeTapeResult:
    observed_hits: int = 0
    fresh_hit_anchors: int = 0
    eligible_hit_updates: int = 0
    rejected_fair_updates: int = 0
    invalidated_candidates: int = 0
    expired_candidates: int = 0
    confirmed_signals: int = 0
    trades: int = 0
    yes_trades: int = 0
    no_trades: int = 0
    reversion_exits: int = 0
    momentum_exits: int = 0
    timeout_exits: int = 0
    settlements: int = 0
    fees: float = 0.0
    capital: float = 0.0
    pnl: float = 0.0
    records: list[TapeTradeRecord] = field(default_factory=list)

    @property
    def roi(self) -> float:
        return self.pnl / self.capital if self.capital else 0.0


def trade_signal(
    target: float,
    yes_price: float,
    minimum_edge: float,
) -> tuple[str | None, float]:
    """Return a side only when its edge remains after estimated taker fees.

    ``minimum_edge`` is a net threshold. The fee reserve covers both entry
    and an early taker exit, using the same fixed-dollar position sizing as
    the simulator. Trade-tape prices are complementary, so the executable
    NO price is one minus the observed YES price.
    """
    no_price = 1.0 - yes_price
    yes_edge = (
        target
        - yes_price
        - estimated_round_trip_fee_per_contract(yes_price)
    )
    no_edge = (
        yes_price
        - target
        - estimated_round_trip_fee_per_contract(no_price)
    )
    if yes_edge >= minimum_edge:
        return "yes", yes_edge
    if no_edge >= minimum_edge:
        return "no", no_edge
    return None, max(yes_edge, no_edge)


def segment_value(
    values: dict[str, float], event_type: str, side: str, fallback: float,
) -> float:
    return float(values.get(f"{event_type}:{side}", fallback))


def segmented_trade_signal(
    target: float, yes_price: float, event_type: str, config: TradeTapeConfig,
) -> tuple[str | None, float]:
    """Apply independently calibrated thresholds to each hit/side segment."""
    no_price = 1.0 - yes_price
    edges = {
        "yes": target - yes_price
        - estimated_round_trip_fee_per_contract(yes_price),
        "no": yes_price - target
        - estimated_round_trip_fee_per_contract(no_price),
    }
    eligible = []
    for side, edge in edges.items():
        if config.side_filter != "both" and side != config.side_filter:
            continue
        threshold = segment_value(
            config.minimum_edges_by_segment,
            event_type,
            side,
            config.minimum_edge,
        )
        if edge >= threshold:
            eligible.append((edge - threshold, edge, side))
    if not eligible:
        return None, max(edges.values())
    _, edge, side = max(eligible)
    return side, edge


def compatible_taker(side: str, taker_outcome_side: str) -> bool:
    return side == taker_outcome_side


def position_contracts(price: float, config: TradeTapeConfig) -> float:
    if config.position_sizing == "fixed_payout":
        return CONFIG.bet_size
    if config.position_sizing == "fixed_stake":
        return CONFIG.bet_size / price
    raise ValueError(f"Unknown position sizing: {config.position_sizing}")


def _dynamic_target(candidate_or_position, current_fair: float) -> float:
    return float(anchored_event_target(
        candidate_or_position.anchor_target,
        candidate_or_position.anchor_fair,
        current_fair,
    ))


def _position_exit_target(
    position: TapePosition, current_fair: float, config: TradeTapeConfig,
) -> float:
    if config.exit_target_mode == "frozen":
        return float(position.anchor_target)
    if config.exit_target_mode == "dynamic":
        return _dynamic_target(position, current_fair)
    raise ValueError(f"Unknown exit_target_mode: {config.exit_target_mode}")


def _ns_to_timestamp(value: int | None) -> pd.Timestamp | None:
    return None if value is None else pd.Timestamp(value, tz="UTC")


def _material_state(update) -> tuple:
    return (
        int(update.score_diff_after),
        int(update.outs_when_up_after),
        int(update.runner_on_first_after),
        int(update.runner_on_second_after),
        int(update.runner_on_third_after),
    )


def _price_velocity(
    history: list[tuple[int, float]],
    now_ns: int,
    window_ns: int,
    minimum_trades: int,
) -> float | None:
    """Causal least-squares held-price slope in probability points/second."""
    start_ns = now_ns - window_ns
    recent = [(when, price) for when, price in history if when >= start_ns]
    if len(recent) < minimum_trades:
        return None
    times = np.array([(when - recent[0][0]) / 1e9 for when, _ in recent])
    if times[-1] <= 0:
        return None
    prices = np.array([price for _, price in recent], dtype=float)
    return float(np.polyfit(times, prices, 1)[0])


def simulate_trade_tape(
    trades: pd.DataFrame,
    updates: pd.DataFrame,
    config: TradeTapeConfig,
) -> TradeTapeResult:
    required_trades = {
        "game_pk", "created_time", "yes_price_dollars", "no_price_dollars",
        "count_fp", "taker_outcome_side", "home_win",
    }
    required_updates = {
        "game_pk", "pitch_start_time", "pitch_end_time", "is_hit",
        "completed_event", "completed_event_batting_home", "fair_before",
        "fair_after", "at_bat_number", "pitch_number", "score_diff_after",
        "outs_when_up_after", "runner_on_first_after",
        "runner_on_second_after", "runner_on_third_after",
    }
    if missing := required_trades - set(trades.columns):
        raise ValueError(f"Trade tape is missing columns: {sorted(missing)}")
    if missing := required_updates - set(updates.columns):
        raise ValueError(f"State updates are missing columns: {sorted(missing)}")

    result = TradeTapeResult()
    updates_by_game = {
        int(game_pk): game.sort_values("pitch_end_time")
        for game_pk, game in updates.groupby("game_pk", sort=False)
    }
    momentum_window_ns = int(config.momentum_window_seconds * 1_000_000_000)
    momentum_max_hold_ns = int(
        config.momentum_max_hold_seconds * 1_000_000_000
    )
    maximum_anchor_age_ns = int(
        config.maximum_pre_event_trade_age_seconds * 1_000_000_000
    )
    maximum_entry_age_ns = int(
        config.maximum_event_to_entry_seconds * 1_000_000_000
    )
    maximum_hold_ns = int(config.maximum_hold_seconds * 1_000_000_000)
    allowed_events = frozenset(config.allowed_event_types)

    for game_pk, game_trades in trades.groupby("game_pk", sort=False):
        game_pk = int(game_pk)
        game_updates = updates_by_game.get(game_pk)
        if game_updates is None or game_updates.empty:
            continue
        game_trades = game_trades.sort_values(["created_time", "trade_id"])
        times = pd.to_datetime(
            game_trades["created_time"], utc=True
        ).array.as_unit("ns").asi8
        yes_prices = game_trades["yes_price_dollars"].to_numpy(dtype=float)
        no_prices = game_trades["no_price_dollars"].to_numpy(dtype=float)
        sizes = game_trades["count_fp"].to_numpy(dtype=float)
        taker_sides = game_trades["taker_outcome_side"].astype(str).to_numpy()
        home_win = int(game_trades["home_win"].iloc[-1])

        update_rows = list(game_updates.itertuples(index=False))
        update_index = 0
        current_fair = float(update_rows[0].fair_before)
        candidate: Candidate | None = None
        pending_entry: PendingEntry | None = None
        positions: list[TapePosition] = []
        last_entry_ns: int | None = None

        for trade_index, trade_ns in enumerate(times):
            while (
                update_index < len(update_rows)
                and pd.Timestamp(update_rows[update_index].pitch_end_time).value
                <= trade_ns
            ):
                update = update_rows[update_index]
                active_candidate = (
                    pending_entry.candidate
                    if pending_entry is not None
                    else candidate
                )
                if (
                    active_candidate is not None
                    and pd.Timestamp(update.pitch_end_time).value
                    > active_candidate.event_time_ns
                ):
                    completed_plate_appearance = pd.notna(update.completed_event)
                    material_change = (
                        _material_state(update) != active_candidate.material_state
                    )
                    later_pitch = config.invalidate_on_next_pitch
                    if later_pitch or completed_plate_appearance or material_change:
                        result.invalidated_candidates += 1
                        candidate = None
                        pending_entry = None
                current_fair = float(update.fair_after)
                if bool(update.is_hit) and str(update.completed_event) in allowed_events:
                    result.observed_hits += 1
                    if pd.notna(update.completed_event):
                        batting_home = bool(update.completed_event_batting_home)
                        signed_fair_move = (
                            float(update.fair_after) - float(update.fair_before)
                        ) * (1.0 if batting_home else -1.0)
                        if signed_fair_move < config.minimum_fair_move:
                            result.rejected_fair_updates += 1
                        else:
                            result.eligible_hit_updates += 1
                            pitch_start_ns = pd.Timestamp(
                                update.pitch_start_time
                            ).value
                            anchor_index = int(
                                np.searchsorted(
                                    times, pitch_start_ns, side="left"
                                ) - 1
                            )
                            if (
                                anchor_index >= 0
                                and pitch_start_ns - times[anchor_index]
                                <= maximum_anchor_age_ns
                            ):
                                result.fresh_hit_anchors += 1
                                target = float(anchored_event_target(
                                    yes_prices[anchor_index],
                                    float(update.fair_before),
                                    float(update.fair_after),
                                ))
                                candidate = Candidate(
                                    anchor_target=target,
                                    anchor_fair=float(update.fair_after),
                                    event_time_ns=pd.Timestamp(
                                        update.pitch_end_time
                                    ).value,
                                    event_type=str(update.completed_event),
                                    trigger_at_bat=int(update.at_bat_number),
                                    trigger_pitch=int(update.pitch_number),
                                    material_state=_material_state(update),
                                )
                                pending_entry = None
                update_index += 1

            yes_price = float(yes_prices[trade_index])
            no_price = float(no_prices[trade_index])
            size = float(sizes[trade_index])
            taker_side = taker_sides[trade_index]

            active_candidate = (
                pending_entry.candidate
                if pending_entry is not None
                else candidate
            )
            if (
                active_candidate is not None
                and trade_ns - active_candidate.event_time_ns
                > maximum_entry_age_ns
            ):
                result.expired_candidates += 1
                candidate = None
                pending_entry = None

            remaining_size = size
            closed_positions: list[TapePosition] = []
            for position in positions:
                target = _position_exit_target(position, current_fair, config)
                held_price = yes_price if position.side == "yes" else no_price
                reverted = (
                    position.side == "yes" and yes_price >= target
                ) or (
                    position.side == "no" and yes_price <= target
                )
                velocity = None
                if config.momentum_exit_enabled:
                    position.held_price_history.append((trade_ns, held_price))
                    position.held_price_history = [
                        point for point in position.held_price_history
                        if point[0] >= trade_ns - momentum_window_ns
                    ]
                    velocity = _price_velocity(
                        position.held_price_history, trade_ns,
                        momentum_window_ns, config.minimum_momentum_trades,
                    )
                exit_taker_side = "no" if position.side == "yes" else "yes"
                if position.pending_exit_ns is not None:
                    if (
                        position.pending_exit_reason == "reversion"
                        and not reverted
                        and not config.latch_reversion_exit
                    ):
                        position.pending_exit_ns = None
                        position.pending_exit_reason = None
                    elif (
                        trade_ns > position.pending_exit_ns
                        and (
                            not config.require_compatible_taker
                            or compatible_taker(exit_taker_side, taker_side)
                        )
                        and remaining_size >= position.contracts
                        and (
                            reverted
                            or config.latch_reversion_exit
                            or position.pending_exit_reason != "reversion"
                        )
                    ):
                        exit_price = held_price
                        exit_fee = taker_fee(position.contracts, exit_price)
                        pnl = (
                            position.contracts * exit_price - exit_fee
                            - position.contracts * position.entry_price
                            - position.entry_fee
                        )
                        result.pnl += pnl
                        result.fees += exit_fee
                        result.reversion_exits += int(
                            position.pending_exit_reason == "reversion"
                        )
                        result.timeout_exits += int(
                            position.pending_exit_reason == "timeout"
                        )
                        result.momentum_exits += int(
                            position.pending_exit_reason == "momentum_reversion"
                        )
                        result.records.append(TapeTradeRecord(
                            game_pk=game_pk, side=position.side,
                            event_type=position.event_type,
                            entry_time=_ns_to_timestamp(position.entry_ns),
                            exit_time=_ns_to_timestamp(trade_ns),
                            exit_reason=(position.pending_exit_reason or "reversion"),
                            entry_price=position.entry_price,
                            exit_price=exit_price, contracts=position.contracts,
                            anchor_target=position.anchor_target,
                            anchor_fair=position.anchor_fair,
                            trigger_at_bat=position.trigger_at_bat,
                            trigger_pitch=position.trigger_pitch,
                            trigger_event_time=_ns_to_timestamp(
                                position.trigger_event_time_ns
                            ),
                            pnl=pnl, fees=position.entry_fee + exit_fee,
                        ))
                        remaining_size -= position.contracts
                        closed_positions.append(position)
                        continue
                if position.pending_exit_ns is None:
                    if (
                        maximum_hold_ns > 0
                        and trade_ns - position.entry_ns >= maximum_hold_ns
                    ):
                        position.pending_exit_ns = trade_ns
                        position.pending_exit_reason = "timeout"
                    elif position.momentum_hold_started_ns is not None:
                        position.momentum_high_water = max(
                            float(position.momentum_high_water), held_price
                        )
                        momentum_stopped = (
                            (velocity is not None and velocity <= 0)
                            or held_price <= (
                                position.momentum_high_water
                                - config.momentum_trailing_giveback
                            )
                            or trade_ns - position.momentum_hold_started_ns
                            >= momentum_max_hold_ns
                        )
                        if momentum_stopped:
                            position.pending_exit_ns = trade_ns
                            position.pending_exit_reason = "momentum_reversion"
                    elif reverted:
                        strong_momentum = (
                            config.momentum_exit_enabled
                            and velocity is not None
                            and velocity >= config.minimum_favorable_velocity
                        )
                        if strong_momentum:
                            position.momentum_hold_started_ns = trade_ns
                            position.momentum_high_water = held_price
                        else:
                            position.pending_exit_ns = trade_ns
                            position.pending_exit_reason = "reversion"
            if closed_positions:
                positions = [p for p in positions if p not in closed_positions]

            if pending_entry is not None:
                target = _dynamic_target(pending_entry.candidate, current_fair)
                side, _ = segmented_trade_signal(
                    target, yes_price, pending_entry.candidate.event_type,
                    config,
                )
                if side != pending_entry.side:
                    candidate = pending_entry.candidate
                    candidate.watch_side = None
                    candidate.watch_started_ns = None
                    pending_entry = None
                elif (
                    trade_ns > pending_entry.created_ns
                    and (
                        not config.require_compatible_taker
                        or compatible_taker(side, taker_side)
                    )
                    and (
                        last_entry_ns is None
                        or trade_ns - last_entry_ns >= int(
                            config.minimum_seconds_between_entries * 1e9
                        )
                    )
                ):
                    reversion_move = (
                        yes_price - pending_entry.confirmation_price
                        if side == "yes"
                        else pending_entry.confirmation_price - yes_price
                    )
                    minimum_reversion_move = segment_value(
                        config.minimum_reversion_moves_by_segment,
                        pending_entry.candidate.event_type,
                        side,
                        config.minimum_reversion_move,
                    )
                    if reversion_move < minimum_reversion_move:
                        continue
                    entry_price = yes_price if side == "yes" else no_price
                    contracts = position_contracts(entry_price, config)
                    if remaining_size >= contracts:
                        entry_fee = taker_fee(contracts, entry_price)
                        position = TapePosition(
                            side=side,
                            contracts=contracts,
                            entry_price=entry_price,
                            entry_fee=entry_fee,
                            entry_ns=trade_ns,
                            anchor_target=pending_entry.candidate.anchor_target,
                            anchor_fair=pending_entry.candidate.anchor_fair,
                            event_type=pending_entry.candidate.event_type,
                            trigger_at_bat=(
                                pending_entry.candidate.trigger_at_bat
                            ),
                            trigger_pitch=pending_entry.candidate.trigger_pitch,
                            trigger_event_time_ns=(
                                pending_entry.candidate.event_time_ns
                            ),
                            held_price_history=[(trade_ns, entry_price)],
                        )
                        positions.append(position)
                        last_entry_ns = trade_ns
                        result.trades += 1
                        result.yes_trades += int(side == "yes")
                        result.no_trades += int(side == "no")
                        result.capital += contracts * entry_price + entry_fee
                        result.fees += entry_fee
                        pending_entry = None
                        candidate = None
                        continue

            if candidate is not None and pending_entry is None:
                target = _dynamic_target(candidate, current_fair)
                side, _ = segmented_trade_signal(
                    target, yes_price, candidate.event_type, config
                )
                if side is None:
                    candidate.watch_side = None
                    candidate.watch_started_ns = None
                elif candidate.watch_side != side:
                    candidate.watch_side = side
                    candidate.watch_started_ns = trade_ns
                elif (
                    candidate.watch_started_ns is not None
                    and trade_ns - candidate.watch_started_ns >= int(
                        segment_value(
                            config.confirmation_seconds_by_segment,
                            candidate.event_type,
                            side,
                            config.confirmation_seconds,
                        ) * 1_000_000_000
                    )
                ):
                    result.confirmed_signals += 1
                    pending_entry = PendingEntry(
                        candidate, side, trade_ns, yes_price
                    )
                    candidate = None

        for position in positions:
            won = (
                position.side == "yes" and home_win == 1
            ) or (
                position.side == "no" and home_win == 0
            )
            pnl = (
                (position.contracts if won else 0.0)
                - position.contracts * position.entry_price
                - position.entry_fee
            )
            result.pnl += pnl
            result.settlements += 1
            result.records.append(TapeTradeRecord(
                game_pk=game_pk,
                side=position.side,
                event_type=position.event_type,
                entry_time=_ns_to_timestamp(position.entry_ns),
                exit_time=None,
                exit_reason="settlement",
                entry_price=position.entry_price,
                exit_price=None,
                contracts=position.contracts,
                anchor_target=position.anchor_target,
                anchor_fair=position.anchor_fair,
                trigger_at_bat=position.trigger_at_bat,
                trigger_pitch=position.trigger_pitch,
                trigger_event_time=_ns_to_timestamp(
                    position.trigger_event_time_ns
                ),
                pnl=pnl,
                fees=position.entry_fee,
            ))
    return result
