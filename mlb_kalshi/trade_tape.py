"""Sub-second trade-tape simulation for the hybrid hit residual strategy.

The tape contains executed trades, not quotes. Fills therefore require a
strictly later observed trade on the compatible taker outcome side and enough
reported trade size. This is still a proxy for execution, not a reconstructed
order book.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mlb_kalshi.hybrid import anchored_event_target
from mlb_kalshi.strategy import CONFIG, taker_fee


@dataclass(frozen=True)
class TradeTapeConfig:
    enabled: bool = True
    minimum_edge: float = 0.04
    confirmation_seconds: float = 2.0
    maximum_pre_event_trade_age_seconds: float = 5.0
    minimum_fair_move: float = 0.005


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
    confirmed_signals: int = 0
    trades: int = 0
    yes_trades: int = 0
    no_trades: int = 0
    reversion_exits: int = 0
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
    yes_edge = target - yes_price
    no_edge = yes_price - target
    if yes_edge >= minimum_edge:
        return "yes", yes_edge
    if no_edge >= minimum_edge:
        return "no", no_edge
    return None, max(yes_edge, no_edge)


def compatible_taker(side: str, taker_outcome_side: str) -> bool:
    return side == taker_outcome_side


def _dynamic_target(candidate_or_position, current_fair: float) -> float:
    return float(anchored_event_target(
        candidate_or_position.anchor_target,
        candidate_or_position.anchor_fair,
        current_fair,
    ))


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
    confirmation_ns = int(config.confirmation_seconds * 1_000_000_000)
    maximum_anchor_age_ns = int(
        config.maximum_pre_event_trade_age_seconds * 1_000_000_000
    )

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
        position: TapePosition | None = None
        pending_exit_ns: int | None = None

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
                    position is None
                    and active_candidate is not None
                    and pd.Timestamp(update.pitch_end_time).value
                    > active_candidate.event_time_ns
                ):
                    completed_plate_appearance = pd.notna(update.completed_event)
                    material_change = (
                        _material_state(update) != active_candidate.material_state
                    )
                    if completed_plate_appearance or material_change:
                        result.invalidated_candidates += 1
                        candidate = None
                        pending_entry = None
                current_fair = float(update.fair_after)
                if bool(update.is_hit):
                    result.observed_hits += 1
                    if position is None:
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

            if position is not None:
                target = _dynamic_target(position, current_fair)
                reverted = (
                    position.side == "yes" and yes_price >= target
                ) or (
                    position.side == "no" and yes_price <= target
                )
                exit_taker_side = "no" if position.side == "yes" else "yes"
                if pending_exit_ns is not None:
                    if not reverted:
                        pending_exit_ns = None
                    elif (
                        trade_ns > pending_exit_ns
                        and compatible_taker(exit_taker_side, taker_side)
                        and size >= position.contracts
                    ):
                        exit_price = (
                            yes_price if position.side == "yes" else no_price
                        )
                        exit_fee = taker_fee(position.contracts, exit_price)
                        pnl = (
                            position.contracts * exit_price
                            - exit_fee
                            - position.contracts * position.entry_price
                            - position.entry_fee
                        )
                        result.pnl += pnl
                        result.fees += exit_fee
                        result.reversion_exits += 1
                        result.records.append(TapeTradeRecord(
                            game_pk=game_pk,
                            side=position.side,
                            event_type=position.event_type,
                            entry_time=_ns_to_timestamp(position.entry_ns),
                            exit_time=_ns_to_timestamp(trade_ns),
                            exit_reason="reversion",
                            entry_price=position.entry_price,
                            exit_price=exit_price,
                            contracts=position.contracts,
                            anchor_target=position.anchor_target,
                            anchor_fair=position.anchor_fair,
                            trigger_at_bat=position.trigger_at_bat,
                            trigger_pitch=position.trigger_pitch,
                            trigger_event_time=_ns_to_timestamp(
                                position.trigger_event_time_ns
                            ),
                            pnl=pnl,
                            fees=position.entry_fee + exit_fee,
                        ))
                        position = None
                        pending_exit_ns = None
                        continue
                if position is not None and reverted and pending_exit_ns is None:
                    pending_exit_ns = trade_ns
                continue

            if pending_entry is not None:
                target = _dynamic_target(pending_entry.candidate, current_fair)
                side, _ = trade_signal(target, yes_price, config.minimum_edge)
                if side != pending_entry.side:
                    candidate = pending_entry.candidate
                    candidate.watch_side = None
                    candidate.watch_started_ns = None
                    pending_entry = None
                elif (
                    trade_ns > pending_entry.created_ns
                    and compatible_taker(side, taker_side)
                ):
                    entry_price = yes_price if side == "yes" else no_price
                    contracts = CONFIG.bet_size / entry_price
                    if size >= contracts:
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
                        )
                        result.trades += 1
                        result.yes_trades += int(side == "yes")
                        result.no_trades += int(side == "no")
                        result.capital += CONFIG.bet_size + entry_fee
                        result.fees += entry_fee
                        pending_entry = None
                        candidate = None
                        continue

            if candidate is not None and pending_entry is None:
                target = _dynamic_target(candidate, current_fair)
                side, _ = trade_signal(target, yes_price, config.minimum_edge)
                if side is None:
                    candidate.watch_side = None
                    candidate.watch_started_ns = None
                elif candidate.watch_side != side:
                    candidate.watch_side = side
                    candidate.watch_started_ns = trade_ns
                elif (
                    candidate.watch_started_ns is not None
                    and trade_ns - candidate.watch_started_ns >= confirmation_ns
                ):
                    result.confirmed_signals += 1
                    pending_entry = PendingEntry(candidate, side, trade_ns)
                    candidate = None

        if position is not None:
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
