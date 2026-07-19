"""Event-conditioned residual strategy shared by tuning, replay, and live use."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .strategy import CONFIG, taker_fee


HIT_EVENTS = frozenset({"single", "double", "triple", "home_run"})


@dataclass(frozen=True)
class HybridConfig:
    enabled: bool = True
    minimum_edge: float = 0.03
    max_hold_minutes: float = 5.0
    max_event_gap_seconds: float = 120.0
    live_confirmation_seconds: float = 2.0

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "HybridConfig":
        return cls(**json.loads(path.read_text()))


def _logit(value):
    clipped = np.clip(np.asarray(value, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(clipped / (1.0 - clipped))


def _expit(value):
    value = np.asarray(value, dtype=float)
    return 1.0 / (1.0 + np.exp(-value))


def anchored_event_target(pre_market, pre_fair, post_fair):
    """Apply only the model's state update to the pre-event market anchor.

    Absolute model calibration cancels out: the target moves the market's
    own pre-event probability by the local model's fair log-odds change.
    """
    return _expit(_logit(pre_market) + _logit(post_fair) - _logit(pre_fair))


def hybrid_signal(
    target: float,
    bid: float,
    ask: float,
    minimum_edge: float,
) -> tuple[str | None, float]:
    """Return a contrarian side only when the executable quote is excessive."""
    yes_edge = float(target) - float(ask)
    no_edge = float(bid) - float(target)
    if yes_edge >= minimum_edge and yes_edge >= no_edge:
        return "yes", yes_edge
    if no_edge >= minimum_edge:
        return "no", no_edge
    return None, max(yes_edge, no_edge)


def add_event_targets(
    frame: pd.DataFrame,
    config: HybridConfig,
) -> pd.DataFrame:
    """Mark the first candle that safely observes one isolated completed hit."""
    required = {
        "game_pk", "decision_time", "completed_event",
        "completed_event_sequence", "completed_event_time",
        "completed_event_pitch_start", "kalshi_price", "fair_prob",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing hybrid event fields: {sorted(missing)}")

    result = frame.copy()
    for column in [
        "decision_time", "completed_event_time", "completed_event_pitch_start",
    ]:
        result[column] = pd.to_datetime(result[column], utc=True)
    result = result.sort_values(["game_pk", "decision_time"])
    result["_hybrid_row_id"] = np.arange(len(result))
    grouped = result.groupby("game_pk", sort=False)
    result["previous_event_sequence"] = grouped[
        "completed_event_sequence"
    ].shift(1).fillna(0)
    result["event_sequence_increment"] = (
        result["completed_event_sequence"]
        - result["previous_event_sequence"]
    )
    event_detection = (
        result["completed_event"].isin(HIT_EVENTS)
        & result["event_sequence_increment"].eq(1)
        & (result["decision_time"] >= result["completed_event_time"])
    )

    # Anchor to the last candle that had closed by the terminal pitch's start.
    # The immediately preceding candle may already contain the hit reaction.
    left = result.loc[event_detection, [
        "_hybrid_row_id", "game_pk", "completed_event_pitch_start",
    ]].dropna(subset=["completed_event_pitch_start"])
    right = result[[
        "game_pk", "decision_time", "kalshi_price", "fair_prob",
    ]].rename(columns={
        "decision_time": "pre_event_time",
        "kalshi_price": "pre_event_market",
        "fair_prob": "pre_event_fair",
    })
    matched = pd.merge_asof(
        left.sort_values("completed_event_pitch_start"),
        right.sort_values("pre_event_time"),
        left_on="completed_event_pitch_start",
        right_on="pre_event_time",
        by="game_pk",
        direction="backward",
        allow_exact_matches=True,
    )
    matched = matched.set_index("_hybrid_row_id")
    for column in ["pre_event_time", "pre_event_market", "pre_event_fair"]:
        result[column] = result["_hybrid_row_id"].map(matched[column])

    result["event_detection_delay_seconds"] = (
        result["decision_time"] - result["completed_event_time"]
    ).dt.total_seconds()
    isolated_hit = (
        event_detection
        & result["pre_event_market"].notna()
        & result["pre_event_fair"].notna()
        & result["event_detection_delay_seconds"].between(
            0, config.max_event_gap_seconds
        )
    )
    result["hybrid_event"] = isolated_hit
    result["hybrid_target"] = np.nan
    result.loc[isolated_hit, "hybrid_target"] = anchored_event_target(
        result.loc[isolated_hit, "pre_event_market"],
        result.loc[isolated_hit, "pre_event_fair"],
        result.loc[isolated_hit, "fair_prob"],
    )
    result["market_move"] = (
        result["kalshi_price"] - result["pre_event_market"]
    )
    result["expected_move"] = (
        result["hybrid_target"] - result["pre_event_market"]
    )
    result["excess_move"] = (
        result["kalshi_price"] - result["hybrid_target"]
    )
    return result.drop(columns="_hybrid_row_id")


@dataclass
class HybridResult:
    trades: int = 0
    yes_trades: int = 0
    no_trades: int = 0
    early_exits: int = 0
    timed_exits: int = 0
    invalidated_exits: int = 0
    settlements: int = 0
    fees: float = 0.0
    capital: float = 0.0
    pnl: float = 0.0

    @property
    def roi(self) -> float:
        return self.pnl / self.capital if self.capital else 0.0


@dataclass
class HybridPosition:
    side: str
    contracts: float
    entry_price: float
    entry_fee: float
    entry_time: pd.Timestamp
    anchor_target: float
    anchor_fair: float
    event_sequence: int


def simulate_hybrid(frame: pd.DataFrame, config: HybridConfig) -> HybridResult:
    """Confirm at the next quote and also execute exits one quote later."""
    result = HybridResult()
    prepared = add_event_targets(frame, config)

    for _, game in prepared.groupby("game_pk", sort=False):
        game = game.sort_values("decision_time")
        position: HybridPosition | None = None
        pending_entry: tuple[str, float, float, int] | None = None
        pending_exit: str | None = None

        for row in game.itertuples():
            bid = float(row.yes_bid_close)
            ask = float(row.yes_ask_close)
            now = pd.Timestamp(row.decision_time)
            event_sequence = int(row.completed_event_sequence)

            if pending_exit is not None and position is not None:
                exit_price = bid if position.side == "yes" else 1.0 - ask
                exit_fee = taker_fee(position.contracts, exit_price)
                proceeds = position.contracts * exit_price - exit_fee
                result.pnl += (
                    proceeds
                    - position.contracts * position.entry_price
                    - position.entry_fee
                )
                result.fees += exit_fee
                if pending_exit == "reversion":
                    result.early_exits += 1
                elif pending_exit == "timeout":
                    result.timed_exits += 1
                else:
                    result.invalidated_exits += 1
                position = None
                pending_exit = None

            if pending_entry is not None and position is None:
                side, anchor_target, anchor_fair, signal_sequence = pending_entry
                target = float(anchored_event_target(
                    anchor_target, anchor_fair, row.fair_prob
                ))
                confirmed_side, _ = hybrid_signal(
                    target, bid, ask, config.minimum_edge
                )
                if confirmed_side == side:
                    price = ask if side == "yes" else 1.0 - bid
                    contracts = CONFIG.bet_size / price
                    entry_fee = taker_fee(contracts, price)
                    position = HybridPosition(
                        side=side,
                        contracts=contracts,
                        entry_price=price,
                        entry_fee=entry_fee,
                        entry_time=now,
                        anchor_target=anchor_target,
                        anchor_fair=anchor_fair,
                        event_sequence=signal_sequence,
                    )
                    result.trades += 1
                    result.yes_trades += int(side == "yes")
                    result.no_trades += int(side == "no")
                    result.capital += CONFIG.bet_size + entry_fee
                    result.fees += entry_fee
                pending_entry = None

            if position is not None:
                target = float(anchored_event_target(
                    position.anchor_target,
                    position.anchor_fair,
                    row.fair_prob,
                ))
                reverted = (
                    position.side == "yes" and bid >= target
                ) or (
                    position.side == "no" and ask <= target
                )
                age_minutes = (
                    now - position.entry_time
                ).total_seconds() / 60.0
                if reverted:
                    pending_exit = "reversion"
                elif age_minutes >= config.max_hold_minutes:
                    pending_exit = "timeout"
            elif pending_entry is None and bool(row.hybrid_event):
                target = float(row.hybrid_target)
                side, _ = hybrid_signal(
                    target, bid, ask, config.minimum_edge
                )
                if side is not None:
                    pending_entry = (
                        side, target, float(row.fair_prob), event_sequence
                    )

        if position is not None:
            home_win = int(game.iloc[-1]["home_win"])
            won = (
                position.side == "yes" and home_win == 1
            ) or (
                position.side == "no" and home_win == 0
            )
            result.pnl += (
                (position.contracts if won else 0.0)
                - position.contracts * position.entry_price
                - position.entry_fee
            )
            result.settlements += 1
    return result
