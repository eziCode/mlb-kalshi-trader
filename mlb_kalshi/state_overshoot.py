"""Causal, event-agnostic market-overshoot candidate construction.

Every completed pitch may create a candidate.  Event names are deliberately
ignored: the signal is the difference between the observed market move and
the log-odds move justified by the local win-expectancy model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from mlb_kalshi.hybrid import anchored_event_target
from mlb_kalshi.strategy import CONFIG, taker_fee
from mlb_kalshi.trade_tape import compatible_taker


@dataclass(frozen=True)
class OvershootConfig:
    enabled: bool = False
    minimum_logit_residual: float = 0.08
    confirmation_seconds: float = 1.0
    maximum_pre_event_trade_age_seconds: float = 5.0
    maximum_entry_latency_seconds: float = 10.0
    maximum_outcome_seconds: float = 120.0
    maximum_adverse_logit_move: float = 0.12
    minimum_reversion_probability: float = 0.55
    minimum_expected_pnl: float = 0.0
    model_path: str = "state_reversion.cbm"


@dataclass
class StateReversionResult:
    candidates: int = 0
    accepted: int = 0
    rejected: int = 0
    reversion_exits: int = 0
    adverse_stop_exits: int = 0
    timeout_exits: int = 0
    settlements: int = 0
    fees: float = 0.0
    capital: float = 0.0
    pnl: float = 0.0
    records: list[dict] = field(default_factory=list)

    @property
    def roi(self) -> float:
        return self.pnl / self.capital if self.capital else 0.0


def _logit(value) -> float:
    value = float(np.clip(value, 1e-4, 1 - 1e-4))
    return float(np.log(value / (1.0 - value)))


def _expit(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(value))))


def signed_logit_residual(yes_price: float, target: float) -> float:
    """Positive means YES is rich; negative means YES is cheap."""
    return _logit(yes_price) - _logit(target)


def _dynamic_target(anchor_target: float, anchor_fair: float, fair: float) -> float:
    return float(anchored_event_target(anchor_target, anchor_fair, fair))


def _state_fields(update) -> dict:
    return {
        "inning_after": float(update.inning_after),
        "inning_topbot_after": float(update.inning_topbot_after),
        "outs_when_up_after": float(update.outs_when_up_after),
        "score_diff_after": float(update.score_diff_after),
        "balls_after": float(update.balls_after),
        "strikes_after": float(update.strikes_after),
        "runner_on_first_after": float(update.runner_on_first_after),
        "runner_on_second_after": float(update.runner_on_second_after),
        "runner_on_third_after": float(update.runner_on_third_after),
    }


def _pre_trade_features(times, prices, sizes, sides, entry_index: int) -> dict:
    start_ns = times[entry_index] - 2_000_000_000
    start = int(np.searchsorted(times, start_ns, side="left"))
    selected_sizes = sizes[start:entry_index]
    volume = float(selected_sizes.sum())
    signed = np.where(sides[start:entry_index] == "yes", 1.0, -1.0)
    return {
        "pre_trade_count_2s": float(entry_index - start),
        "pre_volume_2s": volume,
        "pre_flow_imbalance_2s": (
            float((selected_sizes * signed).sum() / volume) if volume else 0.0
        ),
        "pre_price_volatility_2s": (
            float(np.std(prices[start:entry_index])) if entry_index > start else 0.0
        ),
    }


def build_state_overshoot_candidates(
    trades: pd.DataFrame,
    updates: pd.DataFrame,
    config: OvershootConfig,
) -> pd.DataFrame:
    """Create independently evaluated candidates from every pitch transition.

    A signal must persist and then fill on a strictly later, compatible trade.
    Its outcome is the first later compatible trade after dynamic residual
    reversion; otherwise the position is valued at actual settlement.  These
    independent outcomes are labels, not a portfolio backtest.
    """
    required_trades = {
        "game_pk", "created_time", "trade_id", "yes_price_dollars",
        "no_price_dollars", "count_fp", "taker_outcome_side", "home_win",
    }
    required_updates = {
        "game_pk", "game_date", "pitch_start_time", "pitch_end_time",
        "fair_before", "fair_after", "at_bat_number", "pitch_number",
        "inning_after", "inning_topbot_after", "outs_when_up_after",
        "score_diff_after", "balls_after", "strikes_after",
        "runner_on_first_after", "runner_on_second_after",
        "runner_on_third_after",
    }
    if missing := required_trades - set(trades):
        raise ValueError(f"Trade tape is missing columns: {sorted(missing)}")
    if missing := required_updates - set(updates):
        raise ValueError(f"State updates are missing columns: {sorted(missing)}")

    confirmation_ns = int(config.confirmation_seconds * 1e9)
    anchor_age_ns = int(config.maximum_pre_event_trade_age_seconds * 1e9)
    entry_age_ns = int(config.maximum_entry_latency_seconds * 1e9)
    outcome_ns = int(config.maximum_outcome_seconds * 1e9)
    rows: list[dict] = []

    for game_pk, game_updates in updates.groupby("game_pk", sort=False):
        tape = trades[trades["game_pk"].eq(game_pk)].sort_values(
            ["created_time", "trade_id"]
        )
        if tape.empty:
            continue
        game_updates = game_updates.sort_values("pitch_end_time")
        times = pd.to_datetime(tape["created_time"], utc=True).array.as_unit("ns").asi8
        yes = tape["yes_price_dollars"].to_numpy(float)
        no = tape["no_price_dollars"].to_numpy(float)
        sizes = tape["count_fp"].to_numpy(float)
        sides = tape["taker_outcome_side"].astype(str).to_numpy()
        home_win = int(tape["home_win"].iloc[-1])
        update_rows = list(game_updates.itertuples(index=False))
        update_end_times = np.array([
            pd.Timestamp(row.pitch_end_time).value for row in update_rows
        ], dtype=np.int64)

        for update_pos, update in enumerate(update_rows):
            start_ns = int(update_end_times[update_pos])
            pitch_start_ns = pd.Timestamp(update.pitch_start_time).value
            anchor_i = int(np.searchsorted(times, pitch_start_ns, side="left") - 1)
            if anchor_i < 0 or pitch_start_ns - times[anchor_i] > anchor_age_ns:
                continue
            anchor_market = float(yes[anchor_i])
            target = float(anchored_event_target(
                anchor_market, float(update.fair_before), float(update.fair_after)
            ))
            stop_i = int(np.searchsorted(times, start_ns + entry_age_ns, side="right"))
            first_i = int(np.searchsorted(times, start_ns, side="right"))
            watch_side = None
            watch_ns = None
            entry_i = None
            residual_at_entry = None
            for i in range(first_i, stop_i):
                residual = signed_logit_residual(float(yes[i]), target)
                side = (
                    "no" if residual >= config.minimum_logit_residual else
                    "yes" if residual <= -config.minimum_logit_residual else None
                )
                if side is None:
                    watch_side, watch_ns = None, None
                elif side != watch_side:
                    watch_side, watch_ns = side, int(times[i])
                elif (
                    watch_ns is not None
                    and times[i] - watch_ns >= confirmation_ns
                    and times[i] > watch_ns
                    and compatible_taker(side, sides[i])
                ):
                    price = float(yes[i] if side == "yes" else no[i])
                    contracts = CONFIG.bet_size / price
                    if sizes[i] >= contracts:
                        entry_i, residual_at_entry = i, residual
                        break
            if entry_i is None:
                continue

            side = str(watch_side)
            entry_price = float(yes[entry_i] if side == "yes" else no[entry_i])
            contracts = CONFIG.bet_size / entry_price
            entry_fee = taker_fee(contracts, entry_price)
            deadline_ns = int(times[entry_i] + outcome_ns)
            current_fair = float(update.fair_after)
            next_update = update_pos + 1
            pending_exit_ns = None
            pending_exit_reason = None
            exit_i = None
            exit_target = None
            max_favorable = -np.inf
            max_adverse = -np.inf
            entry_directional_residual = (
                float(residual_at_entry) if side == "no" else -float(residual_at_entry)
            )
            for i in range(entry_i + 1, len(times)):
                while next_update < len(update_rows) and update_end_times[next_update] <= times[i]:
                    current_fair = float(update_rows[next_update].fair_after)
                    next_update += 1
                dynamic_target = _dynamic_target(target, float(update.fair_after), current_fair)
                held = float(yes[i] if side == "yes" else no[i])
                liquidation = contracts * (held - entry_price) - entry_fee - taker_fee(contracts, held)
                max_favorable = max(max_favorable, liquidation)
                max_adverse = max(max_adverse, -liquidation)
                reverted = yes[i] >= dynamic_target if side == "yes" else yes[i] <= dynamic_target
                if pending_exit_ns is not None:
                    exit_taker = "no" if side == "yes" else "yes"
                    if pending_exit_reason == "reversion" and not reverted:
                        pending_exit_ns = None
                        pending_exit_reason = None
                    elif times[i] > pending_exit_ns and compatible_taker(exit_taker, sides[i]) and sizes[i] >= contracts:
                        exit_i, exit_target = i, dynamic_target
                        break
                if pending_exit_ns is not None:
                    continue
                current_residual = signed_logit_residual(float(yes[i]), dynamic_target)
                directional_residual = current_residual if side == "no" else -current_residual
                adverse = directional_residual >= (
                    entry_directional_residual + config.maximum_adverse_logit_move
                )
                if reverted:
                    pending_exit_ns = int(times[i])
                    pending_exit_reason = "reversion"
                elif adverse:
                    pending_exit_ns = int(times[i])
                    pending_exit_reason = "adverse_stop"
                elif times[i] >= deadline_ns:
                    pending_exit_ns = int(times[i])
                    pending_exit_reason = "opportunity_timeout"

            if exit_i is not None:
                exit_price = float(yes[exit_i] if side == "yes" else no[exit_i])
                exit_fee = taker_fee(contracts, exit_price)
                pnl = contracts * (exit_price - entry_price) - entry_fee - exit_fee
                exit_reason = str(pending_exit_reason)
                exit_time = pd.Timestamp(times[exit_i], tz="UTC")
            else:
                won = (side == "yes" and home_win == 1) or (side == "no" and home_win == 0)
                exit_price = 1.0 if won else 0.0
                exit_fee = 0.0
                pnl = (contracts if won else 0.0) - contracts * entry_price - entry_fee
                exit_reason = "settlement"
                exit_time = pd.NaT
                exit_target = np.nan

            target_contract = target if side == "yes" else 1.0 - target
            target_fee = taker_fee(contracts, target_contract)
            target_pnl = contracts * (target_contract - entry_price) - entry_fee - target_fee
            adverse_yes = _expit(
                _logit(float(yes[entry_i]))
                + (config.maximum_adverse_logit_move if side == "no" else -config.maximum_adverse_logit_move)
            )
            adverse_contract = adverse_yes if side == "yes" else 1.0 - adverse_yes
            failure_pnl = (
                contracts * (adverse_contract - entry_price)
                - entry_fee - taker_fee(contracts, adverse_contract)
            )
            row = {
                "policy_version": 2,
                "game_pk": int(game_pk), "game_date": update.game_date,
                "trigger_at_bat": int(update.at_bat_number),
                "trigger_pitch": int(update.pitch_number),
                "trigger_time": pd.Timestamp(start_ns, tz="UTC"),
                "entry_time": pd.Timestamp(times[entry_i], tz="UTC"),
                "exit_time": exit_time, "exit_reason": exit_reason,
                "side": side, "home_win": home_win,
                "anchor_market": anchor_market, "fair_before": float(update.fair_before),
                "fair_after": float(update.fair_after), "target_home_price": target,
                "entry_home_price": float(yes[entry_i]),
                "signed_logit_residual": float(residual_at_entry),
                "absolute_logit_residual": abs(float(residual_at_entry)),
                "fair_logit_move": _logit(update.fair_after) - _logit(update.fair_before),
                "entry_price": entry_price, "target_contract_price": target_contract,
                "contracts": contracts, "entry_fee": entry_fee,
                "target_reversion_pnl": target_pnl,
                "failure_pnl": failure_pnl,
                "breakeven_reversion_probability": (
                    -failure_pnl / (target_pnl - failure_pnl) if target_pnl > failure_pnl else 1.0
                ),
                "entry_latency_seconds": (times[entry_i] - start_ns) / 1e9,
                "entry_side_yes": float(side == "yes"),
                "exit_price": exit_price, "exit_target": exit_target,
                "pnl": pnl, "fees": entry_fee + exit_fee,
                "profitable_reversion": int(exit_reason == "reversion" and pnl > 0),
                "max_favorable_pnl": float(max_favorable if np.isfinite(max_favorable) else 0),
                "max_adverse_pnl": float(max_adverse if np.isfinite(max_adverse) else 0),
                **_state_fields(update),
                **_pre_trade_features(times, yes, sizes, sides, entry_i),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def simulate_state_reversion(
    examples: pd.DataFrame,
    probabilities,
    config: OvershootConfig,
    expected_pnls=None,
) -> StateReversionResult:
    """Apply a model gate and enforce one chronological position per market."""
    frame = examples.copy()
    frame["predicted_reversion_probability"] = np.asarray(probabilities, float)
    if expected_pnls is None:
        frame["expected_pnl"] = (
            frame["predicted_reversion_probability"]
            * frame["target_reversion_pnl"]
            + (1.0 - frame["predicted_reversion_probability"])
            * frame["failure_pnl"]
        )
    else:
        values = np.asarray(expected_pnls, float)
        if len(values) != len(frame):
            raise ValueError("Expected-PnL predictions must align with examples")
        frame["expected_pnl"] = values
    result = StateReversionResult(candidates=len(frame))
    for _, game in frame.groupby("game_pk", sort=False):
        occupied_until = None
        for row in game.sort_values("entry_time").itertuples(index=False):
            if occupied_until is not None and pd.Timestamp(row.entry_time) <= occupied_until:
                continue
            accepted = (
                row.absolute_logit_residual >= config.minimum_logit_residual
                and
                row.predicted_reversion_probability >= config.minimum_reversion_probability
                and row.expected_pnl >= config.minimum_expected_pnl
            )
            if not accepted:
                result.rejected += 1
                continue
            result.accepted += 1
            result.pnl += float(row.pnl)
            result.fees += float(row.fees)
            result.capital += CONFIG.bet_size + float(row.entry_fee)
            result.reversion_exits += int(row.exit_reason == "reversion")
            result.adverse_stop_exits += int(row.exit_reason == "adverse_stop")
            result.timeout_exits += int(row.exit_reason == "opportunity_timeout")
            result.settlements += int(row.exit_reason == "settlement")
            result.records.append({**row._asdict(), "config": asdict(config)})
            occupied_until = (
                pd.Timestamp(row.exit_time)
                if pd.notna(row.exit_time) else pd.Timestamp.max.tz_localize("UTC")
            )
    return result
