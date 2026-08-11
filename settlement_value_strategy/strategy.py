"""Event-agnostic settlement-value modeling and causal trade simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from decimal import Decimal, ROUND_DOWN
from types import SimpleNamespace

import numpy as np
import pandas as pd

TAKER_FEE_RATE = 0.07


def _logit(value):
    clipped = np.clip(np.asarray(value, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(clipped / (1.0 - clipped))


def _expit(value):
    value = np.asarray(value, dtype=float)
    return 1.0 / (1.0 + np.exp(-value))


def market_adjusted_probability(raw_probability, market_probability, calibration):
    """Apply the frozen probability transform used by training and live code."""
    if calibration.get("mode") == "identity":
        return np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1 - 1e-6)
    market_logit = _logit(market_probability)
    model_delta = _logit(raw_probability) - market_logit
    return _expit(
        market_logit
        + float(calibration["intercept"])
        + float(calibration["coefficient"]) * model_delta
    )


def anchored_event_target(pre_market, pre_fair, post_fair):
    return _expit(_logit(pre_market) + _logit(post_fair) - _logit(pre_fair))


AFTER_STATE_COLUMNS = (
    "inning_after", "inning_topbot_after", "outs_when_up_after",
    "score_diff_after", "balls_after", "strikes_after",
    "runner_on_first_after", "runner_on_second_after", "runner_on_third_after",
)


def add_before_and_delta_state(updates: pd.DataFrame) -> pd.DataFrame:
    result = updates.copy()
    result["pitch_end_time"] = pd.to_datetime(result["pitch_end_time"], utc=True)
    result = result.sort_values([
        "game_pk", "pitch_end_time", "at_bat_number", "pitch_number",
    ])
    grouped = result.groupby("game_pk", sort=False)
    for after in AFTER_STATE_COLUMNS:
        if after not in result:
            raise ValueError(f"State updates are missing {after}")
        base = after.removesuffix("_after")
        before = f"{base}_before"
        delta = "delta_outs" if base == "outs_when_up" else f"delta_{base}"
        result[before] = grouped[after].shift(1)
        result[delta] = result[after] - result[before]
    return result


def taker_fee(contracts: float, price: float) -> float:
    if contracts <= 0:
        return 0.0
    principal = contracts * price
    raw = TAKER_FEE_RATE * contracts * price * (1.0 - price)
    rounded_total = math.ceil((principal + raw) * 10_000.0 - 1e-12) / 10_000.0
    return max(0.0, rounded_total - principal)


def contracts_for_budget(
    price: float, budget: float, available_contracts: float | None = None,
) -> float:
    """IOC quantity filled at one observed price, including fees in budget."""
    count = float(Decimal(str(budget / price)).quantize(
        Decimal("0.01"), rounding=ROUND_DOWN
    ))
    while count > 0 and count * price + taker_fee(count, price) > budget + 1e-9:
        count = round(count - .01, 2)
    if available_contracts is not None:
        available = float(Decimal(str(max(0.0, available_contracts))).quantize(
            Decimal("0.01"), rounding=ROUND_DOWN
        ))
        count = min(count, available)
    return max(0.0, count)


def reversal_economics(
    conflicting_positions, exit_price: float, new_expected_pnl: float,
) -> dict:
    """Evaluate selling conflicts before buying the opposite outcome."""
    exit_price = float(exit_price)
    unwind_pnl = 0.0
    exit_fees = 0.0
    proceeds = 0.0
    for position in conflicting_positions:
        fee = taker_fee(float(position.contracts), exit_price)
        gross = float(position.contracts) * exit_price
        entry_cost = (
            float(position.contracts) * float(position.entry_price)
            + float(position.entry_fee)
        )
        proceeds += gross - fee
        exit_fees += fee
        unwind_pnl += gross - fee - entry_cost
    net_value = float(new_expected_pnl) + unwind_pnl
    return {
        "unwind_pnl": unwind_pnl,
        "unwind_loss": max(0.0, -unwind_pnl),
        "exit_fees": exit_fees,
        "net_proceeds": proceeds,
        "new_expected_pnl": float(new_expected_pnl),
        "net_reversal_value": net_value,
        "allowed": bool(conflicting_positions) and net_value > 0.0,
    }


def compatible_taker(side: str, taker_outcome_side: str) -> bool:
    return side == taker_outcome_side


def confirmation_taker_allowed(
    expected_side: str, taker_outcome_side: str, config,
) -> bool:
    if config.confirmation_taker_side == "any":
        return taker_outcome_side in {"yes", "no"}
    return compatible_taker(expected_side, taker_outcome_side)


MISPRICING_FEATURES = (
    "market_home_price", "market_logit", "local_fair_after",
    "local_fair_before", "fair_logit_move", "market_logit_move",
    "anchored_state_target", "market_target_residual",
    "inning_after", "inning_topbot_after", "outs_when_up_after",
    "score_diff_after", "balls_after", "strikes_after",
    "runner_on_first_after", "runner_on_second_after", "runner_on_third_after",
    "delta_inning", "delta_outs", "delta_score_diff", "delta_balls",
    "delta_strikes", "delta_runner_on_first", "delta_runner_on_second",
    "delta_runner_on_third", "anchor_age_seconds", "observation_delay_seconds",
    "pre_trade_count_2s", "pre_volume_2s", "pre_flow_imbalance_2s",
    "pre_price_volatility_2s",
)


@dataclass(frozen=True)
class MispricingConfig:
    enabled: bool = False
    observation_delay_seconds: float = 1.0
    anchor_buffer_seconds: float = 2.0
    maximum_anchor_age_seconds: float = 5.0
    maximum_fill_delay_seconds: float = 5.0
    minimum_expected_pnl: float = 0.50
    minimum_probability_edge: float = 0.03
    bet_size: float = 10.0
    side_filter: str = "both"
    minimum_seconds_between_entries: float = 200.0
    execution_contract: str = "home_both"
    maximum_positions_per_game: int = 0  # Zero means unlimited.
    conditional_stacking: bool = True
    excluded_price_min: float = 0.0
    excluded_price_max: float = 0.0
    confirmation_taker_side: str = "compatible"
    require_post_signal_trade: bool = True
    submission_latency_seconds: float = 0.0
    minimum_entry_inning: int = 1
    early_exit_enabled: bool = False
    early_exit_minimum_hold_seconds: float = 60.0
    early_exit_stop_loss_points: float = 0.20
    early_exit_minimum_inning: int = 1


def entry_state_allowed(row, config: MispricingConfig) -> bool:
    """Shared live/replay eligibility for state-dependent entry filters."""
    inning = (
        row.get("inning_after", 1)
        if isinstance(row, dict)
        else getattr(row, "inning_after", 1)
    )
    return int(float(inning)) >= int(config.minimum_entry_inning)


def execution_indexes(
    times: np.ndarray, signal_ns: int, deadline: int,
    config: MispricingConfig,
) -> range:
    if config.require_post_signal_trade:
        effective_ns = signal_ns + int(
            config.submission_latency_seconds * 1e9
        )
        start = int(np.searchsorted(times, effective_ns, side="right"))
        stop = int(np.searchsorted(times, deadline, side="left"))
        return range(start, stop)
    index = int(np.searchsorted(times, signal_ns, side="right") - 1)
    if (
        index < 0
        or signal_ns - int(times[index])
        > int(config.maximum_fill_delay_seconds * 1e9)
    ):
        return range(0)
    return range(index, index + 1)


def execution_price_allowed(price: float, config: MispricingConfig) -> bool:
    """Reject a configured closed interval of execution prices."""
    if config.excluded_price_max <= config.excluded_price_min:
        return True
    return not (
        config.excluded_price_min <= price <= config.excluded_price_max
    )


@dataclass
class MispricingResult:
    signals: int = 0
    orders: int = 0
    trades: int = 0
    yes_trades: int = 0
    no_trades: int = 0
    fees: float = 0.0
    capital: float = 0.0
    pnl: float = 0.0
    records: list[dict] = field(default_factory=list)

    @property
    def roi(self):
        return self.pnl / self.capital if self.capital else 0.0


def mispricing_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(MISPRICING_FEATURES) - set(frame)
    if missing:
        raise ValueError(f"Missing mispricing features: {sorted(missing)}")
    result = frame.loc[:, MISPRICING_FEATURES].apply(pd.to_numeric, errors="coerce")
    if result.isna().any().any():
        bad = result.columns[result.isna().any()].tolist()
        raise ValueError(f"Mispricing features contain nulls: {bad}")
    return result.astype(float)


def _flow(times, prices, sizes, sides, index):
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


def build_mispricing_dataset(
    trades: pd.DataFrame,
    updates: pd.DataFrame,
    config: MispricingConfig | None = None,
) -> pd.DataFrame:
    config = config or MispricingConfig()
    prepared = add_before_and_delta_state(updates).dropna(subset=[
        "inning_before", "outs_when_up_before", "score_diff_before",
        "balls_before", "strikes_before", "runner_on_first_before",
        "runner_on_second_before", "runner_on_third_before",
    ])
    delay_ns = int(config.observation_delay_seconds * 1e9)
    buffer_ns = int(config.anchor_buffer_seconds * 1e9)
    max_anchor_ns = int(config.maximum_anchor_age_seconds * 1e9)
    rows = []
    for game_pk, game_updates in prepared.groupby("game_pk", sort=False):
        tape = trades[trades.game_pk.eq(game_pk)].sort_values(
            ["created_time", "trade_id"]
        )
        if tape.empty:
            continue
        times = pd.to_datetime(tape.created_time, utc=True).array.as_unit("ns").asi8
        prices = tape.yes_price_dollars.to_numpy(float)
        sizes = tape.count_fp.to_numpy(float)
        sides = tape.taker_outcome_side.astype(str).to_numpy()
        update_rows = list(game_updates.sort_values("pitch_end_time").itertuples(index=False))
        update_times = np.array([
            pd.Timestamp(row.pitch_end_time).value for row in update_rows
        ], dtype=np.int64)
        for pos, update in enumerate(update_rows):
            event_ns = int(update_times[pos])
            cutoff = event_ns - buffer_ns
            anchor_i = int(np.searchsorted(times, cutoff, side="left") - 1)
            signal_i = int(np.searchsorted(times, event_ns + delay_ns, side="left"))
            if (
                anchor_i < 0 or signal_i >= len(times)
                or cutoff - times[anchor_i] > max_anchor_ns
                or times[signal_i] > event_ns + 10_000_000_000
            ):
                continue
            anchor_market = float(prices[anchor_i])
            market = float(prices[signal_i])
            target = float(anchored_event_target(
                anchor_market, update.fair_before, update.fair_after
            ))
            fair_move = _logit(update.fair_after) - _logit(update.fair_before)
            market_move = _logit(market) - _logit(anchor_market)
            row = {
                "dataset_version": 1,
                "atomic_play_input": bool(update.atomic_play_input),
                "game_pk": int(game_pk), "game_date": update.game_date,
                "home_win": int(update.home_win),
                "trigger_at_bat": int(update.at_bat_number),
                "trigger_pitch": int(update.pitch_number),
                "trigger_time": pd.Timestamp(event_ns, tz="UTC"),
                "signal_time": pd.Timestamp(times[signal_i], tz="UTC"),
                "next_update_time": (
                    pd.Timestamp(update_times[pos + 1], tz="UTC")
                    if pos + 1 < len(update_rows) else pd.NaT
                ),
                "market_home_price": market, "market_logit": _logit(market),
                "local_fair_after": float(update.fair_after),
                "local_fair_before": float(update.fair_before),
                "fair_logit_move": fair_move, "market_logit_move": market_move,
                "anchored_state_target": target,
                "market_target_residual": market - target,
                "anchor_age_seconds": (cutoff - times[anchor_i]) / 1e9,
                "observation_delay_seconds": (times[signal_i] - event_ns) / 1e9,
                **{name: float(getattr(update, name)) for name in [
                    "inning_after", "inning_topbot_after", "outs_when_up_after",
                    "score_diff_after", "balls_after", "strikes_after",
                    "runner_on_first_after", "runner_on_second_after",
                    "runner_on_third_after", "delta_inning", "delta_outs",
                    "delta_score_diff", "delta_balls", "delta_strikes",
                    "delta_runner_on_first", "delta_runner_on_second",
                    "delta_runner_on_third",
                ]},
                **_flow(times, prices, sizes, sides, signal_i),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def signal_economics(probability: float, yes_price: float, bet_size: float = 10.0):
    no_price = 1.0 - yes_price
    yes_contracts = bet_size / yes_price
    no_contracts = bet_size / no_price
    yes_fee = taker_fee(yes_contracts, yes_price)
    no_fee = taker_fee(no_contracts, no_price)
    yes_ev = yes_contracts * (probability - yes_price) - yes_fee
    no_ev = no_contracts * ((1.0 - probability) - no_price) - no_fee
    return yes_ev, no_ev


def model_signal(
    probability: float, market_home_price: float, config: MispricingConfig,
) -> tuple[str, float, float, bool]:
    """Return the identical model-side decision for replay and live trading."""
    yes_ev, no_ev = signal_economics(
        probability, market_home_price, config.bet_size
    )
    if yes_ev >= no_ev:
        side = "yes"
        expected_pnl = yes_ev
        edge = probability - market_home_price
    else:
        side = "no"
        expected_pnl = no_ev
        edge = market_home_price - probability
    eligible = (
        (config.side_filter == "both" or side == config.side_filter)
        and expected_pnl >= config.minimum_expected_pnl
        and edge >= config.minimum_probability_edge
    )
    return side, float(expected_pnl), float(edge), bool(eligible)


def simulate_mispricing(
    frame: pd.DataFrame,
    probabilities,
    trades: pd.DataFrame,
    config: MispricingConfig,
) -> MispricingResult:
    work = frame.copy()
    work["fair_probability"] = np.asarray(probabilities, float)
    result = MispricingResult(signals=len(work))
    tape_by_game = {
        int(game): group.sort_values(["created_time", "trade_id"])
        for game, group in trades.groupby("game_pk", sort=False)
    }
    for game_pk, game in work.groupby("game_pk", sort=False):
        tape = tape_by_game.get(int(game_pk))
        if tape is None:
            continue
        times = pd.to_datetime(tape.created_time, utc=True).array.as_unit("ns").asi8
        yes = tape.yes_price_dollars.to_numpy(float)
        no = tape.no_price_dollars.to_numpy(float)
        sizes = tape.count_fp.to_numpy(float)
        taker_sides = tape.taker_outcome_side.astype(str).to_numpy()
        last_fill_ns: int | None = None
        positions = 0
        for row in game.sort_values("signal_time").itertuples(index=False):
            if (
                config.maximum_positions_per_game > 0
                and positions >= config.maximum_positions_per_game
            ):
                break
            yes_ev, no_ev = signal_economics(
                row.fair_probability, row.market_home_price, config.bet_size
            )
            yes_edge = row.fair_probability - row.market_home_price
            no_edge = row.market_home_price - row.fair_probability
            if yes_ev >= no_ev:
                side, expected, edge = "yes", yes_ev, yes_edge
            else:
                side, expected, edge = "no", no_ev, no_edge
            if config.side_filter != "both" and side != config.side_filter:
                continue
            if expected < config.minimum_expected_pnl or edge < config.minimum_probability_edge:
                continue
            result.orders += 1
            signal_ns = pd.Timestamp(row.signal_time).value
            deadline = signal_ns + int(config.maximum_fill_delay_seconds * 1e9)
            if pd.notna(row.next_update_time):
                deadline = min(deadline, pd.Timestamp(row.next_update_time).value)
            indexes = execution_indexes(times, signal_ns, deadline, config)
            if last_fill_ns is not None:
                next_allowed_ns = last_fill_ns + int(
                    config.minimum_seconds_between_entries * 1e9
                )
                indexes = (i for i in indexes if times[i] >= next_allowed_ns)
            for i in indexes:
                if config.require_post_signal_trade and not confirmation_taker_allowed(
                    side, taker_sides[i], config
                ):
                    continue
                price = float(yes[i] if side == "yes" else no[i])
                if not execution_price_allowed(price, config):
                    continue
                requested_contracts = contracts_for_budget(price, config.bet_size)
                contracts = contracts_for_budget(price, config.bet_size, sizes[i])
                if contracts <= 0:
                    continue
                fill_yes = float(yes[i])
                fill_yes_ev, fill_no_ev = signal_economics(
                    row.fair_probability, fill_yes, config.bet_size
                )
                fill_expected = fill_yes_ev if side == "yes" else fill_no_ev
                fill_edge = (
                    row.fair_probability - fill_yes
                    if side == "yes" else fill_yes - row.fair_probability
                )
                if fill_expected < config.minimum_expected_pnl or fill_edge < config.minimum_probability_edge:
                    continue
                # Eligibility is decided for the requested order, as it is
                # live; an IOC may then return a smaller actual fill.
                requested_fee = taker_fee(requested_contracts, price)
                requested_expected = requested_contracts * fill_edge - requested_fee
                if requested_expected < config.minimum_expected_pnl:
                    continue
                fee = taker_fee(contracts, price)
                fill_expected = contracts * fill_edge - fee
                won = (side == "yes" and row.home_win == 1) or (
                    side == "no" and row.home_win == 0
                )
                pnl = (contracts if won else 0.0) - contracts * price - fee
                result.trades += 1
                result.yes_trades += int(side == "yes")
                result.no_trades += int(side == "no")
                result.fees += fee
                result.capital += contracts * price + fee
                result.pnl += pnl
                result.records.append({
                    **row._asdict(), "side": side,
                    "fill_time": pd.Timestamp(times[i], tz="UTC"),
                    "fill_price": price, "contracts": contracts,
                    "entry_fee": fee, "predicted_expected_pnl": fill_expected,
                    "pnl": pnl,
                })
                last_fill_ns = int(times[i])
                positions += 1
                break
    return result


def simulate_away_yes(
    frame: pd.DataFrame,
    probabilities,
    away_trades: pd.DataFrame,
    config: MispricingConfig,
) -> MispricingResult:
    """Route the model's away-team view into the paired away YES contract.

    The forecast remains the home win probability. An away YES contract has
    fair value ``1 - home_probability`` and settles identically to home NO,
    while using the independent paired market's price, size, and trade flow.
    """
    work = frame.copy()
    work["fair_probability"] = np.asarray(probabilities, float)
    result = MispricingResult(signals=len(work))
    tape_by_game = {
        int(game): group.sort_values(["created_time", "trade_id"])
        for game, group in away_trades.groupby("game_pk", sort=False)
    }
    for game_pk, game in work.groupby("game_pk", sort=False):
        tape = tape_by_game.get(int(game_pk))
        if tape is None or tape.empty:
            continue
        times = pd.to_datetime(
            tape.created_time, utc=True
        ).array.as_unit("ns").asi8
        prices = tape.yes_price_dollars.to_numpy(float)
        sizes = tape.count_fp.to_numpy(float)
        taker_sides = tape.taker_outcome_side.astype(str).to_numpy()
        last_fill_ns: int | None = None
        positions = 0
        best_away_fair = float("-inf")
        best_expected_return = float("-inf")
        for row in game.sort_values("signal_time").itertuples(index=False):
            if (
                config.maximum_positions_per_game > 0
                and positions >= config.maximum_positions_per_game
            ):
                break
            model_side, _, _, model_eligible = model_signal(
                float(row.fair_probability),
                float(row.market_home_price),
                config,
            )
            if not model_eligible or model_side != "no":
                continue
            away_fair = 1.0 - float(row.fair_probability)
            signal_ns = pd.Timestamp(row.signal_time).value
            deadline = signal_ns + int(config.maximum_fill_delay_seconds * 1e9)
            if pd.notna(row.next_update_time):
                deadline = min(deadline, pd.Timestamp(row.next_update_time).value)
            indexes = execution_indexes(times, signal_ns, deadline, config)
            indexes = list(indexes)
            start = indexes[0] if indexes else 0
            if not indexes or (config.require_post_signal_trade and start == 0):
                continue
            observed_index = start - 1 if config.require_post_signal_trade else start
            observed_price = float(prices[observed_index])
            observed_contracts = config.bet_size / observed_price
            observed_fee = taker_fee(observed_contracts, observed_price)
            observed_ev = (
                observed_contracts * (away_fair - observed_price)
                - observed_fee
            )
            if (
                away_fair - observed_price < config.minimum_probability_edge
                or observed_ev < config.minimum_expected_pnl
            ):
                continue
            result.orders += 1
            if last_fill_ns is not None:
                next_allowed_ns = last_fill_ns + int(
                    config.minimum_seconds_between_entries * 1e9
                )
                indexes = [i for i in indexes if times[i] >= next_allowed_ns]
            for index in indexes:
                if config.require_post_signal_trade and not confirmation_taker_allowed(
                    "yes", taker_sides[index], config
                ):
                    continue
                price = float(prices[index])
                if not execution_price_allowed(price, config):
                    continue
                requested_contracts = contracts_for_budget(price, config.bet_size)
                contracts = contracts_for_budget(
                    price, config.bet_size, sizes[index]
                )
                if contracts <= 0:
                    continue
                fee = taker_fee(contracts, price)
                expected = contracts * (away_fair - price) - fee
                edge = away_fair - price
                requested_fee = taker_fee(requested_contracts, price)
                requested_expected = (
                    requested_contracts * edge - requested_fee
                )
                expected_return = expected / (contracts * price + fee)
                if (
                    requested_expected < config.minimum_expected_pnl
                    or edge < config.minimum_probability_edge
                ):
                    continue
                if config.conditional_stacking and positions > 0 and not (
                    away_fair > best_away_fair
                    and expected_return > best_expected_return
                ):
                    continue
                won = int(row.home_win) == 0
                pnl = (contracts if won else 0.0) - contracts * price - fee
                result.trades += 1
                result.yes_trades += 1
                result.fees += fee
                result.capital += contracts * price + fee
                result.pnl += pnl
                result.records.append({
                    **row._asdict(),
                    "model_side": model_side,
                    "side": "yes",
                    "execution_contract": "away_yes",
                    "fill_time": pd.Timestamp(times[index], tz="UTC"),
                    "fill_price": price,
                    "contracts": contracts,
                    "entry_fee": fee,
                    "predicted_expected_pnl": expected,
                    "pnl": pnl,
                })
                last_fill_ns = int(times[index])
                positions += 1
                best_away_fair = max(best_away_fair, away_fair)
                best_expected_return = max(
                    best_expected_return, expected_return
                )
                break
    return result


def simulate_paired_both(
    frame: pd.DataFrame,
    probabilities,
    home_trades: pd.DataFrame,
    away_trades: pd.DataFrame,
    config: MispricingConfig,
) -> MispricingResult:
    """Buy home YES for home signals and paired away YES for away signals."""
    work = frame.copy()
    work["fair_probability"] = np.asarray(probabilities, float)
    result = MispricingResult(signals=len(work))
    home_by_game = {
        int(game): group.sort_values(["created_time", "trade_id"])
        for game, group in home_trades.groupby("game_pk", sort=False)
    }
    away_by_game = {
        int(game): group.sort_values(["created_time", "trade_id"])
        for game, group in away_trades.groupby("game_pk", sort=False)
    }
    for game_pk, game in work.groupby("game_pk", sort=False):
        tapes = {
            "yes": home_by_game.get(int(game_pk)),
            "no": away_by_game.get(int(game_pk)),
        }
        if tapes["yes"] is None and tapes["no"] is None:
            continue
        last_fill_ns: int | None = None
        positions = 0
        active_positions: list[dict] = []
        best_probability = {"yes": float("-inf"), "no": float("-inf")}
        best_expected_return = {"yes": float("-inf"), "no": float("-inf")}
        for row in game.sort_values("signal_time").itertuples(index=False):
            if not entry_state_allowed(row, config):
                continue
            model_side, _, _, eligible = model_signal(
                float(row.fair_probability),
                float(row.market_home_price),
                config,
            )
            if not eligible:
                continue
            tape = tapes[model_side]
            if tape is None or tape.empty:
                continue
            times = pd.to_datetime(
                tape.created_time, utc=True
            ).array.as_unit("ns").asi8
            prices = tape.yes_price_dollars.to_numpy(float)
            sizes = tape.count_fp.to_numpy(float)
            taker_sides = tape.taker_outcome_side.astype(str).to_numpy()
            signal_ns = pd.Timestamp(row.signal_time).value
            deadline = signal_ns + int(config.maximum_fill_delay_seconds * 1e9)
            if pd.notna(row.next_update_time):
                deadline = min(deadline, pd.Timestamp(row.next_update_time).value)
            indexes = execution_indexes(times, signal_ns, deadline, config)
            if last_fill_ns is not None:
                next_allowed = last_fill_ns + int(
                    config.minimum_seconds_between_entries * 1e9
                )
                indexes = (i for i in indexes if times[i] >= next_allowed)
            execution_probability = (
                float(row.fair_probability)
                if model_side == "yes"
                else 1.0 - float(row.fair_probability)
            )
            result.orders += 1
            for index in indexes:
                if config.require_post_signal_trade and not confirmation_taker_allowed(
                    "yes", taker_sides[index], config
                ):
                    continue
                price = float(prices[index])
                if not execution_price_allowed(price, config):
                    continue
                requested_contracts = contracts_for_budget(price, config.bet_size)
                contracts = contracts_for_budget(
                    price, config.bet_size, sizes[index]
                )
                if contracts <= 0:
                    continue
                fee = taker_fee(contracts, price)
                edge = execution_probability - price
                expected = contracts * edge - fee
                requested_fee = taker_fee(requested_contracts, price)
                requested_expected = (
                    requested_contracts * edge - requested_fee
                )
                expected_return = expected / (contracts * price + fee)
                if (
                    requested_expected < config.minimum_expected_pnl
                    or edge < config.minimum_probability_edge
                ):
                    continue
                conflicting_side = "no" if model_side == "yes" else "yes"
                conflicts = [
                    item for item in active_positions
                    if item["model_side"] == conflicting_side
                ]
                if conflicts:
                    conflict_tape = tapes[conflicting_side]
                    if conflict_tape is None or conflict_tape.empty:
                        continue
                    conflict_times = pd.to_datetime(
                        conflict_tape.created_time, utc=True
                    ).array.as_unit("ns").asi8
                    exit_index = int(np.searchsorted(
                        conflict_times, times[index], side="right"
                    ) - 1)
                    if exit_index < 0:
                        continue
                    exit_price = float(
                        conflict_tape.iloc[exit_index].yes_price_dollars
                    )
                    economics = reversal_economics(
                        [item["position"] for item in conflicts],
                        exit_price, expected,
                    )
                    if not economics["allowed"]:
                        continue
                    for conflict in conflicts:
                        position = conflict["position"]
                        exit_fee = taker_fee(position.contracts, exit_price)
                        realized = (
                            position.contracts * exit_price - exit_fee
                            - position.contracts * position.entry_price
                            - position.entry_fee
                        )
                        old_pnl = float(conflict["record"]["pnl"])
                        result.pnl += realized - old_pnl
                        result.fees += exit_fee
                        conflict["record"].update({
                            "pnl": realized,
                            "exit_reason": "profitable_reversal",
                            "exit_time": pd.Timestamp(
                                conflict_times[exit_index], tz="UTC"
                            ),
                            "exit_price": exit_price,
                            "exit_fee": exit_fee,
                        })
                        active_positions.remove(conflict)
                        positions -= 1
                if (
                    config.maximum_positions_per_game > 0
                    and positions >= config.maximum_positions_per_game
                ):
                    continue
                if config.conditional_stacking and not (
                    execution_probability > best_probability[model_side]
                    and expected_return > best_expected_return[model_side]
                ):
                    continue
                won = (
                    (model_side == "yes" and int(row.home_win) == 1)
                    or (model_side == "no" and int(row.home_win) == 0)
                )
                principal = contracts * price
                pnl = (contracts if won else 0.0) - principal - fee
                result.trades += 1
                result.yes_trades += int(model_side == "yes")
                result.no_trades += int(model_side == "no")
                result.fees += fee
                result.capital += principal + fee
                result.pnl += pnl
                record = {
                    **row._asdict(), "model_side": model_side,
                    "side": "yes" if model_side == "yes" else "away_yes",
                    "execution_contract": (
                        "home_yes" if model_side == "yes" else "away_yes"
                    ),
                    "fill_time": pd.Timestamp(times[index], tz="UTC"),
                    "fill_price": price, "contracts": contracts,
                    "entry_fee": fee, "predicted_expected_pnl": expected,
                    "pnl": pnl,
                }
                result.records.append(record)
                active_positions.append({
                    "model_side": model_side,
                    "position": SimpleNamespace(
                        contracts=contracts, entry_price=price,
                        entry_fee=fee,
                    ),
                    "record": record,
                })
                last_fill_ns = int(times[index])
                positions += 1
                best_probability[model_side] = execution_probability
                best_expected_return[model_side] = expected_return
                break
    return result
