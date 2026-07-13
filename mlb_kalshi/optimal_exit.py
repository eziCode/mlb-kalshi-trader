"""Fitted continuation-value policy for deciding whether to sell a position."""

from __future__ import annotations

from dataclasses import dataclass, field

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd

from mlb_kalshi.hybrid import anchored_event_target
from mlb_kalshi.strategy import taker_fee
from mlb_kalshi.trade_tape import TapeTradeRecord


EXIT_FEATURES = (
    "position_is_yes",
    "residual",
    "residual_change_1",
    "residual_change_5",
    "position_return",
    "fair_position",
    "target_position",
    "market_position_price",
    "price_momentum_1",
    "price_momentum_5",
    "price_volatility_30",
    "proxy_spread",
    "snapshot_gap_seconds",
    "seconds_since_entry",
    "inning",
    "inning_topbot",
    "outs",
    "score_diff",
    "balls",
    "strikes",
    "runner_on_first",
    "runner_on_second",
    "runner_on_third",
)


@dataclass(frozen=True)
class ExitPolicyConfig:
    enabled: bool = False
    continuation_margin: float = 0.02
    confirmation_seconds: float = 2.0


@dataclass
class ExitPolicyRecord:
    trajectory_id: int
    game_pk: int
    side: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp | None
    exit_reason: str
    pnl: float
    fees: float


@dataclass
class ExitPolicyResult:
    trades: int = 0
    model_exits: int = 0
    settlements: int = 0
    fees: float = 0.0
    capital: float = 0.0
    pnl: float = 0.0
    records: list[ExitPolicyRecord] = field(default_factory=list)

    @property
    def roi(self) -> float:
        return self.pnl / self.capital if self.capital else 0.0


def exit_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(EXIT_FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing exit-policy features: {sorted(missing)}")
    result = frame.loc[:, EXIT_FEATURES].apply(pd.to_numeric, errors="coerce")
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)


def make_live_exit_feature_row(
    *,
    side: str,
    entry_price: float,
    entry_fee: float,
    contracts: float,
    entry_time: pd.Timestamp,
    now: pd.Timestamp,
    fair_probability: float,
    target: float,
    midpoint: float,
    bid: float,
    ask: float,
    state: dict,
    history: list[dict],
) -> tuple[pd.DataFrame, dict]:
    """Build the live equivalent of one trade-tape exit snapshot."""
    side_yes = side == "yes"
    market_position_price = midpoint if side_yes else 1.0 - midpoint
    fair_position = fair_probability if side_yes else 1.0 - fair_probability
    target_position = target if side_yes else 1.0 - target
    liquidation_price = bid if side_yes else 1.0 - ask
    exit_fee = taker_fee(contracts, liquidation_price)
    exit_value = liquidation_price - exit_fee / contracts
    residual = target_position - market_position_price
    previous = history[-1] if history else None
    fifth = history[-5] if len(history) >= 5 else None
    prices = [point["market_position_price"] for point in history[-29:]]
    prices.append(market_position_price)
    row = {
        "position_is_yes": float(side_yes),
        "residual": residual,
        "residual_change_1": (
            residual - previous["residual"] if previous else 0.0
        ),
        "residual_change_5": residual - fifth["residual"] if fifth else 0.0,
        "position_return": (
            exit_value - entry_price - entry_fee / contracts
        ),
        "fair_position": fair_position,
        "target_position": target_position,
        "market_position_price": market_position_price,
        "price_momentum_1": (
            market_position_price - previous["market_position_price"]
            if previous else 0.0
        ),
        "price_momentum_5": (
            market_position_price - fifth["market_position_price"]
            if fifth else 0.0
        ),
        "price_volatility_30": float(np.std(prices, ddof=1)) if len(prices) > 1 else 0.0,
        "proxy_spread": ask - bid,
        "snapshot_gap_seconds": (
            (now - previous["snapshot_time"]).total_seconds()
            if previous else 0.0
        ),
        "seconds_since_entry": (now - entry_time).total_seconds(),
        "inning": state["inning"],
        "inning_topbot": state["inning_topbot"],
        "outs": state["outs_when_up"],
        "score_diff": state["score_diff"],
        "balls": state["balls"],
        "strikes": state["strikes"],
        "runner_on_first": state["runner_on_first"],
        "runner_on_second": state["runner_on_second"],
        "runner_on_third": state["runner_on_third"],
    }
    point = {
        "snapshot_time": now,
        "residual": residual,
        "market_position_price": market_position_price,
    }
    return exit_feature_frame(pd.DataFrame([row])), point


def _last_rows_by_second(trades: pd.DataFrame) -> pd.DataFrame:
    tape = trades.sort_values(["created_time", "trade_id"]).copy()
    tape["trade_second"] = tape["created_time"].dt.floor("s")
    overall = tape.groupby("trade_second", as_index=False).agg(
        snapshot_time=("created_time", "max"),
        yes_price=("yes_price_dollars", "last"),
        second_high=("yes_price_dollars", "max"),
        second_low=("yes_price_dollars", "min"),
    )
    sides = []
    for side in ("yes", "no"):
        selected = (
            tape[tape["taker_outcome_side"] == side]
            .groupby("trade_second", as_index=False)
            .tail(1)[[
                "trade_second", "yes_price_dollars", "no_price_dollars",
                "count_fp",
            ]]
            .rename(columns={
                "yes_price_dollars": f"{side}_taker_yes_price",
                "no_price_dollars": f"{side}_taker_no_price",
                "count_fp": f"{side}_taker_size",
            })
        )
        sides.append(selected)
    return overall.merge(sides[0], on="trade_second", how="left").merge(
        sides[1], on="trade_second", how="left"
    ).sort_values("snapshot_time")


def _attach_game_state(
    seconds: pd.DataFrame,
    updates: pd.DataFrame,
) -> pd.DataFrame:
    state_columns = {
        "fair_after": "current_fair",
        "inning_after": "inning",
        "inning_topbot_after": "inning_topbot",
        "outs_when_up_after": "outs",
        "score_diff_after": "score_diff",
        "balls_after": "balls",
        "strikes_after": "strikes",
        "runner_on_first_after": "runner_on_first",
        "runner_on_second_after": "runner_on_second",
        "runner_on_third_after": "runner_on_third",
    }
    right = updates[["pitch_end_time", *state_columns]].rename(
        columns=state_columns
    ).sort_values("pitch_end_time")
    result = pd.merge_asof(
        seconds.sort_values("snapshot_time"),
        right,
        left_on="snapshot_time",
        right_on="pitch_end_time",
        direction="backward",
    )
    if not updates.empty:
        first = updates.iloc[0]
        defaults = {
            "current_fair": first["fair_before"],
            "inning": first["inning_after"],
            "inning_topbot": first["inning_topbot_after"],
            "outs": first["outs_when_up_after"],
            "score_diff": first["score_diff_after"],
            "balls": first["balls_after"],
            "strikes": first["strikes_after"],
            "runner_on_first": first["runner_on_first_after"],
            "runner_on_second": first["runner_on_second_after"],
            "runner_on_third": first["runner_on_third_after"],
        }
        for column, value in defaults.items():
            result[column] = result[column].fillna(value)
    return result


def build_exit_trajectories(
    trades: pd.DataFrame,
    updates: pd.DataFrame,
    entries: list[TapeTradeRecord],
) -> pd.DataFrame:
    """Create reproducible one-active-second decision paths after each entry."""
    entries_by_game: dict[int, list[tuple[int, TapeTradeRecord]]] = {}
    for trajectory_id, entry in enumerate(entries):
        entries_by_game.setdefault(int(entry.game_pk), []).append(
            (trajectory_id, entry)
        )
    updates_by_game = {
        int(game_pk): game.sort_values("pitch_end_time")
        for game_pk, game in updates.groupby("game_pk", sort=False)
    }
    paths = []
    for game_pk, game_trades in trades.groupby("game_pk", sort=False):
        game_pk = int(game_pk)
        game_entries = entries_by_game.get(game_pk)
        game_updates = updates_by_game.get(game_pk)
        if not game_entries or game_updates is None:
            continue
        seconds = _attach_game_state(
            _last_rows_by_second(game_trades), game_updates
        )
        home_win = int(game_trades["home_win"].iloc[-1])
        for trajectory_id, entry in game_entries:
            path = seconds[seconds["snapshot_time"] > entry.entry_time].copy()
            if path.empty:
                continue
            side_yes = entry.side == "yes"
            path["trajectory_id"] = trajectory_id
            path["game_pk"] = game_pk
            path["game_date"] = game_trades["game_date"].iloc[-1]
            path["side"] = entry.side
            path["event_type"] = entry.event_type
            path["trigger_at_bat"] = entry.trigger_at_bat
            path["trigger_pitch"] = entry.trigger_pitch
            path["trigger_event_time"] = entry.trigger_event_time
            path["entry_time"] = entry.entry_time
            path["entry_price"] = entry.entry_price
            path["entry_fee"] = entry.fees if entry.exit_reason == "settlement" else np.nan
            # The record stores total fees after a reversion. Reconstruct the
            # entry fee directly so every trajectory has the same semantics.
            path["entry_fee"] = taker_fee(entry.contracts, entry.entry_price)
            path["contracts"] = entry.contracts
            path["home_win"] = home_win
            path["terminal_value"] = float(
                (side_yes and home_win == 1) or (not side_yes and home_win == 0)
            )
            path["position_is_yes"] = float(side_yes)
            path["dynamic_target"] = anchored_event_target(
                entry.anchor_target,
                entry.anchor_fair,
                path["current_fair"],
            )
            path["fair_position"] = (
                path["current_fair"] if side_yes else 1.0 - path["current_fair"]
            )
            path["target_position"] = (
                path["dynamic_target"] if side_yes else 1.0 - path["dynamic_target"]
            )
            path["market_position_price"] = (
                path["yes_price"] if side_yes else 1.0 - path["yes_price"]
            )
            path["residual"] = (
                path["target_position"] - path["market_position_price"]
            )
            if side_yes:
                path["exit_price"] = path["no_taker_yes_price"]
                compatible_size = path["no_taker_size"]
            else:
                path["exit_price"] = path["yes_taker_no_price"]
                compatible_size = path["yes_taker_size"]
            path.loc[compatible_size < entry.contracts, "exit_price"] = np.nan
            exit_fees = [
                taker_fee(entry.contracts, price) if pd.notna(price) else np.nan
                for price in path["exit_price"]
            ]
            path["exit_fee"] = exit_fees
            path["exit_value"] = (
                path["exit_price"] - path["exit_fee"] / entry.contracts
            )
            mark_value = path["exit_value"].fillna(path["market_position_price"])
            path["position_return"] = (
                mark_value - entry.entry_price
                - path["entry_fee"] / entry.contracts
            )
            yes_taker = path["yes_taker_yes_price"]
            no_taker = path["no_taker_yes_price"]
            path["proxy_spread"] = (yes_taker - no_taker).abs()
            path["snapshot_gap_seconds"] = (
                path["snapshot_time"].diff().dt.total_seconds().fillna(0.0)
            )
            path["seconds_since_entry"] = (
                path["snapshot_time"] - entry.entry_time
            ).dt.total_seconds()
            path["residual_change_1"] = path["residual"].diff(1)
            path["residual_change_5"] = path["residual"].diff(5)
            path["price_momentum_1"] = path["market_position_price"].diff(1)
            path["price_momentum_5"] = path["market_position_price"].diff(5)
            path["price_volatility_30"] = (
                path["market_position_price"].rolling(30, min_periods=2).std()
            )
            paths.append(path[[
                "trajectory_id", "game_pk", "game_date", "side", "event_type",
                "trigger_at_bat", "trigger_pitch", "trigger_event_time",
                "entry_time", "snapshot_time", "entry_price", "entry_fee",
                "contracts", "home_win", "terminal_value", "exit_price",
                "exit_fee", "exit_value", *EXIT_FEATURES,
            ]])
    if not paths:
        return pd.DataFrame()
    return pd.concat(paths, ignore_index=True).sort_values(
        ["trajectory_id", "snapshot_time"]
    ).reset_index(drop=True)


def train_continuation_model(
    trajectories: pd.DataFrame,
    policy_iterations: int = 5,
) -> CatBoostRegressor:
    """Fitted policy iteration using realized next-stop continuation payoffs."""
    frame = trajectories.sort_values(
        ["trajectory_id", "snapshot_time"]
    ).reset_index(drop=True)
    features = exit_feature_frame(frame)
    targets = frame["terminal_value"].to_numpy(dtype=float)
    trajectory_size = frame.groupby("trajectory_id")["trajectory_id"].transform(
        "size"
    ).to_numpy(dtype=float)
    sample_weight = (1.0 / trajectory_size)
    sample_weight /= sample_weight.mean()
    model: CatBoostRegressor | None = None
    for iteration in range(policy_iterations):
        model = CatBoostRegressor(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            loss_function="Quantile:alpha=0.25",
            l2_leaf_reg=10.0,
            random_seed=42 + iteration,
            allow_writing_files=False,
            verbose=False,
        )
        model.fit(features, targets, sample_weight=sample_weight)
        continuation = np.clip(model.predict(features), 0.0, 1.0)
        exits = (
            frame["exit_value"].notna().to_numpy()
            & (frame["exit_value"].fillna(-np.inf).to_numpy() >= continuation)
        )
        new_targets = np.empty(len(frame), dtype=float)
        for _, indices in frame.groupby("trajectory_id", sort=False).groups.items():
            positions = np.asarray(indices, dtype=int)
            future_payoff = float(frame.loc[positions[-1], "terminal_value"])
            for position in positions[::-1]:
                new_targets[position] = future_payoff
                if exits[position]:
                    future_payoff = float(frame.loc[position, "exit_value"])
        targets = new_targets
    assert model is not None
    return model


def evaluate_exit_policy(
    trajectories: pd.DataFrame,
    model: CatBoostRegressor,
    config: ExitPolicyConfig,
) -> ExitPolicyResult:
    frame = trajectories.sort_values(
        ["game_pk", "entry_time", "trajectory_id", "snapshot_time"]
    ).copy()
    frame["continuation_value"] = np.clip(
        model.predict(exit_feature_frame(frame)), 0.0, 1.0
    )
    result = ExitPolicyResult()
    occupied_until: dict[int, pd.Timestamp | None] = {}
    confirmation = pd.Timedelta(seconds=config.confirmation_seconds)

    for trajectory_id, path in frame.groupby("trajectory_id", sort=False):
        path = path.sort_values("snapshot_time")
        first = path.iloc[0]
        game_pk = int(first["game_pk"])
        entry_time = pd.Timestamp(first["entry_time"])
        prior_exit = occupied_until.get(game_pk)
        if prior_exit is None and game_pk in occupied_until:
            continue
        if prior_exit is not None and entry_time <= prior_exit:
            continue

        watch_started: pd.Timestamp | None = None
        pending = False
        exit_row = None
        for row in path.itertuples():
            has_exit_quote = pd.notna(row.exit_value)
            advantage = (
                float(row.exit_value) - float(row.continuation_value)
                if has_exit_quote
                else -np.inf
            )
            should_exit = advantage >= config.continuation_margin
            now = pd.Timestamp(row.snapshot_time)
            if pending:
                if not has_exit_quote:
                    continue
                if should_exit:
                    exit_row = row
                    break
                pending = False
                watch_started = None
            if not has_exit_quote:
                continue
            if should_exit:
                if watch_started is None:
                    watch_started = now
                elif now - watch_started >= confirmation:
                    pending = True
            else:
                watch_started = None

        contracts = float(first["contracts"])
        entry_price = float(first["entry_price"])
        entry_fee = float(first["entry_fee"])
        result.trades += 1
        result.capital += contracts * entry_price + entry_fee
        result.fees += entry_fee
        if exit_row is not None:
            exit_price = float(exit_row.exit_price)
            exit_fee = float(exit_row.exit_fee)
            pnl = (
                contracts * exit_price - exit_fee
                - contracts * entry_price - entry_fee
            )
            exit_time = pd.Timestamp(exit_row.snapshot_time)
            result.model_exits += 1
            result.fees += exit_fee
            occupied_until[game_pk] = exit_time
            reason = "model_exit"
        else:
            payout = contracts * float(first["terminal_value"])
            pnl = payout - contracts * entry_price - entry_fee
            exit_time = None
            result.settlements += 1
            occupied_until[game_pk] = None
            reason = "settlement"
        result.pnl += pnl
        result.records.append(ExitPolicyRecord(
            trajectory_id=int(trajectory_id),
            game_pk=game_pk,
            side=str(first["side"]),
            entry_time=entry_time,
            exit_time=exit_time,
            exit_reason=reason,
            pnl=pnl,
            fees=entry_fee + (float(exit_row.exit_fee) if exit_row else 0.0),
        ))
    return result
