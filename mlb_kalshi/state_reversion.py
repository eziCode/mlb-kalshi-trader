"""State-only transition diagnostics and executable-reversion features."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from mlb_kalshi.strategy import taker_fee


# Compact feature contract for the event-agnostic overshoot classifier.  The
# older STATE_ONLY_REVERSION_FEATURES contract below is retained so historical
# study artifacts remain readable.
OVERSHOOT_REVERSION_FEATURES = (
    "signed_logit_residual",
    "absolute_logit_residual",
    "fair_logit_move",
    "market_logit_move",
    "entry_price",
    "target_contract_price",
    "target_reversion_pnl",
    "failure_pnl",
    "breakeven_reversion_probability",
    "entry_latency_seconds",
    "anchor_age_seconds",
    "inning_after",
    "inning_topbot_after",
    "outs_when_up_after",
    "score_diff_after",
    "balls_after",
    "strikes_after",
    "runner_on_first_after",
    "runner_on_second_after",
    "runner_on_third_after",
    "pre_trade_count_2s",
    "pre_volume_2s",
    "pre_flow_imbalance_2s",
    "pre_price_volatility_2s",
    "entry_side_yes",
)


AFTER_STATE_COLUMNS = (
    "inning_after",
    "inning_topbot_after",
    "outs_when_up_after",
    "score_diff_after",
    "balls_after",
    "strikes_after",
    "runner_on_first_after",
    "runner_on_second_after",
    "runner_on_third_after",
)


def _delta_column(after_column: str) -> str:
    base = after_column.removesuffix("_after")
    return "delta_outs" if base == "outs_when_up" else f"delta_{base}"

STATE_ONLY_REVERSION_FEATURES = (
    "fair_before",
    "fair_after",
    "fair_home_move",
    "fair_batting_move",
    "target_home_price",
    "entry_home_price",
    "probability_residual",
    "entry_contract_price",
    "target_contract_price",
    "target_profit_per_contract",
    "maximum_loss_per_contract",
    "breakeven_reversion_probability",
    "event_to_entry_seconds",
    "inning_after",
    "inning_topbot_after",
    "outs_when_up_after",
    "score_diff_after",
    "balls_after",
    "strikes_after",
    "runner_on_first_after",
    "runner_on_second_after",
    "runner_on_third_after",
    "delta_inning",
    "delta_outs",
    "delta_score_diff",
    "delta_balls",
    "delta_strikes",
    "delta_runner_on_first",
    "delta_runner_on_second",
    "delta_runner_on_third",
    "pre_trade_count_2s",
    "pre_volume_2s",
    "pre_flow_imbalance_2s",
    "pre_price_volatility_2s",
    "entry_side_yes",
)


@dataclass(frozen=True)
class StateReversionConfig:
    enabled: bool = False
    minimum_probability: float = 0.50
    minimum_break_even_margin: float = 0.0


def accepted_state_reversions(
    frame: pd.DataFrame,
    predictions,
    config: StateReversionConfig,
) -> pd.Series:
    probability = np.asarray(predictions, dtype=float)
    required = (
        frame["breakeven_reversion_probability"].to_numpy(dtype=float)
        + config.minimum_break_even_margin
    )
    return pd.Series(
        (probability >= config.minimum_probability) & (probability >= required),
        index=frame.index,
    )


def add_before_and_delta_state(updates: pd.DataFrame) -> pd.DataFrame:
    result = updates.copy()
    result["pitch_end_time"] = pd.to_datetime(
        result["pitch_end_time"], utc=True
    )
    result = result.sort_values([
        "game_pk", "pitch_end_time", "at_bat_number", "pitch_number",
    ])
    grouped = result.groupby("game_pk", sort=False)
    for after in AFTER_STATE_COLUMNS:
        if after not in result:
            raise ValueError(f"State updates are missing {after}")
        base = after.removesuffix("_after")
        before = f"{base}_before"
        result[before] = grouped[after].shift(1)
        result[_delta_column(after)] = result[after] - result[before]
    return result


def transition_direction(frame: pd.DataFrame) -> pd.Series:
    """Heuristic expected batting-team direction using state changes only."""
    batting_home = frame["inning_topbot_after"].eq(1)
    batting_score_delta = np.where(
        batting_home, frame["delta_score_diff"], -frame["delta_score_diff"]
    )
    runner_before = (
        frame["runner_on_first_before"]
        + 2 * frame["runner_on_second_before"]
        + 3 * frame["runner_on_third_before"]
    )
    runner_after = (
        frame["runner_on_first_after"]
        + 2 * frame["runner_on_second_after"]
        + 3 * frame["runner_on_third_after"]
    )
    expected = np.zeros(len(frame), dtype=float)
    same_half = frame["delta_inning"].eq(0) & frame[
        "inning_topbot_after"
    ].eq(frame["inning_topbot_before"])
    expected = np.where(batting_score_delta > 0, 1.0, expected)
    expected = np.where(batting_score_delta < 0, -1.0, expected)
    undecided = expected == 0
    expected = np.where(
        undecided & same_half & frame["delta_outs"].gt(0), -1.0, expected
    )
    undecided = expected == 0
    expected = np.where(
        undecided & same_half & runner_after.gt(runner_before), 1.0, expected
    )
    expected = np.where(
        (expected == 0) & same_half & runner_after.lt(runner_before),
        -1.0,
        expected,
    )
    return pd.Series(expected, index=frame.index, name="expected_direction")


def transition_diagnostics(updates: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frame = add_before_and_delta_state(updates).dropna(subset=[
        "fair_before", "fair_after", "inning_topbot_before",
    ]).copy()
    frame["fair_home_move"] = frame["fair_after"] - frame["fair_before"]
    frame["fair_batting_move"] = np.where(
        frame["inning_topbot_after"].eq(1),
        frame["fair_home_move"],
        -frame["fair_home_move"],
    )
    frame["expected_direction"] = transition_direction(frame)
    comparable = frame["expected_direction"].ne(0)
    frame["direction_violation"] = comparable & (
        frame["fair_batting_move"] * frame["expected_direction"] < 0
    )
    frame["count_reset"] = (
        frame["balls_after"].eq(0)
        & frame["strikes_after"].eq(0)
        & (
            frame["balls_before"].gt(0)
            | frame["strikes_before"].gt(0)
        )
    )
    signature = [
        "inning_topbot_after", "delta_inning", "delta_outs",
        "delta_score_diff", "delta_runner_on_first",
        "delta_runner_on_second", "delta_runner_on_third", "count_reset",
    ]
    grouped = frame.groupby(signature, dropna=False)["fair_batting_move"]
    report = grouped.agg(
        transitions="size", mean="mean", std="std", median="median",
        minimum="min", maximum="max",
    ).reset_index()
    report["positive_rate"] = grouped.apply(lambda values: values.gt(0).mean()).values
    report = report.sort_values("transitions", ascending=False)
    comparable_count = int(comparable.sum())
    summary = {
        "transitions": int(len(frame)),
        "games": int(frame["game_pk"].nunique()),
        "comparable_direction_transitions": comparable_count,
        "direction_violations": int(frame["direction_violation"].sum()),
        "direction_violation_rate": (
            float(frame.loc[comparable, "direction_violation"].mean())
            if comparable_count else 0.0
        ),
        "count_reset_transitions": int(frame["count_reset"].sum()),
        "median_absolute_fair_move": float(frame["fair_home_move"].abs().median()),
        "p95_absolute_fair_move": float(frame["fair_home_move"].abs().quantile(0.95)),
    }
    return report, summary


def _attach_pre_trade_features(
    records: pd.DataFrame,
    trades: pd.DataFrame,
    window_seconds: float = 2.0,
) -> pd.DataFrame:
    result = records.copy()
    for column in (
        "pre_trade_count_2s", "pre_volume_2s", "pre_flow_imbalance_2s",
        "pre_price_volatility_2s",
    ):
        result[column] = 0.0
    window_ns = int(window_seconds * 1e9)
    for game_pk, indexes in result.groupby("game_pk").groups.items():
        tape = trades[trades["game_pk"].eq(game_pk)].sort_values([
            "created_time", "trade_id",
        ])
        if tape.empty:
            continue
        times = pd.to_datetime(tape["created_time"], utc=True).array.as_unit("ns").asi8
        prices = tape["yes_price_dollars"].to_numpy(dtype=float)
        sizes = tape["count_fp"].to_numpy(dtype=float)
        sides = tape["taker_outcome_side"].astype(str).to_numpy()
        for index in indexes:
            entry_ns = pd.Timestamp(result.at[index, "entry_time"]).value
            start = int(np.searchsorted(times, entry_ns - window_ns, side="left"))
            stop = int(np.searchsorted(times, entry_ns, side="left"))
            if stop <= start:
                continue
            selected_sizes = sizes[start:stop]
            volume = float(selected_sizes.sum())
            signed = np.where(sides[start:stop] == "yes", 1.0, -1.0)
            result.at[index, "pre_trade_count_2s"] = stop - start
            result.at[index, "pre_volume_2s"] = volume
            result.at[index, "pre_flow_imbalance_2s"] = (
                float((selected_sizes * signed).sum() / volume) if volume else 0.0
            )
            result.at[index, "pre_price_volatility_2s"] = float(
                np.std(prices[start:stop])
            )
    return result


def build_state_reversion_examples(
    records: pd.DataFrame,
    updates: pd.DataFrame,
    trades: pd.DataFrame,
) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame()
    prepared = add_before_and_delta_state(updates)
    join_columns = [
        "game_pk", "at_bat_number", "pitch_number", "game_date",
        "fair_before", "fair_after", *AFTER_STATE_COLUMNS,
        *[
            f"{column.removesuffix('_after')}_before"
            for column in AFTER_STATE_COLUMNS
        ],
        *[_delta_column(column) for column in AFTER_STATE_COLUMNS],
    ]
    examples = records.merge(
        prepared[join_columns],
        left_on=["game_pk", "trigger_at_bat", "trigger_pitch"],
        right_on=["game_pk", "at_bat_number", "pitch_number"],
        how="inner",
        validate="many_to_one",
    )
    before_and_delta = [
        column for column in examples.columns
        if column.endswith("_before") or column.startswith("delta_")
    ]
    examples = examples.dropna(subset=before_and_delta).copy()
    examples["entry_time"] = pd.to_datetime(examples["entry_time"], utc=True)
    examples["trigger_event_time"] = pd.to_datetime(
        examples["trigger_event_time"], utc=True
    )
    examples["fair_home_move"] = examples["fair_after"] - examples["fair_before"]
    examples["fair_batting_move"] = np.where(
        examples["inning_topbot_after"].eq(1),
        examples["fair_home_move"], -examples["fair_home_move"],
    )
    examples["target_home_price"] = examples["anchor_target"]
    examples["entry_home_price"] = np.where(
        examples["side"].eq("yes"),
        examples["entry_price"], 1.0 - examples["entry_price"],
    )
    examples["probability_residual"] = (
        examples["target_home_price"] - examples["entry_home_price"]
    ).abs()
    examples["entry_contract_price"] = examples["entry_price"]
    examples["target_contract_price"] = np.where(
        examples["side"].eq("yes"),
        examples["target_home_price"], 1.0 - examples["target_home_price"],
    )
    examples["entry_side_yes"] = examples["side"].eq("yes").astype(float)
    examples["event_to_entry_seconds"] = (
        examples["entry_time"] - examples["trigger_event_time"]
    ).dt.total_seconds()
    examples["entry_fee"] = [
        taker_fee(contracts, price)
        for contracts, price in zip(examples["contracts"], examples["entry_price"])
    ]
    examples["target_exit_fee"] = [
        taker_fee(contracts, price)
        for contracts, price in zip(
            examples["contracts"], examples["target_contract_price"]
        )
    ]
    examples["target_reversion_pnl"] = (
        examples["contracts"]
        * (examples["target_contract_price"] - examples["entry_price"])
        - examples["entry_fee"]
        - examples["target_exit_fee"]
    )
    examples["maximum_settlement_loss"] = (
        examples["contracts"] * examples["entry_price"] + examples["entry_fee"]
    )
    examples["target_profit_per_contract"] = (
        examples["target_reversion_pnl"] / examples["contracts"]
    )
    examples["maximum_loss_per_contract"] = (
        examples["maximum_settlement_loss"] / examples["contracts"]
    )
    gain = examples["target_reversion_pnl"].clip(lower=0)
    examples["breakeven_reversion_probability"] = (
        examples["maximum_settlement_loss"]
        / (examples["maximum_settlement_loss"] + gain).replace(0, np.nan)
    ).fillna(1.0)
    examples["profitable_reversion"] = (
        examples["exit_reason"].ne("settlement") & examples["pnl"].gt(0)
    ).astype(int)
    examples = _attach_pre_trade_features(examples, trades)
    missing = set(STATE_ONLY_REVERSION_FEATURES) - set(examples.columns)
    if missing:
        raise ValueError(f"Missing state-only classifier features: {sorted(missing)}")
    return examples


def state_reversion_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[:, STATE_ONLY_REVERSION_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )
    if result.isna().any().any():
        bad = result.columns[result.isna().any()].tolist()
        raise ValueError(f"State-only features contain nulls: {bad}")
    return result.astype(float)


def overshoot_reversion_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the compact, event-free classifier input contract."""
    missing = set(OVERSHOOT_REVERSION_FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing overshoot classifier features: {sorted(missing)}")
    result = frame.loc[:, OVERSHOOT_REVERSION_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )
    if result.isna().any().any():
        bad = result.columns[result.isna().any()].tolist()
        raise ValueError(f"Overshoot classifier features contain nulls: {bad}")
    return result.astype(float)
