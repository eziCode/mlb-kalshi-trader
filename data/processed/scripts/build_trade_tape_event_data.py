"""Build exact-timestamp MLB state updates and a filtered Kalshi trade tape.

No network calls are made. MLB pitch start/end timestamps are read from the
existing compressed live-feed cache, and Kalshi rows come from the downloaded
trade-level parquet file. The output contains observed trades, not inferred
quotes or a reconstructed order book.
"""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

from catboost import CatBoostClassifier
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlb_kalshi.strategy import state_feature_frame  # noqa: E402


PITCH_STATE_PATH = (
    PROJECT_ROOT / "data/processed/mlb_game_state/pitch_state_features.parquet"
)
MLB_CACHE_DIR = PROJECT_ROOT / "data/raw/mlb_timestamps/cache/2026"
TRADE_DIR = PROJECT_ROOT / "data/raw/kalshi_live_market_logs"
TRAIN_PATH = PROJECT_ROOT / "data/processed/train/training_dataset.parquet"
TEST_PATH = PROJECT_ROOT / "data/processed/test/test_dataset.parquet"
MODEL_PATH = (
    PROJECT_ROOT / "models/market_reaction_model/local_win_expectancy.cbm"
)
OUTPUT_DIR = PROJECT_ROOT / "data/processed/trade_tape"
STATE_OUTPUT = OUTPUT_DIR / "state_updates.parquet"
TRADE_OUTPUT = OUTPUT_DIR / "home_market_trades.parquet"

STATE_COLUMNS = [
    "pregame_prob", "inning", "inning_topbot", "outs_when_up",
    "score_diff", "balls", "strikes", "runner_on_first",
    "runner_on_second", "runner_on_third",
]


def parse_utc(value) -> pd.Timestamp | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return pd.Timestamp(parsed.astimezone(timezone.utc))


def load_game_map() -> pd.DataFrame:
    columns = [
        "game_pk", "game_date", "market_ticker", "pregame_prob",
        "home_win", "first_pitch_time", "last_pitch_time",
    ]
    frames = [
        pd.read_parquet(TRAIN_PATH, columns=columns),
        pd.read_parquet(TEST_PATH, columns=columns),
    ]
    result = pd.concat(frames, ignore_index=True).drop_duplicates("game_pk")
    result["game_pk"] = result["game_pk"].astype("int64")
    result["game_date"] = pd.to_datetime(result["game_date"]).dt.date
    return result.sort_values("game_date").reset_index(drop=True)


def extract_cached_pitch_times(game_pks: set[int]) -> pd.DataFrame:
    rows: list[dict] = []
    for game_pk in sorted(game_pks):
        path = MLB_CACHE_DIR / f"{game_pk}.json.gz"
        if not path.exists():
            continue
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        plays = (
            payload.get("liveData", {}).get("plays", {}).get("allPlays", [])
        )
        for play in plays:
            at_bat_index = play.get("atBatIndex")
            if at_bat_index is None:
                continue
            pitches = [
                event for event in (play.get("playEvents") or [])
                if event.get("isPitch") and event.get("pitchNumber") is not None
            ]
            if not pitches:
                continue
            terminal_pitch = max(int(pitch["pitchNumber"]) for pitch in pitches)
            event_type = str(
                play.get("result", {}).get("eventType") or ""
            ).lower()
            batting_home = not bool(play.get("about", {}).get("isTopInning"))
            for pitch in pitches:
                pitch_number = int(pitch["pitchNumber"])
                rows.append({
                    "game_pk": game_pk,
                    "at_bat_number": int(at_bat_index) + 1,
                    "pitch_number": pitch_number,
                    "pitch_start_time": parse_utc(pitch.get("startTime")),
                    "pitch_end_time": parse_utc(pitch.get("endTime")),
                    "completed_event": (
                        event_type if pitch_number == terminal_pitch else None
                    ),
                    "completed_event_batting_home": (
                        batting_home if pitch_number == terminal_pitch else None
                    ),
                })
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("No cached MLB pitch times matched mapped games")
    result = result.dropna(subset=["pitch_start_time", "pitch_end_time"])
    return result.drop_duplicates(
        ["game_pk", "at_bat_number", "pitch_number"], keep="last"
    )


def add_runner_flags(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    state = result["runner_state"].fillna("000").astype(str).str.zfill(3)
    result["runner_on_first"] = pd.to_numeric(state.str[0], errors="coerce")
    result["runner_on_second"] = pd.to_numeric(state.str[1], errors="coerce")
    result["runner_on_third"] = pd.to_numeric(state.str[2], errors="coerce")
    encoded = result["inning_topbot"].replace({"Top": 0, "Bot": 1})
    result["inning_topbot"] = pd.to_numeric(encoded, errors="coerce")
    return result


def build_state_updates(game_map: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "game_pk", "at_bat_number", "pitch_number", "inning",
        "inning_topbot", "outs_when_up", "score_diff", "runner_state",
        "balls", "strikes",
    ]
    states = pd.read_parquet(PITCH_STATE_PATH, columns=columns)
    states = states[states["game_pk"].isin(game_map["game_pk"])].copy()
    states = add_runner_flags(states)
    states = states.merge(
        game_map[[
            "game_pk", "game_date", "market_ticker", "pregame_prob", "home_win",
        ]],
        on="game_pk",
        how="inner",
    ).sort_values(["game_pk", "at_bat_number", "pitch_number"])

    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    states["fair_before"] = model.predict_proba(
        state_feature_frame(states)
    )[:, 1]
    states["fair_after"] = states.groupby("game_pk")["fair_before"].shift(-1)
    post_state_sources = [
        "inning", "inning_topbot", "outs_when_up", "score_diff", "balls",
        "strikes", "runner_on_first", "runner_on_second", "runner_on_third",
    ]
    for column in post_state_sources:
        states[f"{column}_after"] = states.groupby("game_pk")[column].shift(-1)

    pitch_times = extract_cached_pitch_times(set(game_map["game_pk"]))
    updates = states.merge(
        pitch_times,
        on=["game_pk", "at_bat_number", "pitch_number"],
        how="inner",
    ).dropna(subset=["fair_after", "pitch_start_time", "pitch_end_time"])
    updates["is_hit"] = updates["completed_event"].isin(
        ["single", "double", "triple", "home_run"]
    )
    output_columns = [
        "game_pk", "game_date", "market_ticker", "home_win",
        "at_bat_number", "pitch_number", "pitch_start_time", "pitch_end_time",
        "completed_event", "completed_event_batting_home", "is_hit",
        "fair_before", "fair_after",
        "inning_after", "inning_topbot_after", "outs_when_up_after",
        "score_diff_after", "balls_after", "strikes_after",
        "runner_on_first_after", "runner_on_second_after",
        "runner_on_third_after",
    ]
    return updates[output_columns].sort_values(
        ["game_pk", "pitch_end_time", "at_bat_number", "pitch_number"]
    ).reset_index(drop=True)


def find_trade_file() -> Path:
    files = sorted(TRADE_DIR.glob("kalshi_mlb_trades_*_non_block.parquet"))
    if not files:
        raise FileNotFoundError(f"No downloaded trade parquet in {TRADE_DIR}")
    if len(files) > 1:
        raise RuntimeError(f"Expected one consolidated trade file, found {files}")
    return files[0]


def build_home_trade_tape(
    game_map: pd.DataFrame,
    updates: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "market_ticker", "trade_id", "created_time", "yes_price_dollars",
        "no_price_dollars", "count_fp", "taker_outcome_side",
        "taker_book_side",
    ]
    trades = pd.read_parquet(find_trade_file(), columns=columns)
    trades = trades[trades["market_ticker"].isin(game_map["market_ticker"])].copy()
    trades["created_time"] = pd.to_datetime(trades["created_time"], utc=True)
    trades = trades.merge(
        game_map[["game_pk", "game_date", "market_ticker", "home_win"]],
        on="market_ticker",
        how="inner",
    )

    windows = updates.groupby("game_pk", as_index=False).agg(
        first_pitch_start=("pitch_start_time", "min"),
        last_pitch_end=("pitch_end_time", "max"),
    )
    trades = trades.merge(windows, on="game_pk", how="inner")
    in_game = trades[
        (trades["created_time"] >= trades["first_pitch_start"])
        & (trades["created_time"] <= trades["last_pitch_end"])
    ]
    pregame = (
        trades[trades["created_time"] < trades["first_pitch_start"]]
        .sort_values(["game_pk", "created_time", "trade_id"])
        .groupby("game_pk", as_index=False)
        .tail(1)
    )
    result = pd.concat([pregame, in_game], ignore_index=True)
    result = result.drop_duplicates("trade_id").sort_values(
        ["game_pk", "created_time", "trade_id"]
    )
    return result[[
        "game_pk", "game_date", "market_ticker", "home_win", "trade_id",
        "created_time", "yes_price_dollars", "no_price_dollars", "count_fp",
        "taker_outcome_side", "taker_book_side",
    ]].reset_index(drop=True)


def main() -> None:
    game_map = load_game_map()
    print(f"Mapped home markets: {len(game_map):,}")
    updates = build_state_updates(game_map)
    print(
        f"Exact state updates: {len(updates):,}; "
        f"completed hits: {updates['is_hit'].sum():,}"
    )
    trades = build_home_trade_tape(game_map, updates)
    print(
        f"Filtered home-market trades: {len(trades):,} across "
        f"{trades['game_pk'].nunique():,} games"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    updates.to_parquet(STATE_OUTPUT, index=False)
    trades.to_parquet(TRADE_OUTPUT, index=False)
    print(f"Saved {STATE_OUTPUT}")
    print(f"Saved {TRADE_OUTPUT}")


if __name__ == "__main__":
    main()
