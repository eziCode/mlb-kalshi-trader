"""Recreate the original normalized trade/state inputs from raw sources.

Inputs under ``raw/``:
  statcast/*.parquet
  mlb_feeds/<game_pk>.json or .json.gz
  kalshi_trades.parquet
Optional ``game_market_map.csv`` resolves ambiguous games and must contain
``game_pk,market_ticker``. Unambiguous games are matched by date and the
home-team suffix in the Kalshi market ticker.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
STATE_FEATURES = (
    "pregame_batting_prob", "inning", "batting_team_is_home",
    "outs_when_up", "batting_score_diff", "balls", "strikes",
    "runner_on_first", "runner_on_second", "runner_on_third",
)
TEAM_ALIASES = {
    "ARI": "AZ", "AZ": "AZ", "ATH": "ATH", "OAK": "ATH",
    "CHW": "CWS", "CWS": "CWS", "KC": "KC", "KCR": "KC",
    "LAA": "LAA", "LAD": "LAD", "NYM": "NYM", "NYY": "NYY",
    "SD": "SD", "SDP": "SD", "SF": "SF", "SFG": "SF",
    "TB": "TB", "TBR": "TB", "WSH": "WSH", "WSN": "WSH",
}


def canonical_team(value: str) -> str:
    value = str(value).upper()
    return TEAM_ALIASES.get(value, value)


def load_feed(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def pitch_times(feed_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted([*feed_dir.glob("*.json"), *feed_dir.glob("*.json.gz")]):
        game_pk = int(path.name.split(".")[0])
        feed = load_feed(path)
        for play in feed.get("liveData", {}).get("plays", {}).get("allPlays", []):
            at_bat = play.get("atBatIndex")
            if at_bat is None:
                continue
            pitches = [event for event in play.get("playEvents", []) if event.get("isPitch")]
            for pitch in pitches:
                if pitch.get("pitchNumber") is None or not pitch.get("endTime"):
                    continue
                rows.append({
                    "game_pk": game_pk,
                    "at_bat_number": int(at_bat) + 1,
                    "pitch_number": int(pitch["pitchNumber"]),
                    "pitch_start_time": pd.to_datetime(pitch.get("startTime"), utc=True),
                    "pitch_end_time": pd.to_datetime(pitch["endTime"], utc=True),
                })
    if not rows:
        raise RuntimeError(f"No pitch timestamps found in {feed_dir}")
    return pd.DataFrame(rows).drop_duplicates(
        ["game_pk", "at_bat_number", "pitch_number"], keep="last"
    )


def load_statcast(directory: Path) -> pd.DataFrame:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Statcast parquet files in {directory}")
    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    required = {
        "game_pk", "game_date", "home_team", "at_bat_number", "pitch_number",
        "inning", "inning_topbot", "outs_when_up", "home_score", "away_score",
        "balls", "strikes", "on_1b", "on_2b", "on_3b",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Statcast data missing columns: {sorted(missing)}")
    frame["game_date"] = pd.to_datetime(frame.game_date).dt.date
    frame["inning_topbot"] = frame.inning_topbot.map({"Top": 0, "Bot": 1}).fillna(
        pd.to_numeric(frame.inning_topbot, errors="coerce")
    )
    frame["score_diff"] = frame.home_score - frame.away_score
    frame["runner_on_first"] = frame.on_1b.notna().astype(int)
    frame["runner_on_second"] = frame.on_2b.notna().astype(int)
    frame["runner_on_third"] = frame.on_3b.notna().astype(int)
    return frame.sort_values(["game_pk", "at_bat_number", "pitch_number"])


def game_table(states: pd.DataFrame, times: pd.DataFrame) -> pd.DataFrame:
    timing = times.groupby("game_pk", as_index=False).agg(
        first_pitch_time=("pitch_start_time", "min"),
        last_pitch_time=("pitch_end_time", "max"),
    )
    games = states.groupby("game_pk", as_index=False).agg(
        game_date=("game_date", "first"), home_team=("home_team", "first"),
        final_home_score=("home_score", "last"),
        final_away_score=("away_score", "last"),
    )
    games["home_win"] = (games.final_home_score > games.final_away_score).astype(int)
    games["home_code"] = games.home_team.map(canonical_team)
    return games.merge(timing, on="game_pk", how="inner")


def map_markets(games: pd.DataFrame, trades: pd.DataFrame, override: Path) -> pd.DataFrame:
    if override.exists():
        mapping = pd.read_csv(override, usecols=["game_pk", "market_ticker"])
        return games.merge(mapping, on="game_pk", how="inner")
    market_dates = trades[["market_ticker", "game_date"]].drop_duplicates().copy()
    market_dates["game_date"] = pd.to_datetime(market_dates.game_date).dt.date
    candidates = games.merge(market_dates, on="game_date", how="left")
    candidates = candidates[candidates.apply(
        lambda row: str(row.market_ticker).endswith(f"-{row.home_code}"), axis=1
    )]
    counts = candidates.groupby("game_pk").size()
    ambiguous = counts[counts.ne(1)]
    if not ambiguous.empty:
        raise RuntimeError(
            "Ambiguous/missing market mapping for game_pk values "
            f"{ambiguous.index.tolist()}; provide raw/game_market_map.csv"
        )
    return candidates


def state_model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    batting_home = frame.inning_topbot.astype(int)
    result = pd.DataFrame(index=frame.index)
    result["pregame_batting_prob"] = np.where(
        batting_home.eq(1), frame.pregame_prob, 1 - frame.pregame_prob
    )
    result["inning"] = frame.inning
    result["batting_team_is_home"] = batting_home
    result["outs_when_up"] = frame.outs_when_up
    result["batting_score_diff"] = np.where(
        batting_home.eq(1), frame.score_diff, -frame.score_diff
    )
    for name in ("balls", "strikes", "runner_on_first", "runner_on_second", "runner_on_third"):
        result[name] = frame[name]
    return result.loc[:, STATE_FEATURES].astype(float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=RAW)
    args = parser.parse_args()
    raw = args.raw
    states = load_statcast(raw / "statcast")
    times = pitch_times(raw / "mlb_feeds")
    trades = pd.read_parquet(raw / "kalshi_trades.parquet")
    trades["game_date"] = pd.to_datetime(trades.game_date).dt.date
    trades["created_time"] = pd.to_datetime(trades.created_time, utc=True)
    games = map_markets(
        game_table(states, times), trades, raw / "game_market_map.csv"
    )
    trades = trades.merge(
        games[["game_pk", "game_date", "market_ticker", "home_win", "first_pitch_time", "last_pitch_time"]],
        on=["game_date", "market_ticker"], how="inner", suffixes=("", "_game"),
    )
    anchors = (
        trades[trades.created_time < trades.first_pitch_time]
        .sort_values(["game_pk", "created_time", "trade_id"])
        .groupby("game_pk", as_index=False).tail(1)
    )
    anchors["pregame_prob"] = anchors.yes_price_dollars.astype(float)
    games = games.merge(anchors[["game_pk", "pregame_prob"]], on="game_pk", how="inner")
    states = states.merge(games[["game_pk", "game_date", "market_ticker", "home_win", "pregame_prob"]], on=["game_pk", "game_date"], how="inner")
    states = states.merge(times, on=["game_pk", "at_bat_number", "pitch_number"], how="inner")
    model = CatBoostClassifier()
    model.load_model(ROOT / "model/local_win_expectancy.cbm")
    batting = model.predict_proba(state_model_frame(states))[:, 1]
    states["fair_before"] = np.where(states.inning_topbot.eq(1), batting, 1 - batting)
    grouped = states.groupby("game_pk", sort=False)
    states["fair_after"] = grouped.fair_before.shift(-1)
    for name in ("inning", "inning_topbot", "outs_when_up", "score_diff", "balls", "strikes", "runner_on_first", "runner_on_second", "runner_on_third"):
        states[f"{name}_after"] = grouped[name].shift(-1)
    output_states = states.dropna(subset=["fair_after"])[[
        "game_pk", "game_date", "market_ticker", "home_win", "at_bat_number",
        "pitch_number", "pitch_start_time", "pitch_end_time", "fair_before",
        "fair_after", "inning_after", "inning_topbot_after", "outs_when_up_after",
        "score_diff_after", "balls_after", "strikes_after", "runner_on_first_after",
        "runner_on_second_after", "runner_on_third_after",
    ]]
    output_trades = trades[
        (trades.created_time >= trades.first_pitch_time)
        & (trades.created_time <= trades.last_pitch_time)
    ].copy()
    pregame = anchors[output_trades.columns.intersection(anchors.columns)]
    output_trades = pd.concat([pregame, output_trades], ignore_index=True)
    output_states.to_parquet(raw / "state_updates.parquet", index=False)
    output_trades.to_parquet(raw / "home_market_trades.parquet", index=False)
    print(f"wrote {len(output_states):,} states and {len(output_trades):,} trades")


if __name__ == "__main__":
    main()
