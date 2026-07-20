"""Build strategy-neutral home/away trade tapes and state updates."""

from __future__ import annotations

import argparse
from datetime import date
import gzip
import json
from pathlib import Path
import sys

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT / "hit_reversion_strategy") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "hit_reversion_strategy"))

from trade_tape_strategy.strategy import state_feature_frame  # noqa: E402


TEAM_ALIASES = {
    "AZ": "ARI", "ARI": "ARI", "CWS": "CHW", "CHW": "CHW",
    "KC": "KC", "KCR": "KC", "OAK": "ATH", "ATH": "ATH",
    "SD": "SD", "SDP": "SD", "SF": "SF", "SFG": "SF",
    "TB": "TB", "TBR": "TB", "WAS": "WSH", "WSH": "WSH",
}
TRADE_COLUMNS = [
    "game_pk", "game_date", "market_ticker", "home_win", "trade_id",
    "created_time", "yes_price_dollars", "no_price_dollars", "count_fp",
    "taker_outcome_side", "taker_book_side",
]

SETTLEMENT_STATE_FEATURES = (
    "pregame_batting_prob", "inning", "batting_team_is_home",
    "outs_when_up", "batting_score_diff", "balls", "strikes",
    "runner_on_first", "runner_on_second", "runner_on_third",
)


def settlement_state_frame(frame: pd.DataFrame) -> pd.DataFrame:
    batting_home = frame.inning_topbot.astype(int)
    result = pd.DataFrame(index=frame.index)
    result["pregame_batting_prob"] = np.where(
        batting_home.eq(1), frame.pregame_prob, 1.0 - frame.pregame_prob
    )
    result["inning"] = frame.inning
    result["batting_team_is_home"] = batting_home
    result["outs_when_up"] = frame.outs_when_up
    result["batting_score_diff"] = np.where(
        batting_home.eq(1), frame.score_diff, -frame.score_diff
    )
    for name in (
        "balls", "strikes", "runner_on_first", "runner_on_second",
        "runner_on_third",
    ):
        result[name] = frame[name]
    return result.loc[:, SETTLEMENT_STATE_FEATURES].astype(float)


def canonical_team(value: object) -> str:
    code = str(value).strip().upper()
    return TEAM_ALIASES.get(code, code)


def load_feed(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def feed_rows(cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pitch_rows: list[dict] = []
    game_rows: list[dict] = []
    paths = sorted({*cache_dir.rglob("*.json"), *cache_dir.rglob("*.json.gz")})
    if not paths:
        raise FileNotFoundError(f"No MLB live-feed cache files under {cache_dir}")
    print(f"Reading {len(paths):,} cached MLB game feeds...", flush=True)
    for index, path in enumerate(paths, 1):
        payload = load_feed(path)
        game_pk = int(payload.get("gamePk") or path.name.split(".")[0])
        game_data = payload.get("gameData") or {}
        live_data = payload.get("liveData") or {}
        final = game_data.get("status", {}).get("abstractGameState") == "Final"
        linescore = live_data.get("linescore") or {}
        teams = linescore.get("teams") or {}
        home_runs = teams.get("home", {}).get("runs")
        away_runs = teams.get("away", {}).get("runs")
        game_rows.append({
            "game_pk": game_pk,
            "home_win": (
                int(int(home_runs) > int(away_runs))
                if final and home_runs is not None and away_runs is not None
                else None
            ),
            "feed_game_date": game_data.get("datetime", {}).get("officialDate"),
        })
        for play in live_data.get("plays", {}).get("allPlays") or []:
            at_bat_index = play.get("atBatIndex")
            if at_bat_index is None:
                continue
            pitches = [
                event for event in play.get("playEvents") or []
                if event.get("isPitch") and event.get("pitchNumber") is not None
            ]
            if not pitches:
                continue
            terminal = max(int(event["pitchNumber"]) for event in pitches)
            event_type = str(play.get("result", {}).get("eventType") or "").lower()
            batting_home = not bool(play.get("about", {}).get("isTopInning"))
            for event in pitches:
                number = int(event["pitchNumber"])
                pitch_rows.append({
                    "game_pk": game_pk,
                    "at_bat_number": int(at_bat_index) + 1,
                    "pitch_number": number,
                    "pitch_start_time": event.get("startTime"),
                    "pitch_end_time": event.get("endTime"),
                    "completed_event": event_type if number == terminal else None,
                    "completed_event_batting_home": (
                        batting_home if number == terminal else None
                    ),
                })
        if index % 250 == 0:
            print(f"  parsed {index:,}/{len(paths):,} feeds", flush=True)
    pitches = pd.DataFrame(pitch_rows)
    pitches["pitch_start_time"] = pd.to_datetime(
        pitches.pitch_start_time, utc=True, errors="coerce"
    )
    pitches["pitch_end_time"] = pd.to_datetime(
        pitches.pitch_end_time, utc=True, errors="coerce"
    )
    pitches = pitches.drop_duplicates(
        ["game_pk", "at_bat_number", "pitch_number"], keep="last"
    )
    games = pd.DataFrame(game_rows).drop_duplicates("game_pk", keep="last")
    timing = pitches.groupby("game_pk", as_index=False).agg(
        first_pitch_time=("pitch_start_time", "min"),
        last_pitch_time=("pitch_end_time", "max"),
    )
    return pitches, games.merge(timing, on="game_pk", how="inner")


def load_pitch_states(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Pitch-state file not found: {path}. Run "
            "build_event_state_features.py first."
        )
    print(f"Loading pitch states from {path}...", flush=True)
    states = pd.read_parquet(path)
    required = {
        "game_pk", "game_date", "home_team", "at_bat_number", "pitch_number",
        "inning", "inning_topbot", "outs_when_up", "score_diff",
        "runner_state", "balls", "strikes",
    }
    if missing := required - set(states):
        raise ValueError(f"Pitch states missing columns: {sorted(missing)}")
    states["game_pk"] = states.game_pk.astype("int64")
    states["game_date"] = pd.to_datetime(states.game_date).dt.date
    encoded = states.inning_topbot.replace({"Top": 0, "top": 0, "Bot": 1, "bot": 1})
    states["inning_topbot"] = pd.to_numeric(encoded, errors="raise")
    runner = states.runner_state.fillna("000").astype(str).str.zfill(3)
    states["runner_on_first"] = pd.to_numeric(runner.str[0], errors="raise")
    states["runner_on_second"] = pd.to_numeric(runner.str[1], errors="raise")
    states["runner_on_third"] = pd.to_numeric(runner.str[2], errors="raise")
    return states


def load_downloaded_trades(directory: Path) -> pd.DataFrame:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No downloaded Kalshi Parquet files in {directory}")
    print(f"Loading {len(files)} Kalshi trade file(s)...", flush=True)
    trades = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    required = {
        "market_ticker", "game_date", "trade_id", "created_time",
        "yes_price_dollars", "no_price_dollars", "count_fp",
        "taker_outcome_side", "taker_book_side",
    }
    if missing := required - set(trades):
        raise ValueError(f"Downloaded trades missing columns: {sorted(missing)}")
    trades["created_time"] = pd.to_datetime(trades.created_time, utc=True)
    trades["game_date"] = pd.to_datetime(trades.game_date).dt.date
    return trades.sort_values(["created_time", "trade_id"]).drop_duplicates("trade_id")


def map_games_to_markets(
    states: pd.DataFrame, games: pd.DataFrame, trades: pd.DataFrame,
) -> pd.DataFrame:
    state_games = states.groupby("game_pk", as_index=False).agg(
        game_date=("game_date", "first"), home_team=("home_team", "first")
    )
    mapped_games = state_games.merge(games, on="game_pk", how="inner")
    if "game_pk" in trades and trades.game_pk.notna().any():
        direct = trades.dropna(subset=["game_pk"])[
            ["game_pk", "market_ticker"]
        ].drop_duplicates()
        counts = direct.groupby("game_pk").market_ticker.nunique()
        if counts.le(1).all():
            return mapped_games.merge(direct, on="game_pk", how="inner")
    mapped_games["home_code"] = mapped_games.home_team.map(canonical_team)
    markets = trades[["game_date", "market_ticker"]].drop_duplicates().copy()
    markets["home_code"] = markets.market_ticker.str.rsplit("-", n=1).str[-1].map(
        canonical_team
    )
    markets = markets.sort_values(["game_date", "home_code", "market_ticker"])
    markets["ordinal"] = markets.groupby(["game_date", "home_code"]).cumcount()
    mapped_games = mapped_games.sort_values(
        ["game_date", "home_code", "first_pitch_time"]
    )
    mapped_games["ordinal"] = mapped_games.groupby(
        ["game_date", "home_code"]
    ).cumcount()
    result = mapped_games.merge(
        markets, on=["game_date", "home_code", "ordinal"], how="left"
    )
    if result.market_ticker.isna().any():
        missing = result.loc[result.market_ticker.isna(), "game_pk"].tolist()
        print(
            f"Skipping {len(missing):,} MLB games without a traded Kalshi "
            f"home market (examples: {missing[:10]})",
            flush=True,
        )
        result = result.dropna(subset=["market_ticker"])
    if result.empty:
        raise RuntimeError("No MLB games mapped to traded Kalshi home markets")
    return result.reset_index(drop=True)


def build_shared(
    pitch_states: Path, feed_cache: Path, trade_dir: Path,
    model_path: Path, output_dir: Path,
    settlement_model_train_end: date | None = None,
    settlement_model_output: Path | None = None,
    settlement_state_output: Path | None = None,
    settlement_pregame_priors: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    states = load_pitch_states(pitch_states)
    pitch_times, feed_games = feed_rows(feed_cache)
    downloaded = load_downloaded_trades(trade_dir)
    games = map_games_to_markets(states, feed_games, downloaded)
    games = games.dropna(subset=["home_win"]).copy()
    games["home_win"] = games.home_win.astype(int)
    print(f"Mapped {len(games):,} completed games to home markets", flush=True)

    downloaded = downloaded.drop(
        columns=["game_pk", "home_win"], errors="ignore"
    )
    home_trades = downloaded.merge(
        games[[
            "game_pk", "game_date", "market_ticker", "home_win",
            "first_pitch_time", "last_pitch_time",
        ]], on=["game_date", "market_ticker"], how="inner",
    )
    anchor_rows = (
        home_trades[home_trades.created_time < home_trades.first_pitch_time]
        .sort_values(["game_pk", "created_time", "trade_id"])
        .groupby("game_pk", as_index=False).tail(1)
    )
    anchors = anchor_rows[["game_pk", "yes_price_dollars"]].rename(
        columns={"yes_price_dollars": "pregame_prob"}
    )
    games = games.merge(anchors, on="game_pk", how="inner")
    home_trades = home_trades[
        (home_trades.created_time >= home_trades.first_pitch_time)
        & (home_trades.created_time <= home_trades.last_pitch_time)
    ]
    home_trades = pd.concat([anchor_rows, home_trades], ignore_index=True)
    for column in TRADE_COLUMNS:
        if column not in home_trades:
            home_trades[column] = None
    home_trades = home_trades[TRADE_COLUMNS].drop_duplicates("trade_id").sort_values(
        ["game_pk", "created_time", "trade_id"]
    )

    # Preserve the paired away-team YES market. Buying this contract is the
    # same settlement view as buying NO on the home-team market, but it has an
    # independent book and materially different fill opportunities.
    games["event_ticker"] = games.market_ticker.str.rsplit("-", n=1).str[0]
    market_map = downloaded[["market_ticker"]].drop_duplicates().copy()
    market_map["event_ticker"] = market_map.market_ticker.str.rsplit(
        "-", n=1
    ).str[0]
    paired = games[[
        "game_pk", "game_date", "market_ticker", "home_win",
        "first_pitch_time", "last_pitch_time", "event_ticker",
    ]].rename(columns={"market_ticker": "home_market_ticker"}).merge(
        market_map, on="event_ticker", how="inner"
    )
    paired = paired[paired.market_ticker.ne(paired.home_market_ticker)]
    counts = paired.groupby("game_pk").market_ticker.nunique()
    paired = paired[paired.game_pk.isin(counts[counts.eq(1)].index)]
    away_trades = downloaded.merge(
        paired, on=["game_date", "market_ticker", "event_ticker"], how="inner"
    )
    if "market_result" in away_trades:
        expected = away_trades.home_win.map({0: "yes", 1: "no"})
        bad_games = set(away_trades.loc[
            away_trades.market_result.notna()
            & away_trades.market_result.astype(str).str.lower().ne(expected),
            "game_pk",
        ])
        if bad_games:
            print(
                f"Excluding {len(bad_games)} paired markets with inconsistent "
                "settlement mapping",
                flush=True,
            )
            away_trades = away_trades[~away_trades.game_pk.isin(bad_games)]
    away_trades = away_trades[
        (away_trades.created_time >= away_trades.first_pitch_time)
        & (away_trades.created_time <= away_trades.last_pitch_time)
    ].copy()
    away_columns = [*TRADE_COLUMNS, "home_market_ticker"]
    for column in away_columns:
        if column not in away_trades:
            away_trades[column] = None
    away_trades = away_trades[away_columns].drop_duplicates("trade_id").sort_values(
        ["game_pk", "created_time", "trade_id"]
    )

    work = states.merge(
        games[["game_pk", "game_date", "market_ticker", "home_win", "pregame_prob"]],
        on=["game_pk", "game_date"], how="inner",
    ).sort_values(["game_pk", "at_bat_number", "pitch_number"])
    model = CatBoostClassifier()
    model.load_model(model_path)
    work["fair_before"] = model.predict_proba(state_feature_frame(work))[:, 1]
    work["fair_after"] = work.groupby("game_pk").fair_before.shift(-1)
    post = [
        "inning", "inning_topbot", "outs_when_up", "score_diff", "balls",
        "strikes", "runner_on_first", "runner_on_second", "runner_on_third",
    ]
    for column in post:
        work[f"{column}_after"] = work.groupby("game_pk")[column].shift(-1)
    work = work.drop(columns=[
        "pitch_start_time", "pitch_end_time", "completed_event",
        "completed_event_batting_home",
    ], errors="ignore").merge(
        pitch_times, on=["game_pk", "at_bat_number", "pitch_number"], how="inner"
    ).dropna(subset=["fair_after", "pitch_start_time", "pitch_end_time"])
    work["is_hit"] = work.completed_event.isin(
        ["single", "double", "triple", "home_run"]
    )
    state_columns = [
        "game_pk", "game_date", "market_ticker", "home_win", "at_bat_number",
        "pitch_number", "pitch_start_time", "pitch_end_time", "completed_event",
        "completed_event_batting_home", "is_hit", "fair_before", "fair_after",
        *[f"{column}_after" for column in post],
    ]
    updates = work[state_columns].sort_values(
        ["game_pk", "pitch_end_time", "at_bat_number", "pitch_number"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    home_trades.to_parquet(output_dir / "home_market_trades.parquet", index=False)
    away_trades.to_parquet(output_dir / "away_market_trades.parquet", index=False)
    updates.to_parquet(output_dir / "state_updates.parquet", index=False)

    if settlement_model_train_end is not None:
        if settlement_model_output is None or settlement_state_output is None:
            raise ValueError(
                "Settlement model and state output paths are required with "
                "--settlement-model-train-end"
            )
        settlement = states.merge(
            games[[
                "game_pk", "game_date", "market_ticker", "home_win",
                "pregame_prob",
            ]],
            on=["game_pk", "game_date"], how="inner",
        ).sort_values(["game_pk", "at_bat_number", "pitch_number"])
        if settlement_pregame_priors is None:
            raise ValueError("Settlement MLB pregame priors are required")
        priors = pd.read_parquet(
            settlement_pregame_priors, columns=["game_pk", "pregame_prob"]
        )
        settlement = settlement.drop(columns="pregame_prob").merge(
            priors, on="game_pk", how="inner"
        )
        settlement_model = CatBoostClassifier(
        )
        settlement_model.load_model(settlement_model_output)
        batting_probability = settlement_model.predict_proba(
            settlement_state_frame(settlement), thread_count=-1
        )[:, 1]
        settlement["fair_before"] = np.where(
            settlement.inning_topbot.astype(int).eq(1),
            batting_probability, 1.0 - batting_probability,
        )
        settlement["fair_after"] = settlement.groupby(
            "game_pk"
        ).fair_before.shift(-1)
        for column in post:
            settlement[f"{column}_after"] = settlement.groupby(
                "game_pk"
            )[column].shift(-1)
        settlement = settlement.drop(columns=[
            "pitch_start_time", "pitch_end_time", "completed_event",
            "completed_event_batting_home",
        ], errors="ignore").merge(
            pitch_times,
            on=["game_pk", "at_bat_number", "pitch_number"], how="inner",
        ).dropna(subset=["fair_after", "pitch_start_time", "pitch_end_time"])
        settlement["is_hit"] = settlement.completed_event.isin(
            ["single", "double", "triple", "home_run"]
        )
        settlement_updates = settlement[state_columns].sort_values(
            ["game_pk", "pitch_end_time", "at_bat_number", "pitch_number"]
        )
        settlement_state_output.parent.mkdir(parents=True, exist_ok=True)
        settlement_updates.to_parquet(settlement_state_output, index=False)
        print(
            f"Wrote {len(settlement_updates):,} leakage-free settlement "
            f"state updates to {settlement_state_output}", flush=True,
        )
    print(
        f"Wrote {len(home_trades):,} home trades, {len(away_trades):,} away "
        f"trades, and {len(updates):,} state updates "
        f"to {output_dir}", flush=True,
    )
    return home_trades, updates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pitch-states", type=Path,
        default=REPOSITORY_ROOT / "data/processed/mlb_game_state/pitch_state_features.parquet",
    )
    parser.add_argument(
        "--feed-cache", type=Path,
        default=REPOSITORY_ROOT / "data/raw/mlb_timestamps/cache",
    )
    parser.add_argument(
        "--trade-dir", type=Path,
        default=REPOSITORY_ROOT / "data/raw/kalshi_live_market_logs",
    )
    parser.add_argument(
        "--model", type=Path,
        default=REPOSITORY_ROOT / "hit_reversion_strategy/models/local_win_expectancy.cbm",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPOSITORY_ROOT / "data/shared",
    )
    parser.add_argument("--settlement-model-train-end", type=date.fromisoformat)
    parser.add_argument("--settlement-model-output", type=Path)
    parser.add_argument("--settlement-state-output", type=Path)
    parser.add_argument("--settlement-pregame-priors", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_shared(
        args.pitch_states, args.feed_cache, args.trade_dir,
        args.model, args.output_dir, args.settlement_model_train_end,
        args.settlement_model_output, args.settlement_state_output,
        args.settlement_pregame_priors,
    )
