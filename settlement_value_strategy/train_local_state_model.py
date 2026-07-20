"""Train the local win model from pitch-clock-era MLB states only."""

from __future__ import annotations

import argparse
from datetime import date
import json
import math
from pathlib import Path

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data/processed/mlb_game_state/pitch_state_features.parquet"
MODEL = ROOT / "model/local_win_expectancy.cbm"
PRIORS = ROOT.parent / "data/settlement_value/mlb_pregame_priors.parquet"
PRIOR_STATE = ROOT / "model/mlb_pregame_prior.json"
FEATURES = (
    "pregame_batting_prob", "inning", "batting_team_is_home",
    "outs_when_up", "batting_score_diff",
    "balls", "strikes", "runner_on_first", "runner_on_second",
    "runner_on_third",
)


def batting_home_indicator(frame: pd.DataFrame) -> pd.Series:
    encoded = frame.inning_topbot.replace(
        {"Top": 0, "top": 0, "Bot": 1, "bot": 1}
    )
    return pd.to_numeric(encoded, errors="raise").astype(int)


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    batting_home = batting_home_indicator(frame)
    runners = frame.runner_state.fillna("000").astype(str).str.zfill(3)
    return pd.DataFrame({
        "pregame_batting_prob": np.where(
            batting_home.eq(1), frame.pregame_prob, 1.0 - frame.pregame_prob
        ),
        "inning": frame.inning,
        "batting_team_is_home": batting_home,
        "outs_when_up": frame.outs_when_up,
        "batting_score_diff": np.where(
            batting_home.eq(1), frame.score_diff, -frame.score_diff
        ),
        "balls": frame.balls,
        "strikes": frame.strikes,
        "runner_on_first": pd.to_numeric(runners.str[0]),
        "runner_on_second": pd.to_numeric(runners.str[1]),
        "runner_on_third": pd.to_numeric(runners.str[2]),
    }).loc[:, FEATURES].astype(float)


def build_rolling_priors(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    games = frame.sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"]
    ).groupby("game_pk", as_index=False).agg(
        game_date=("game_date", "first"), home_team=("home_team", "first"),
        away_team=("away_team", "first"),
        home_score=("post_home_score", "max"),
        away_score=("post_away_score", "max"),
    ).sort_values(["game_date", "game_pk"])
    ratings: dict[str, float] = {}
    rows = []
    home_advantage = 24.0
    k_factor = 20.0
    for game in games.itertuples(index=False):
        home_rating = ratings.get(str(game.home_team), 1500.0)
        away_rating = ratings.get(str(game.away_team), 1500.0)
        probability = 1.0 / (
            1.0 + 10.0 ** (-(home_rating + home_advantage - away_rating) / 400.0)
        )
        rows.append({
            "game_pk": int(game.game_pk), "game_date": game.game_date,
            "home_team": str(game.home_team), "away_team": str(game.away_team),
            "pregame_prob": probability,
        })
        outcome = float(game.home_score > game.away_score)
        margin = abs(float(game.home_score) - float(game.away_score))
        multiplier = math.log1p(max(1.0, margin))
        change = k_factor * multiplier * (outcome - probability)
        ratings[str(game.home_team)] = home_rating + change
        ratings[str(game.away_team)] = away_rating - change
    state = {
        "method": "chronological_elo", "initial_rating": 1500.0,
        "home_advantage": home_advantage, "k_factor": k_factor,
        "ratings": ratings,
    }
    return pd.DataFrame(rows), state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=MODEL)
    parser.add_argument("--priors-output", type=Path, default=PRIORS)
    parser.add_argument("--prior-state-output", type=Path, default=PRIOR_STATE)
    parser.add_argument("--start-season", type=int, default=2023)
    parser.add_argument("--train-end", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    frame = pd.read_parquet(args.data)
    frame["game_date"] = pd.to_datetime(frame.game_date).dt.date
    priors, prior_state = build_rolling_priors(frame)
    args.priors_output.parent.mkdir(parents=True, exist_ok=True)
    priors.to_parquet(args.priors_output, index=False)
    args.prior_state_output.parent.mkdir(parents=True, exist_ok=True)
    args.prior_state_output.write_text(json.dumps(prior_state, indent=2))
    frame = frame.merge(
        priors[["game_pk", "pregame_prob"]], on="game_pk", how="inner"
    )
    dates = frame.game_date
    frame = frame[
        (dates >= date(args.start_season, 1, 1)) & (dates < args.train_end)
    ].copy().sort_values(["game_pk", "at_bat_number", "pitch_number"])
    final = frame.groupby("game_pk", as_index=False).tail(1)
    home_win = final.set_index("game_pk").apply(
        lambda row: int(row.home_score > row.away_score), axis=1
    )
    frame["home_win"] = frame.game_pk.map(home_win)
    frame = frame.dropna(subset=["home_win"])
    batting_home = batting_home_indicator(frame)
    label = np.where(batting_home.eq(1), frame.home_win, 1 - frame.home_win)
    counts = frame.groupby("game_pk").game_pk.transform("size")
    model = CatBoostClassifier(
        iterations=500, depth=5, learning_rate=.03, l2_leaf_reg=20,
        loss_function="Logloss", random_seed=42, verbose=100,
        allow_writing_files=False,
    )
    model.fit(feature_frame(frame), label, sample_weight=1.0 / counts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(args.output)
    print(
        f"Saved {args.output} from {frame.game_pk.nunique():,} MLB games "
        f"({args.start_season} through {args.train_end})"
    )


if __name__ == "__main__":
    main()
