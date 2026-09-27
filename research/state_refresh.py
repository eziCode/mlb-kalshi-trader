"""Train a cutoff-safe baseball probability model in a separate experiment.

python -m research.state_refresh --output-dir data/reboot/state_refresh_1
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import gzip
import json
import math
from pathlib import Path

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd

from research.slate_study import digest, empty_output
from settlement_value_strategy.train_local_state_model import FEATURES

ROOT = Path(__file__).resolve().parents[1]
CUTOFF = date(2026, 7, 21)
VALIDATION_END = date(2026, 8, 12)
PARAMETERS = dict(iterations=500, depth=5, learning_rate=.03, l2_leaf_reg=20,
                  loss_function="Logloss", random_seed=42, thread_count=4,
                  verbose=100, allow_writing_files=False)
RAW_COLUMNS = ["game_pk", "game_date", "game_type", "home_team", "away_team", "inning",
               "inning_topbot", "outs_when_up", "home_score", "away_score", "balls", "strikes",
               "on_1b", "on_2b", "on_3b", "at_bat_number", "pitch_number", "post_home_score", "post_away_score"]


def labels_from_feed(payload):
    data, live = payload.get("gameData") or {}, payload.get("liveData") or {}
    if (data.get("status") or {}).get("abstractGameState") != "Final":
        return None
    teams = (live.get("linescore") or {}).get("teams") or {}
    home, away = (teams.get(side, {}).get("runs") for side in ("home", "away"))
    endings = [play.get("about", {}).get("endTime") for play in (live.get("plays") or {}).get("allPlays", [])]
    endings = [datetime.fromisoformat(s.replace("Z", "+00:00")) for s in endings if s]
    if home is None or away is None or home == away or not endings:
        return None
    # Include every resumed play. A game carrying an old official date cannot
    # enter training or ratings before it actually finished.
    return {"home_win": int(home > away), "final_home_score": home, "final_away_score": away,
            "available_date": max(endings).date()}


def game_labels(frame, cache):
    games = frame.groupby("game_pk", as_index=False).agg(game_date=("game_date", "first"),
        home_team=("home_team", "first"), away_team=("away_team", "first"))
    rows, hashes = [], {}
    for game in games.to_dict("records"):
        path = cache / f"{game['game_pk']}.json.gz"
        if not path.exists():
            continue
        with gzip.open(path, "rt") as stream:
            result = labels_from_feed(json.load(stream))
        hashes[str(path.relative_to(ROOT))] = digest(path)
        if result is not None:
            rows.append({**game, **result})
    return pd.DataFrame(rows), hashes


def probability(home, away, ratings):
    return 1 / (1 + 10 ** (-(ratings.get(home, 1500.) + 24. - ratings.get(away, 1500.)) / 400.))


def prior_history(games, cutoff):
    ratings, predictions, rows = {}, {}, []
    dates = sorted(set(games.game_date) | set(games.available_date))
    for day in dates:
        if day >= cutoff:
            break
        # Outcomes are applied after all this date's pregame priors. An old
        # suspended game's result is applied on its actual completion date.
        for game in games[games.game_date.eq(day)].itertuples(index=False):
            p = probability(game.home_team, game.away_team, ratings)
            predictions[game.game_pk] = p
            rows.append({"game_pk": game.game_pk, "pregame_prob": p})
        changes = {}
        for game in games[games.available_date.eq(day)].itertuples(index=False):
            p = predictions.get(game.game_pk)
            if p is None:
                continue
            multiplier = math.log1p(max(1., abs(game.final_home_score - game.final_away_score)))
            change = 20. * multiplier * (game.home_win - p)
            changes[game.home_team] = changes.get(game.home_team, 0.) + change
            changes[game.away_team] = changes.get(game.away_team, 0.) - change
        for team, change in changes.items():
            ratings[team] = ratings.get(team, 1500.) + change
    state = {"method": "prior_calendar_day_completed_games_elo", "cutoff_exclusive": str(cutoff),
             "home_advantage": 24., "k_factor": 20., "ratings": ratings}
    return pd.DataFrame(rows), state


def raw_features(frame):
    home = frame.inning_topbot.str.lower().isin(["bot", "bottom"])
    return pd.DataFrame({"pregame_batting_prob": np.where(home, frame.pregame_prob, 1 - frame.pregame_prob),
        "inning": frame.inning, "batting_team_is_home": home.astype(int), "outs_when_up": frame.outs_when_up,
        "batting_score_diff": (frame.home_score - frame.away_score) * np.where(home, 1, -1),
        "balls": frame.balls, "strikes": frame.strikes,
        "runner_on_first": frame.on_1b.notna().astype(int), "runner_on_second": frame.on_2b.notna().astype(int),
        "runner_on_third": frame.on_3b.notna().astype(int)}).loc[:, FEATURES].astype(float)


def post_features(updates, priors):
    """Normalize a third out using baseball rules, never the next pitch."""
    home = updates.inning_topbot_after.astype(int).to_numpy().copy()
    inning = updates.inning_after.to_numpy().copy()
    outs = updates.outs_when_up_after.to_numpy().copy()
    end = outs >= 3
    inning[end & (home == 1)] += 1
    home[end] = 1 - home[end]
    outs[end] = 0
    runners = [updates[f"runner_on_{base}_after"].to_numpy().copy() for base in ("first", "second", "third")]
    for runner in runners:
        runner[end] = 0
    runners[1][end & (inning >= 10)] = 1
    p = updates.game_pk.map(priors).to_numpy()
    return pd.DataFrame({"pregame_batting_prob": np.where(home, p, 1 - p), "inning": inning,
        "batting_team_is_home": home, "outs_when_up": outs,
        "batting_score_diff": updates.score_diff_after.to_numpy() * np.where(home, 1, -1),
        "balls": np.where(end, 0, updates.balls_after), "strikes": np.where(end, 0, updates.strikes_after),
        "runner_on_first": runners[0], "runner_on_second": runners[1], "runner_on_third": runners[2]},
        index=updates.index).loc[:, FEATURES].astype(float)


def home_probability(model, features):
    p = model.predict_proba(features)[:, 1]
    return np.where(features.batting_team_is_home.eq(1), p, 1 - p)


def calibration(frame, predicted):
    p = np.clip(predicted, 1e-6, 1 - 1e-6)
    y = frame.home_win.to_numpy()
    weight = 1 / frame.groupby("game_pk").game_pk.transform("size").to_numpy()
    weight /= weight.sum()
    bins = []
    for low in np.arange(0., 1., .1):
        use = (p >= low) & (p < low + .1)
        mass = weight[use].sum()
        bins.append({"lower_bound": float(low), "rows": int(use.sum()),
                     "mean_probability": float(np.sum(p[use] * weight[use]) / mass) if mass else None,
                     "observed_win_fraction": float(np.sum(y[use] * weight[use]) / mass) if mass else None})
    return {"games": int(frame.game_pk.nunique()), "rows": len(frame),
            "game_weighted_brier": float(np.sum(weight * (p - y) ** 2)),
            "game_weighted_log_loss": float(-np.sum(weight * (y * np.log(p) + (1-y) * np.log(1-p)))),
            "bins": bins}


def run(output):
    empty_output(output)
    raw_path = ROOT / "data/raw/mlb_statcast/2026.parquet"
    old_path = ROOT / "hit_reversion_strategy/models/local_win_expectancy.cbm"
    files = [raw_path, old_path, Path(__file__), ROOT / "research/STATE_REFRESH_PROTOCOL.md"]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in files}
    manifest = {"started_at": datetime.now(timezone.utc).isoformat(), "cutoff_exclusive": str(CUTOFF),
                "calibration_end_exclusive": str(VALIDATION_END), "hashes": hashes,
                "features": FEATURES, "parameters": PARAMETERS, "deployment_ready": False}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    raw = pd.read_parquet(raw_path, columns=RAW_COLUMNS)
    raw["game_date"] = pd.to_datetime(raw.game_date).dt.date
    raw = raw[raw.game_type.eq("R") & (raw.game_date < VALIDATION_END)].copy()
    games, feed_hashes = game_labels(raw, ROOT / "data/raw/mlb_timestamps/cache/2026")
    manifest["feed_hashes"] = feed_hashes
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    priors, state = prior_history(games, CUTOFF)
    frozen = games.assign(pregame_prob=[probability(g.home_team, g.away_team, state["ratings"])
                                       for g in games.itertuples(index=False)])
    pregame = pd.concat([priors.assign(frozen=False),
        frozen.loc[frozen.game_date >= CUTOFF, ["game_pk", "pregame_prob"]].assign(frozen=True)])
    work = raw.merge(games[["game_pk", "home_win", "available_date"]], on="game_pk", validate="many_to_one")
    work = work.merge(pregame[["game_pk", "pregame_prob"]], on="game_pk", validate="many_to_one")
    train = work[(work.game_date < CUTOFF) & (work.available_date < CUTOFF)].copy()
    check = work[(work.game_date >= CUTOFF) & (work.available_date < VALIDATION_END)].copy()
    if train.empty or check.empty:
        raise ValueError("Training and calibration partitions must both contain completed games")
    x = raw_features(train)
    label = np.where(x.batting_team_is_home.eq(1), train.home_win, 1 - train.home_win)
    weights = 1 / train.groupby("game_pk").game_pk.transform("size").to_numpy()
    weights /= weights.mean()
    model = CatBoostClassifier(**PARAMETERS)
    model.fit(x, label, sample_weight=weights)
    model.save_model(str(output / "model.cbm"))
    (output / "prior.json").write_text(json.dumps(state, indent=2))
    pregame.to_parquet(output / "priors.parquet", index=False)
    check_x = raw_features(check)
    old = CatBoostClassifier().load_model(str(old_path))
    results = {"training_games": int(train.game_pk.nunique()), "training_rows": len(train),
               "raw_regular_games_before_calibration_end": int(raw.game_pk.nunique()),
               "games_with_final_feed_labels": len(games),
               "new_model": calibration(check, home_probability(model, check_x)),
               "old_model_same_inputs": calibration(check, home_probability(old, check_x)),
               "pregame_only": calibration(check, check.pregame_prob.to_numpy()),
               "model_sha256": digest(output / "model.cbm"), "deployment_ready": False}
    for name, expected in {**hashes, **feed_hashes}.items():
        if digest(ROOT / name) != expected:
            raise RuntimeError(f"Research input changed: {name}")
    (output / "calibration.json").write_text(json.dumps(results, indent=2, allow_nan=False))
    print(json.dumps({k: v if not isinstance(v, dict) else {x: y for x, y in v.items() if x != "bins"}
                      for k, v in results.items()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
