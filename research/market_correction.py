"""Test pregame anchoring and a development-only market probability correction."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd

from research.continuous import Execution
from research.settlement_study import ROOT, PARTITIONS, EVENTS, build, settlements, simulate
from research.slate_study import digest, empty_output
from research.state_refresh import home_probability, post_features, probability
from research.validation import summarize

FEATURES = ["market_logit", "anchor_disagreement", "inning_market_logit", "home_sign", "paired_gap"]
EDGES = (0., .01, .03)


def logit(p):
    p = np.clip(p, .001, .999)
    return np.log(p / (1 - p))


def logistic(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def opening_anchors(model_directory):
    raw = pd.read_parquet(ROOT / "data/raw/mlb_statcast/2026.parquet", columns=["game_pk", "home_team", "away_team"])
    teams = raw.groupby("game_pk", as_index=False).first()
    ratings = json.loads((model_directory / "prior.json").read_text())["ratings"]
    priors = {g.game_pk: probability(g.home_team, g.away_team, ratings) for g in teams.itertuples(index=False)}
    model = CatBoostClassifier().load_model(str(model_directory / "model.cbm"))
    anchors = []
    for chunk in json.loads((ROOT / "data/reboot/dataset.json").read_text())["chunks"]:
        directory = ROOT / "data/reboot" / chunk
        states = pd.read_parquet(directory / "state_updates.parquet", columns=["game_pk", "pitch_start_time"])
        trades = pd.read_parquet(directory / "home_market_trades.parquet", columns=["game_pk", "created_time", "yes_price_dollars"])
        start = states.groupby("game_pk").pitch_start_time.min()
        trades["age"] = (trades.game_pk.map(start) - trades.created_time).dt.total_seconds()
        pre = trades[(trades.age > 0) & (trades.age <= 300)].sort_values("created_time").groupby("game_pk").tail(1).copy()
        initial = pd.DataFrame({"game_pk": pre.game_pk.to_numpy(), "inning_after": 1., "inning_topbot_after": 0,
            "outs_when_up_after": 0, "score_diff_after": 0., "balls_after": 0, "strikes_after": 0,
            "runner_on_first_after": 0., "runner_on_second_after": 0., "runner_on_third_after": 0.})
        initial_p = home_probability(model, post_features(initial, priors))
        anchors.extend({"game_pk": int(pk), "offset": float(logit(price) - logit(base)),
                        "available_time": int(pd.Timestamp(stamp).value + 2e9),
                        "pregame_price": float(price), "model_initial_probability": float(base)}
                       for pk, price, stamp, base in zip(pre.game_pk, pre.yes_price_dollars, pre.created_time, initial_p))
    return pd.DataFrame(anchors).set_index("game_pk")


def annotate(frame, anchors):
    frame = frame.copy()
    sign = np.where(frame.market.eq(0), 1., -1.)
    own_fair = frame.mid + frame.fair_gap
    home_fair = np.where(sign == 1, own_fair, 1 - own_fair)
    anchor_home = logistic(logit(home_fair) + frame.game_pk.map(anchors.offset))
    available = frame.game_pk.map(anchors.available_time).to_numpy()
    anchor_home = np.where(frame.time.to_numpy() > available, anchor_home, np.nan)
    frame["anchor_probability"] = np.where(sign == 1, anchor_home, 1 - anchor_home)
    index = pd.MultiIndex.from_frame(frame[["time", "game_pk"]])
    home_mid = frame[frame.market.eq(0)].set_index(["time", "game_pk"]).mid.reindex(index).to_numpy()
    away_mid = frame[frame.market.eq(1)].set_index(["time", "game_pk"]).mid.reindex(index).to_numpy()
    frame["market_offset"] = sign * logit(home_mid)
    frame["market_logit"] = frame.market_offset
    frame["anchor_disagreement"] = sign * (logit(anchor_home) - logit(home_mid))
    frame["inning_market_logit"] = frame.market_offset * frame.inning / 9
    frame["home_sign"] = sign
    frame["paired_gap"] = sign * (1 - home_mid - away_mid)
    return frame


def fit_correction(frame, penalty=20.):
    start, end = PARTITIONS["development"]
    train = frame[(frame.game_date >= start) & (frame.game_date <= end)]
    use = train.entry_allowed & np.isfinite(train[FEATURES]).all(axis=1) & np.isfinite(train.market_offset)
    train = train[use]
    if train.empty:
        raise ValueError("No valid development features")
    x = train[FEATURES].to_numpy(float)
    weight = 1 / train.groupby("game_pk").game_pk.transform("size").to_numpy()
    scale = np.sqrt(np.average(x ** 2, axis=0, weights=weight))
    scale[scale < 1e-9] = 1.
    x /= scale
    base, y = train.market_offset.to_numpy(), train.outcome.to_numpy()
    beta = np.zeros(x.shape[1])

    def loss(b):
        z = base + x @ b
        return np.sum(weight * (np.logaddexp(0, z) - y * z)) + penalty * np.dot(b, b) / 2

    for iteration in range(50):
        p = logistic(base + x @ beta)
        gradient = x.T @ (weight * (p - y)) + penalty * beta
        hessian = x.T @ ((weight * p * (1 - p))[:, None] * x) + penalty * np.eye(x.shape[1])
        step = np.linalg.solve(hessian, gradient)
        if np.max(np.abs(step)) < 1e-8:
            break
        rate, old = 1., loss(beta)
        while loss(beta - rate * step) > old and rate > 1e-6:
            rate /= 2
        beta -= rate * step
    else:
        raise ValueError("Logistic correction did not converge")
    return {"features": FEATURES, "coefficients": beta.tolist(), "scales": scale.tolist(),
            "penalty": penalty, "training_rows": len(train), "training_games": int(train.game_pk.nunique()),
            "maximum_training_date": str(train.game_date.max()), "iterations": iteration + 1}


def corrected_probability(frame, model):
    return logistic(frame.market_offset.to_numpy() +
                    (frame[FEATURES].to_numpy() / np.array(model["scales"])) @ np.array(model["coefficients"]))


def policy_metrics(frame, prediction, edge, terminal, execution, start, end):
    part = frame[(frame.game_date >= start) & (frame.game_date <= end)].copy()
    part["fair_gap"] = prediction[part.index] - part.mid
    trades, counter = simulate(part, edge, terminal, execution)
    dates = part.groupby("game_pk").game_date.first().to_dict()
    metric = summarize(trades, dates, start, end)
    metric.update(counter, entries_per_game=len(trades) / len(dates),
                  game_coverage=trades.game_pk.nunique() / len(dates),
                  every_game_traded=trades.game_pk.nunique() == len(dates))
    return trades, metric


def run(args):
    output, model_dir = args.output_dir, args.model_dir.resolve()
    empty_output(output)
    execution = Execution(latency=args.latency, penalty=args.penalty, state_delay=args.state_delay)
    files = [Path(__file__), ROOT / "research/MARKET_CORRECTION_PROTOCOL.md", ROOT / "research/state_refresh.py",
             ROOT / "research/settlement_study.py", ROOT / "research/continuous.py", ROOT / "research/passive.py",
             ROOT / "research/validation.py", ROOT / "hit_reversion_strategy/scripts/backtest.py", EVENTS,
             ROOT / "data/raw/mlb_statcast/2026.parquet", ROOT / "data/reboot/dataset.json",
             model_dir / "model.cbm", model_dir / "prior.json", model_dir / "manifest.json"]
    for chunk in json.loads((ROOT / "data/reboot/dataset.json").read_text())["chunks"]:
        files.extend((ROOT / "data/reboot" / chunk).glob("*.*"))
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in files}
    manifest = {"started_at": datetime.now(timezone.utc).isoformat(), "hashes": hashes,
        "execution": asdict(execution), "edges": EDGES, "role": "previously_inspected_research",
        "deployment_ready": False}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    frame = annotate(build(model_dir, execution, args.workers), opening_anchors(model_dir))
    frame.to_parquet(output / "decision_frame.parquet", index=False)
    model = fit_correction(frame)
    (output / "correction.json").write_text(json.dumps(model, indent=2))
    predictions = {"anchored": frame.anchor_probability.to_numpy(), "corrected": corrected_probability(frame, model)}
    terminal, results = settlements(), {}
    for mode, prediction in predictions.items():
        for edge in EDGES:
            name = f"{mode}_{round(edge*100):02d}c"
            results[name] = {"mode": mode, "edge": edge}
            for partition in ("development", "selection"):
                trades, metric = policy_metrics(frame, prediction, edge, terminal, execution, *PARTITIONS[partition])
                results[name][partition] = metric
                trades.to_csv(output / f"{name}_{partition}.csv", index=False)
                print(f"{name} {partition}: {len(trades)} entries, ${metric['pnl']:.4f}", flush=True)
    eligible = [name for name, r in results.items() if all(r[p]["pnl"] > 0 for p in ("development", "selection"))]
    selected = max(eligible or list(results), key=lambda name: results[name]["selection"]["day_block_bootstrap_pnl_95pct"][0])
    frozen = {"frozen_at": datetime.now(timezone.utc).isoformat(), "candidate": selected,
              "eligible": eligible, "fallback": not bool(eligible), "deployment_ready": False,
              "correction_sha256": digest(output / "correction.json")}
    (output / "selection.json").write_text(json.dumps(frozen, indent=2))
    for name, result in results.items():
        trades, metric = policy_metrics(frame, predictions[result["mode"]], result["edge"], terminal,
                                        execution, *PARTITIONS["chronological_check"])
        result["chronological_check"] = metric
        trades.to_csv(output / f"{name}_chronological_check.csv", index=False)
        print(f"{name} chronological_check: {len(trades)} entries, ${metric['pnl']:.4f}", flush=True)
    for name, expected in hashes.items():
        if digest(ROOT / name) != expected:
            raise RuntimeError(f"Input changed during replay: {name}")
    (output / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--latency", type=float, default=.68)
    parser.add_argument("--penalty", type=float, default=.01)
    parser.add_argument("--state-delay", type=float, default=2.)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
