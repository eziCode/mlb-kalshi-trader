"""Stress a frozen market correction without refitting or reselecting it."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path

import pandas as pd

from research.continuous import Execution
from research.market_correction import (annotate, corrected_probability, opening_anchors,
                                         policy_metrics)
from research.settlement_study import PARTITIONS, ROOT, refreshed_chunk, settlements
from research.slate_study import digest, empty_output
from research.state_refresh import probability

SCENARIOS = {"reference": {}, "latency_1_5s": {"latency": 1.5}, "latency_3s": {"latency": 3.},
             "two_cent_cost": {"penalty": .02}, "publication_plus_2s": {"state_delay": 4.},
             "one_percent_volume": {"participation": .01}, "fifty_ms_window": {"window": .05}}


def verify_run(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["hashes"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen research input changed: {name}")
    selection = json.loads((directory / "selection.json").read_text())
    if digest(directory / "correction.json") != selection["correction_sha256"]:
        raise ValueError("Frozen correction changed")
    return manifest, selection


def run(args):
    output, original, model_dir = args.output_dir, args.frozen_run, args.model_dir.resolve()
    manifest, selection = verify_run(original)
    empty_output(output)
    candidate = json.loads((original / "summary.json").read_text())[selection["candidate"]]
    correction = json.loads((original / "correction.json").read_text())
    start, end = PARTITIONS["chronological_check"]
    hashes = {name: digest(original / name) for name in ("manifest.json", "selection.json", "correction.json", "summary.json")}
    check_manifest = {"started_at": datetime.now(timezone.utc).isoformat(), "source_sha256": digest(Path(__file__)),
        "frozen_run": str(original), "frozen_artifact_hashes": hashes, "candidate": selection["candidate"],
        "scenarios": SCENARIOS, "deployment_ready": False, "role": "previously_inspected_chronological_stress"}
    (output / "manifest.json").write_text(json.dumps(check_manifest, indent=2))
    raw = pd.read_parquet(ROOT / "data/raw/mlb_statcast/2026.parquet", columns=["game_pk", "home_team", "away_team"])
    teams = raw.groupby("game_pk", as_index=False).first()
    ratings = json.loads((model_dir / "prior.json").read_text())["ratings"]
    priors = {g.game_pk: probability(g.home_team, g.away_team, ratings) for g in teams.itertuples(index=False)}
    chunks = [c for c in json.loads((ROOT / "data/reboot/dataset.json").read_text())["chunks"]
              if c[:10] <= str(end) and c[11:] >= str(start)]
    anchors, terminal, results = opening_anchors(model_dir), settlements(), {}
    for name, changes in SCENARIOS.items():
        execution = replace(Execution(**manifest["execution"]), **changes)
        jobs = [(chunk, model_dir / "model.cbm", priors, asdict(execution)) for chunk in chunks]
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            parts = list(pool.map(refreshed_chunk, jobs))
        frame = pd.concat(parts, ignore_index=True).sort_values(["time", "game_pk", "market"]).reset_index(drop=True)
        frame = annotate(frame, anchors)
        prediction = (corrected_probability(frame, correction) if candidate["mode"] == "corrected"
                      else frame.anchor_probability.to_numpy())
        trades, metric = policy_metrics(frame, prediction, candidate["edge"], terminal, execution, start, end)
        trades.to_csv(output / f"{name}.csv", index=False)
        results[name] = metric
        print(f"{name}: {len(trades)} entries, {metric['game_coverage']:.1%} coverage, ${metric['pnl']:.4f}", flush=True)
    verify_run(original)
    if any(digest(original / name) != expected for name, expected in hashes.items()):
        raise RuntimeError("Frozen artifacts changed during stress replay")
    if digest(Path(__file__)) != check_manifest["source_sha256"]:
        raise RuntimeError("Stress source changed during replay")
    (output / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-run", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
