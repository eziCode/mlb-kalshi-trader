"""Prepare and evaluate newer data with frozen models and explicit execution assumptions.

Run from the repository root. This module never submits orders or retrains models.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from dataclasses import asdict, fields, replace
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hit_reversion_strategy"))
from scripts.backtest import (  # noqa: E402
    apply_live_paired_execution_prices, apply_publication_latency,
    TRADE_COLUMNS, AWAY_TRADE_COLUMNS, STATE_COLUMNS,
)
from trade_tape_strategy.core import TradeTapeConfig, TapeTradeRecord, simulate_trade_tape
from trade_tape_strategy.reversion_value import CompetingRisksModel
from trade_tape_strategy.strategy import taker_fee
from research.validation import constrain_cash, summarize

MODELS = ROOT / "hit_reversion_strategy/models"
DEFAULT_DATA = ROOT / "data/reboot"
DEFAULT_OUTPUT = ROOT / "research/results/reboot"
# Fixed sensitivity grid, reported in full. No parameter search or winner selection.
SCENARIOS = {
    "bounded_reference": {},
    "latency_1_5s": {"entry_submission_latency_seconds": 1.5, "exit_submission_latency_seconds": 1.5},
    "latency_3s": {"entry_submission_latency_seconds": 3., "exit_submission_latency_seconds": 3.},
    "two_cent_cost": {"execution_price_penalty": .02},
    "one_percent_volume": {"maximum_volume_participation": .01},
    "fifty_ms_window": {"execution_window_seconds": .05},
    "publication_plus_2s": {"publication_delay": 2.},
}


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def run(command):
    print(" ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=ROOT, check=True)


def prepare(args):
    raw = ROOT / "data/raw/reboot_kalshi"
    if not args.skip_downloads:
        run([sys.executable, ROOT / "data/download_scripts/download_mlb_statcast.py",
             "--start-date", args.start_date, "--end-date", args.end_date])
        run([sys.executable, ROOT / "data/download_scripts/download_mlb_pitch_timestamps.py",
             "--seasons", *range(args.start_date.year, args.end_date.year + 1),
             "--start-date", args.start_date, "--end-date", args.end_date])
        run([sys.executable, ROOT / "data/download_scripts/download_live_kalshi_market_logs.py",
             "--start-date", args.start_date, "--end-date", args.end_date, "--output-dir", raw,
             "--workers", args.workers])
    manifests = [json.loads(path.read_text()) for path in raw.glob("acquisition_*.json")]
    covered = [item for item in manifests if item.get("complete") and not item.get("smoke_test")
               and date.fromisoformat(item["start_date"]) <= args.start_date
               and date.fromisoformat(item["end_date"]) >= args.end_date]
    if not covered:
        raise RuntimeError("A complete acquisition manifest covering this window is required")
    run([sys.executable, ROOT / "data/processing_scripts/build_event_state_features.py"])
    args.data_dir.mkdir(parents=True, exist_ok=True)
    chunks = []
    first = args.start_date
    while first <= args.end_date:
        last = min(first + timedelta(days=6), args.end_date)
        destination = args.data_dir / f"{first}_{last}"
        run([sys.executable, ROOT / "data/processing_scripts/build_shared_data.py",
             "--trade-dir", raw, "--output-dir", destination,
             "--pregame-prior-state", MODELS / "mlb_pregame_prior.json",
             "--prior-state-as-of", "2026-07-20",
             "--start-date", first, "--end-date", last])
        chunks.append(str(destination.relative_to(args.data_dir)))
        first = last + timedelta(days=1)
    (args.data_dir / "dataset.json").write_text(json.dumps({
        "start_date": str(args.start_date), "end_date": str(args.end_date),
        "chunks": chunks, "acquisition": covered,
    }, indent=2))


def evaluate_chunk(job):
    args, chunk, config_dict = job
    config = TradeTapeConfig(**config_dict)
    scorer = CompetingRisksModel(MODELS, MODELS / "competing_risks.metadata.json")
    records = {name: [] for name in SCENARIOS}
    counters = {name: {"expired_entry_orders": 0, "expired_exit_orders": 0, "confirmed_signals": 0} for name in SCENARIOS}
    game_dates, file_hashes = {}, {}
    directory = args.data_dir / chunk
    metadata = json.loads((directory / "state_updates.metadata.json").read_text())
    coverage = metadata.get("coverage", {})
    if metadata.get("state_contract") != "atomic_pitch_or_play_v2" or metadata.get("prior_source") != "frozen_mlb_ratings":
        raise ValueError(f"Incompatible state provenance: {directory}")
    if metadata["model_sha256"] != digest(MODELS / "local_win_expectancy.cbm") or metadata["prior_sha256"] != digest(MODELS / "mlb_pregame_prior.json"):
        raise ValueError("State data was built with different frozen model inputs")
    for path in directory.glob("*.parquet"):
        file_hashes[str(path.relative_to(args.data_dir))] = digest(path)
    filters = [("game_date", ">=", args.start_date), ("game_date", "<=", args.end_date)]
    home = pd.read_parquet(directory / "home_market_trades.parquet", columns=TRADE_COLUMNS, filters=filters)
    away = pd.read_parquet(directory / "away_market_trades.parquet", columns=AWAY_TRADE_COLUMNS, filters=filters)
    updates = pd.read_parquet(directory / "state_updates.parquet", columns=STATE_COLUMNS, filters=filters)
    if home.empty:
        return chunk, records, counters, game_dates, file_hashes, coverage
    if not set(home.game_pk).issubset(set(updates.game_pk)):
        raise ValueError("Home trade tape contains games without state updates")
    new_dates = home.groupby("game_pk").game_date.first().to_dict()
    if set(new_dates) & set(game_dates):
        raise ValueError("Overlapping games across dataset chunks")
    game_dates.update(new_dates)
    tape = apply_live_paired_execution_prices(home, away)
    delayed = apply_publication_latency(updates)
    for name, changes in SCENARIOS.items():
        scenario = replace(config, **{key: value for key, value in changes.items() if key != "publication_delay"})
        state = delayed.copy()
        state["event_available_time"] += pd.to_timedelta(changes.get("publication_delay", 0), unit="s")
        result = simulate_trade_tape(tape, state, scenario, scorer)
        records[name].extend(asdict(row) for row in result.records)
        for key in counters[name]:
            counters[name][key] += getattr(result, key)
        print(f"{chunk} {name}: {result.trades} fills, proxy PnL ${result.pnl:.2f}", flush=True)
    return chunk, records, counters, game_dates, file_hashes, coverage


def evaluate(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "summary.json").exists():
        raise RuntimeError("Choose a new --output-dir; completed evaluations are not overwritten")
    dataset_path = args.data_dir / "dataset.json"
    dataset = json.loads(dataset_path.read_text())
    if args.start_date < date.fromisoformat(dataset["start_date"]) or args.end_date > date.fromisoformat(dataset["end_date"]):
        raise ValueError("Requested evaluation extends beyond dataset coverage")
    config_path = MODELS / "trade_tape_config.json"
    config = TradeTapeConfig(**json.loads(config_path.read_text()))
    config = replace(config, execution_window_seconds=.25, execution_price_penalty=.01,
                     maximum_volume_participation=.1, require_compatible_taker=True, require_post_signal_trade=True)
    if config.direct_value_model_enabled or not config.competing_risks_enabled:
        raise ValueError("This frozen protocol expects the packaged competing-risks policy")
    local_metadata = json.loads((MODELS / "local_win_expectancy.metadata.json").read_text())
    gate_metadata = json.loads((MODELS / "competing_risks.metadata.json").read_text())
    if str(args.start_date) < gate_metadata["fit_end_exclusive"] or str(args.start_date) < local_metadata["training_cutoff_exclusive"]:
        raise ValueError("Evaluation overlaps model training")
    if local_metadata["model_sha256"] != digest(MODELS / "local_win_expectancy.cbm"):
        raise ValueError("Local state model hash does not match its training metadata")
    model_files = [config_path, MODELS / "local_win_expectancy.cbm", MODELS / "local_win_expectancy.metadata.json",
                   MODELS / "competing_risks.metadata.json", MODELS / "event_observation_latency.json",
                   MODELS / "mlb_pregame_prior.json", *sorted(MODELS.glob("competing_risks_*.cbm"))]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in model_files}
    source_hashes = {str(path.relative_to(ROOT)): digest(path) for path in [Path(__file__), ROOT / "research/validation.py",
        ROOT / "hit_reversion_strategy/trade_tape_strategy/core.py", ROOT / "hit_reversion_strategy/scripts/backtest.py",
        ROOT / "data/processing_scripts/build_shared_data.py", ROOT / "settlement_value_strategy/play_eligibility.py",
        ROOT / "hit_reversion_strategy/trade_tape_strategy/strategy.py", ROOT / "hit_reversion_strategy/trade_tape_strategy/hybrid.py",
        ROOT / "hit_reversion_strategy/trade_tape_strategy/reversion_value.py"]}
    manifest = {"started_at": datetime.now(timezone.utc).isoformat(), "start_date": str(args.start_date),
        "end_date": str(args.end_date), "starting_cash": args.starting_cash, "base_config": asdict(config),
        "scenarios": SCENARIOS, "model_hashes": hashes, "source_hashes": source_hashes,
        "dataset_sha256": digest(dataset_path), "dataset_files": {}, "coverage": {}}
    # Written before any outcomes are scored. Not a claim of external preregistration.
    manifest_path = args.output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    records = {name: [] for name in SCENARIOS}
    counters = {name: {"expired_entry_orders": 0, "expired_exit_orders": 0, "confirmed_signals": 0} for name in SCENARIOS}
    game_dates = {}
    jobs = [(args, chunk, asdict(config)) for chunk in dataset["chunks"]]
    # Spawn avoids inheriting CatBoost/Arrow native thread state on macOS.
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        for chunk, chunk_records, chunk_counters, new_dates, file_hashes, coverage in executor.map(evaluate_chunk, jobs):
            if set(new_dates) & set(game_dates):
                raise ValueError("Overlapping games across dataset chunks")
            game_dates.update(new_dates)
            manifest["dataset_files"].update(file_hashes)
            manifest["coverage"][chunk] = coverage
            for name in SCENARIOS:
                records[name].extend(chunk_records[name])
                for key in counters[name]:
                    counters[name][key] += chunk_counters[name][key]
    if not game_dates:
        raise ValueError("No games in the requested evaluation")
    if hashes != {str(path.relative_to(ROOT)): digest(path) for path in model_files}:
        raise RuntimeError("Frozen model inputs changed during evaluation")
    if source_hashes != {name: digest(ROOT / name) for name in source_hashes}:
        raise RuntimeError("Replay source changed during evaluation")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    report = {"status": "execution_proxy_diagnostic_not_live_validation", "deployment_ready": False,
        "start_date": str(args.start_date), "end_date": str(args.end_date), "scenarios": {},
        "limitations": ["No historical quotes, depth, queue position, or actual IOC fills",
            "Final MLB archives may include later official corrections; reception timestamps are modeled",
            "Publication-delay profile is from one July sample; fixed costs/windows are stress assumptions",
            "Earlier development and live monitoring mean later dates are not automatically a blind holdout",
            "Settlement cash is withheld throughout the replay; partial exits release cash only at final exit",
            "Cash admission filters proposed fills; pending-order reservations and replacement signals after cash rejection are not simulated",
            "Positive proxy PnL never enables deployment"]}
    for name, rows in records.items():
        frame = pd.DataFrame(rows, columns=[field.name for field in fields(TapeTradeRecord)])
        for column in ("pnl", "fees", "contracts", "entry_price"):
            frame[column] = pd.to_numeric(frame[column])
        selected, cash = constrain_cash(frame, args.starting_cash, taker_fee)
        selected.to_csv(args.output_dir / f"{name}_trades.csv", index=False)
        report["scenarios"][name] = {**summarize(selected, game_dates, args.start_date, args.end_date),
            **cash, **counters[name], "unconstrained_trades": len(frame), "unconstrained_proxy_pnl": float(frame.pnl.sum())}
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    lines = ["# Frozen-policy later-period diagnostic", "", f"{args.start_date} through {args.end_date}; ${args.starting_cash:.2f} starting cash.", "",
             "Historical executions are a fill proxy. This report does not authorize deployment.", "",
             "| Scenario | Fills | Net proxy PnL | Day-bootstrap 95% interval |", "| --- | ---: | ---: | ---: |"]
    for name, result in report["scenarios"].items():
        low, high = result["day_block_bootstrap_pnl_95pct"]
        lines.append(f"| {name} | {result['trades']} | ${result['pnl']:.2f} | ${low:.2f} to ${high:.2f} |")
    lines.extend(["", *[f"- {item}" for item in report["limitations"]], ""])
    (args.output_dir / "REPORT.md").write_text("\n".join(lines))
    print(f"Report: {args.output_dir / 'REPORT.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "evaluate"])
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2026, 8, 12))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date(2026, 9, 26))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-downloads", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--starting-cash", type=float, default=100.)
    args = parser.parse_args()
    if args.start_date > args.end_date or args.end_date >= date.today():
        parser.error("Use an ordered date range ending before today")
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8")
    if not math.isfinite(args.starting_cash) or args.starting_cash <= 0:
        parser.error("--starting-cash must be positive and finite")
    (prepare if args.action == "prepare" else evaluate)(args)


if __name__ == "__main__":
    main()
