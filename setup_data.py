"""Download and process every input required by both trading strategies."""

from __future__ import annotations

import argparse
from datetime import date
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
DOWNLOADERS = ROOT / "data/download_scripts"
PROCESSORS = ROOT / "data/processing_scripts"


def run_step(
    number: int, total: int, label: str, command: list[str], *, dry_run: bool,
) -> None:
    print("\n" + "=" * 78, flush=True)
    print(f"[{number}/{total}] {label}", flush=True)
    print("$ " + " ".join(command), flush=True)
    print("=" * 78, flush=True)
    if dry_run:
        return
    started = time.monotonic()
    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    elapsed = time.monotonic() - started
    print(f"Completed: {label} ({elapsed / 60:.1f} minutes)", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "strategy", choices=("mispricing", "trade-tape", "both"),
        help="Strategy whose datasets should be prepared.",
    )
    parser.add_argument(
        "--start-date", type=date.fromisoformat, default=date(2025, 4, 16),
        help="First Kalshi MLB game date to download (default: 2025-04-16).",
    )
    parser.add_argument(
        "--end-date", type=date.fromisoformat, default=date.today(),
        help="Last Kalshi MLB game date to download (default: today).",
    )
    parser.add_argument(
        "--skip-downloads", action="store_true",
        help="Reprocess files already under data/raw without network calls.",
    )
    parser.add_argument(
        "--skip-mlb-downloads", action="store_true",
        help="Reuse existing Statcast and MLB feeds but continue Kalshi download.",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Ignore reusable MLB/Kalshi caches and download again.",
    )
    parser.add_argument(
        "--max-games", type=int,
        help="Limit MLB feeds and Kalshi events for a smoke test.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print every stage and command without executing them.",
    )
    args = parser.parse_args()
    if args.start_date > args.end_date:
        parser.error("--start-date must not be after --end-date")
    if args.max_games is not None and args.max_games < 1:
        parser.error("--max-games must be positive")
    return args


def main() -> None:
    args = parse_args()
    python = sys.executable
    steps: list[tuple[str, list[str]]] = []
    if not args.skip_downloads and not args.skip_mlb_downloads:
        steps.append((
            "Download Statcast pitch data",
            [python, str(DOWNLOADERS / "download_mlb_statcast.py")],
        ))
        timestamp_command = [
            python, str(DOWNLOADERS / "download_mlb_pitch_timestamps.py")
        ]
        if args.max_games is not None:
            timestamp_command += ["--max-games", str(args.max_games)]
        if args.refresh:
            timestamp_command.append("--refresh")
        steps.append(("Download authoritative MLB live feeds", timestamp_command))
    steps.append((
        "Build causal MLB pitch-state features",
        [python, str(PROCESSORS / "build_event_state_features.py")],
    ))
    if not args.skip_downloads:
        trade_command = [
            python, str(DOWNLOADERS / "download_live_kalshi_market_logs.py"),
            "--start-date", args.start_date.isoformat(),
            "--end-date", args.end_date.isoformat(),
        ]
        if args.max_games is not None:
            trade_command += ["--max-games", str(args.max_games)]
        if args.refresh:
            trade_command += ["--refresh-markets", "--refresh-trades"]
        steps.append(("Download exact Kalshi MLB executions", trade_command))
    shared_command = [python, str(PROCESSORS / "build_shared_data.py")]
    if args.strategy in {"mispricing", "both"}:
        shared_command += [
            "--settlement-model-train-end", "2026-06-17",
            "--settlement-model-output",
            str(ROOT / "settlement_value_strategy/model/local_win_expectancy.cbm"),
            "--settlement-state-output",
            str(ROOT / "data/settlement_value/state_updates.parquet"),
        ]
    steps.append(("Build shared strategy inputs", shared_command))
    if args.strategy in {"mispricing", "both"}:
        steps.append((
            "Build mispricing decisions and compact execution tape",
            [python, "-m", "settlement_value_strategy.prepare_data"],
        ))
    print(
        f"Preparing {args.strategy} data from {args.start_date} "
        f"through {args.end_date}",
        flush=True,
    )
    for number, (label, command) in enumerate(steps, 1):
        run_step(number, len(steps), label, command, dry_run=args.dry_run)
    if args.dry_run:
        print("\nDry run complete; no files were changed.", flush=True)
        return
    print("\nShared data setup complete:", flush=True)
    print(f"  {ROOT / 'data/shared/home_market_trades.parquet'}", flush=True)
    print(f"  {ROOT / 'data/shared/away_market_trades.parquet'}", flush=True)
    print(f"  {ROOT / 'data/shared/state_updates.parquet'}", flush=True)
    if args.strategy in {"mispricing", "both"}:
        print(f"  {ROOT / 'data/settlement_value/decision_rows.parquet'}", flush=True)
        print(f"  {ROOT / 'data/settlement_value/execution_trades.parquet'}", flush=True)
        print(f"  {ROOT / 'data/settlement_value/away_execution_trades.parquet'}", flush=True)


if __name__ == "__main__":
    main()
