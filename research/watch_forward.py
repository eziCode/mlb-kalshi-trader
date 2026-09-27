"""Periodically freeze and score an active read-only capture without retuning."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from research.slate_study import digest, empty_output, snapshot
from research.state_refresh import ROOT


def run(args):
    empty_output(args.output_dir)
    files = [ROOT / "research" / name for name in ("watch_forward.py", "forward_value.py", "market_check.py",
        "market_correction.py", "maker_portfolio.py", "maker_study.py", "passive.py", "record.py",
        "slate_study.py", "state_refresh.py", "FORWARD_VALUE_PROTOCOL.md", "PASSIVE_SLATE_PROTOCOL.md")]
    hashes = {str(p): digest(p) for p in files}
    (args.output_dir / "manifest.json").write_text(json.dumps({"started_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.capture.resolve()), "source_hashes": hashes, "interval_seconds": args.interval,
        "orders_enabled": False, "automatic_retuning": False}, indent=2))
    deadline, count = time.monotonic() + args.duration, 0
    while time.monotonic() < deadline:
        if any(digest(Path(p)) != expected for p, expected in hashes.items()):
            raise RuntimeError("Watched research source changed; start a separate run after review")
        count += 1
        directory = args.output_dir / f"prefix_{count:03d}"
        capture, metadata = snapshot(args.capture, directory)
        summaries = {}
        for latency in (.68, 1.5, 3.):
            output = directory / f"value_latency_{latency}"
            with (directory / f"value_latency_{latency}.log").open("w") as log:
                subprocess.run([sys.executable, "-m", "research.forward_value", str(capture), "--slate", str(args.slate),
                    "--model-dir", str(args.model_dir), "--frozen-run", str(args.frozen_run), "--output-dir", str(output),
                    "--latency", str(latency)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            summaries[str(latency)] = json.loads((output / "summary.json").read_text())
        if args.passive:
            with (directory / "passive.log").open("w") as log:
                subprocess.run([sys.executable, "-m", "research.slate_study", "evaluate", str(capture),
                    "--slate", str(args.slate), "--output-dir", str(directory / "passive"), "--shared-cash", "100"],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        brief = {"prefix": str(directory), "records": metadata["complete_records"],
                 "capture_complete": metadata["capture_complete"], "value": {
                     key: {"entries": r["entry_orders_filled"], "live_games": r["games_observed_live"],
                           "final_games": r["games_observed_final"], "marked_pnl": r["portfolio"]["liquidation_marked_pnl"],
                           "valid": r["all_replays_valid"]} for key, r in summaries.items()}}
        print(json.dumps(brief), flush=True)
        if metadata["capture_complete"] or not all(r["all_replays_valid"] for r in summaries.values()):
            break
        next_run = min(time.monotonic() + args.interval, deadline)
        while time.monotonic() < next_run:
            time.sleep(min(30., max(0., next_run - time.monotonic())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--slate", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--frozen-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=900.)
    parser.add_argument("--duration", type=float, default=28800.)
    parser.add_argument("--passive", action="store_true")
    args = parser.parse_args()
    if not 60 <= args.interval <= 3600 or not 60 <= args.duration <= 86400:
        parser.error("interval must be 60–3600 seconds and duration 60–86400 seconds")
    run(args)


if __name__ == "__main__":
    main()
