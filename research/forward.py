"""Discover a daily slate, record it read-only, and run passive shadow replays.

python -m research.forward --date 2026-09-27 --duration-seconds 14400
Authentication is used by the recorder only. No orders are submitted.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date, datetime
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

from research.passive import replay

ROOT = Path(__file__).resolve().parents[1]


def discover(day):
    # Reuse the tested schedule/team/doubleheader matching. Importing this
    # module does not launch its coordinator, executor, or worker processes.
    for directory in (ROOT / "hit_reversion_strategy", ROOT / "hit_reversion_strategy/scripts"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    from scripts.paper_trade import discover_daily_games
    games, warnings = discover_daily_games(day)
    rows = [{**asdict(game), "scheduled_time": game.scheduled_time.isoformat()} for game in games]
    return {"date": str(day), "games": rows, "warnings": warnings, "orders_enabled": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat,
                        default=datetime.now(ZoneInfo("America/New_York")).date())
    parser.add_argument("--duration-seconds", type=float, default=14400.)
    parser.add_argument("--poll-seconds", type=float, default=2.)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/raw/forward_observations")
    parser.add_argument("--discover-only", action="store_true")
    args = parser.parse_args()
    if not 0 < args.duration_seconds <= 86400 or not .25 <= args.poll_seconds <= 60:
        parser.error("Invalid duration or poll interval")
    slate = discover(args.date)
    print(json.dumps(slate, indent=2), flush=True)
    if args.discover_only or not slate["games"]:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "research.record", "--duration-seconds", str(args.duration_seconds),
           "--poll-seconds", str(args.poll_seconds), "--output-dir", str(args.output_dir)]
    for game in slate["games"]:
        cmd += ["--game-pk", str(game["game_pk"]), "--ticker", game["market_ticker"],
                "--ticker", game["away_market_ticker"]]
    print(f"Recording {len(slate['games'])} games for {args.duration_seconds:g} seconds; no orders.", flush=True)
    completed = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    capture_path = Path(completed.stdout.strip().splitlines()[-1])
    result_dir = capture_path.parent / f"{capture_path.name.removesuffix('.jsonl.gz')}_shadow"
    result_dir.mkdir(exist_ok=False)
    (result_dir / "slate.json").write_text(json.dumps(slate, indent=2))
    summaries = []
    for game in slate["games"]:
        # One home market per game; its YES/NO sides share one inventory.
        # Individual shadow accounts are never represented as one portfolio.
        result, fills = replay(capture_path, game["market_ticker"])
        result["game_pk"] = game["game_pk"]
        (result_dir / f"{game['game_pk']}.json").write_text(json.dumps({**result, "fills": fills}, indent=2))
        summaries.append({k: result[k] for k in ("game_pk", "ticker", "shadow_fill_events", "entry_contracts",
            "inventory", "liquidation_marked_pnl", "full_replay_valid", "continuity_error")})
    (result_dir / "summary.json").write_text(json.dumps({"games": summaries,
        "account_scope": "Independent per-game diagnostics; no combined portfolio return",
        "deployment_ready": False}, indent=2))
    print(f"Capture: {capture_path}\nShadow reports: {result_dir}", flush=True)


if __name__ == "__main__":
    main()
