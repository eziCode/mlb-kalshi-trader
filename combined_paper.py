"""Run both paper strategies behind one shared Kalshi WebSocket."""

from __future__ import annotations

import argparse
from datetime import date
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import requests


ROOT = Path(__file__).resolve().parent
FEED_URL = "http://127.0.0.1:8765"


def _stop(processes: list[subprocess.Popen]) -> None:
    for process in reversed(processes):
        if process.poll() is None:
            process.terminate()
    for process in reversed(processes):
        if process.poll() is None:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()


def _wait_for_feed(process: subprocess.Popen) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Shared Kalshi feed exited with {process.returncode}")
        try:
            if requests.get(f"{FEED_URL}/health", timeout=1).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(0.25)
    raise RuntimeError("Shared Kalshi feed did not become ready")


def run(game_date: date | None) -> int:
    state_root = Path(os.getenv("PAPER_STATE_DIR", "/app/state"))
    settlement_dir = state_root / "settlement-value"
    hit_dir = state_root / "hit-reversion"
    settlement_dir.mkdir(parents=True, exist_ok=True)
    hit_dir.mkdir(parents=True, exist_ok=True)
    selected = game_date or date.today()

    common = os.environ.copy()
    common["KALSHI_FEED_URL"] = FEED_URL
    common["PYTHONUNBUFFERED"] = "1"
    settlement_env = common.copy()
    settlement_env.update({
        "ALLOW_UNVALIDATED_MISPRICING": "1",
        "PAPER_PORTFOLIO_DB": os.getenv(
            "SETTLEMENT_VALUE_PORTFOLIO_DB",
            str(settlement_dir / f"settlement_value_portfolio_{selected}.sqlite3"),
        ),
    })
    hit_env = common.copy()
    hit_env.update({
        "ALLOW_UNVALIDATED_HYBRID": "1",
        "PAPER_PORTFOLIO_DB": os.getenv(
            "HIT_REVERSION_PORTFOLIO_DB",
            str(hit_dir / f"hit_reversion_portfolio_{selected}.sqlite3"),
        ),
    })

    processes: list[subprocess.Popen] = []
    try:
        feed = subprocess.Popen(
            [sys.executable, "-u", "-m", "shared_kalshi_feed", "serve"],
            cwd=ROOT, env=common,
        )
        processes.append(feed)
        _wait_for_feed(feed)
        date_args = ["--date", selected.isoformat()]
        settlement = subprocess.Popen(
            [sys.executable, "-u", "-m",
             "settlement_value_strategy.live_paper_trader", "--continuous",
             *date_args],
            cwd=ROOT, env=settlement_env,
        )
        hit = subprocess.Popen(
            [sys.executable, "-u", "scripts/paper_trade.py", "--continuous",
             *date_args],
            cwd=ROOT / "hit_reversion_strategy", env=hit_env,
        )
        processes.extend([settlement, hit])
        print(
            "Combined paper runtime started: one Kalshi WebSocket, "
            "settlement-value + hit-reversion",
            flush=True,
        )
        while True:
            for name, process in (
                ("shared feed", feed), ("settlement-value", settlement),
                ("hit-reversion", hit),
            ):
                code = process.poll()
                if code is not None:
                    print(f"{name} exited with status {code}", flush=True)
                    return code or 1
            time.sleep(1)
    finally:
        _stop(processes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, default=None)
    args = parser.parse_args()
    raise SystemExit(run(args.date))


def raise_system_exit() -> None:
    raise SystemExit(143)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: raise_system_exit())
    main()
