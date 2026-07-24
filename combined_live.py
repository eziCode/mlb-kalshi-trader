"""Run both real-money strategies behind one Kalshi and one MLB feed."""

from __future__ import annotations

import argparse
from datetime import date, datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import requests


ROOT = Path(__file__).resolve().parent
FEED_URL = "http://127.0.0.1:8765"
MLB_FEED_URL = "http://127.0.0.1:8766"


def stop(processes: list[subprocess.Popen]) -> None:
    for process in reversed(processes):
        if process.poll() is None:
            process.terminate()
    for process in reversed(processes):
        if process.poll() is None:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()


def wait_ready(process: subprocess.Popen, url: str, name: str) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{name} exited with {process.returncode}")
        try:
            if requests.get(f"{url}/health", timeout=1).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(.25)
    raise RuntimeError(f"{name} did not become ready")


def run(selected: date | None) -> int:
    state = Path(os.getenv("LIVE_STATE_DIR", "/app/live-state"))
    state.mkdir(parents=True, exist_ok=True)
    state_date = selected or datetime.now(
        ZoneInfo(os.getenv("SLATE_TIMEZONE", "America/Chicago"))
    ).date()
    common = os.environ.copy()
    common.update({
        "KALSHI_FEED_URL": FEED_URL,
        "MLB_FEED_URL": MLB_FEED_URL,
        "KALSHI_FEED_BIND": "127.0.0.1",
        "MLB_FEED_BIND": "127.0.0.1",
        "PYTHONUNBUFFERED": "1",
        "PAPER_LOG_DIR": os.getenv("PAPER_LOG_DIR", "/app/live-logs"),
    })
    settlement = common.copy()
    settlement["PAPER_PORTFOLIO_DB"] = str(
        state / f"settlement_value_portfolio_{state_date}.sqlite3"
    )
    hit = common.copy()
    hit.update({
        "ALLOW_UNVALIDATED_HYBRID": "1",
        "SUPPRESS_SLATE_SUMMARY": "1",
        "PAPER_PORTFOLIO_DB": str(
            state / "hit-reversion" / f"hit_reversion_portfolio_{state_date}.sqlite3"
        ),
    })
    processes: list[subprocess.Popen] = []
    try:
        kalshi = subprocess.Popen(
            [sys.executable, "-u", "-m", "shared_kalshi_feed", "serve"],
            cwd=ROOT, env=common,
        )
        processes.append(kalshi)
        wait_ready(kalshi, FEED_URL, "Kalshi feed")
        mlb = subprocess.Popen(
            [sys.executable, "-u", "-m", "shared_mlb_feed"],
            cwd=ROOT, env=common,
        )
        processes.append(mlb)
        wait_ready(mlb, MLB_FEED_URL, "MLB feed")
        date_args = [] if selected is None else ["--date", selected.isoformat()]
        settlement_worker = subprocess.Popen(
            [sys.executable, "-u", "-m",
             "settlement_value_strategy.live_paper_trader", "--continuous",
             *date_args], cwd=ROOT, env=settlement,
        )
        hit_worker = subprocess.Popen(
            [sys.executable, "-u", "scripts/paper_trade.py", "--continuous",
             *date_args], cwd=ROOT / "hit_reversion_strategy", env=hit,
        )
        processes.extend([settlement_worker, hit_worker])
        print(
            "Combined LIVE runtime started: one Kalshi WebSocket, one "
            "adaptive MLB feed, settlement-value + hit-reversion",
            flush=True,
        )
        while True:
            for name, process in (
                ("Kalshi feed", kalshi), ("MLB feed", mlb),
                ("settlement-value", settlement_worker),
                ("hit-reversion", hit_worker),
            ):
                code = process.poll()
                if code is not None:
                    print(f"{name} exited with status {code}", flush=True)
                    return code or 1
            time.sleep(1)
    finally:
        stop(processes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat)
    args = parser.parse_args()
    raise SystemExit(run(args.date))


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(143)))
    main()
