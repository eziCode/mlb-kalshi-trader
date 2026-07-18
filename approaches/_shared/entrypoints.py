"""Stable wrappers around legacy scripts during the repository migration."""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_script(relative_path: str) -> None:
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(path)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    runpy.run_path(str(path), run_name="__main__")


def run_paper(approach: str, replay_script: str, live_supported: bool = False) -> None:
    mode = os.getenv("PAPER_MODE", "live" if live_supported else "replay")
    if mode == "live":
        if not live_supported:
            raise SystemExit(
                f"{approach} has no live adapter. Set PAPER_MODE=replay to "
                "paper-replay recorded logs."
            )
        run_script("live_trading_engine/paper_trader.py")
        return
    if mode != "replay":
        raise SystemExit("PAPER_MODE must be 'live' or 'replay'")
    run_script(replay_script)
