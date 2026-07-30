"""Slate-scoped strategy ledger paths for continuous live trading."""

from __future__ import annotations

from datetime import date
import os
from pathlib import Path


def settlement_value_path(game_date: date, fallback_dir: Path) -> Path:
    explicit = os.getenv("PAPER_PORTFOLIO_DB")
    if explicit:
        return Path(explicit)
    directory = Path(os.getenv(
        "SETTLEMENT_VALUE_PORTFOLIO_DIR",
        os.getenv("PAPER_PORTFOLIO_DIR", str(fallback_dir)),
    ))
    return directory / f"settlement_value_portfolio_{game_date}.sqlite3"


def hit_reversion_path(game_date: date, fallback_dir: Path) -> Path:
    explicit = os.getenv("PAPER_PORTFOLIO_DB")
    if explicit:
        return Path(explicit)
    directory = Path(os.getenv(
        "HIT_REVERSION_PORTFOLIO_DIR",
        os.getenv("PAPER_PORTFOLIO_DIR", str(fallback_dir)),
    ))
    return directory / f"hit_reversion_portfolio_{game_date}.sqlite3"
