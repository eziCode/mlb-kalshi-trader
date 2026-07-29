"""Authoritative live-account reporting with per-strategy shadow-book detail."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
import sqlite3


@dataclass(frozen=True)
class StrategyBookSnapshot:
    pnl: float
    open_value: float
    open_positions: int


def read_strategy_book(path: str | Path) -> StrategyBookSnapshot | None:
    """Read a strategy book without treating its duplicated seed cash as capital."""
    if not path:
        return None
    database = Path(path)
    if not database.is_file():
        return None
    uri = f"file:{database.resolve()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=5)) as connection:
        row = connection.execute(
            "SELECT starting_cash, cash FROM portfolio WHERE id=1"
        ).fetchone()
        if row is None:
            return None
        position_row = connection.execute(
            "SELECT COALESCE(SUM(contracts * mark_price), 0), COUNT(*) "
            "FROM positions"
        ).fetchone()
    open_value = float(position_row[0])
    return StrategyBookSnapshot(
        pnl=float(row[1]) + open_value - float(row[0]),
        open_value=open_value,
        open_positions=int(position_row[1]),
    )


def format_live_portfolio_summary(
    client,
    *,
    settlement_value_db: str | Path,
    hit_reversion_db: str | Path,
) -> str:
    """Format one account total; strategy books contribute detail, not capital."""
    account = client.balance_snapshot()
    parts = [
        "Portfolio summary: "
        f"total=${account['total']:.2f} cash=${account['cash']:.2f} "
        f"positions=${account['position_value']:.2f}"
    ]
    for name, path in (
        ("settlement_value", settlement_value_db),
        ("hit_reversion", hit_reversion_db),
    ):
        snapshot = read_strategy_book(path)
        if snapshot is None:
            parts.append(f"{name}: unavailable")
        else:
            parts.append(
                f"{name}: PnL=${snapshot.pnl:+.2f} "
                f"open_value=${snapshot.open_value:.2f} "
                f"open_positions={snapshot.open_positions}"
            )
    return " | ".join(parts)
