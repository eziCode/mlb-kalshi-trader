"""Cash accounting and uncertainty for chronological strategy diagnostics."""
from __future__ import annotations

import heapq
import math

import numpy as np
import pandas as pd


def constrain_cash(records: pd.DataFrame, starting_cash: float, fee) -> tuple[pd.DataFrame, dict]:
    """Admit whole proposed orders in time order without borrowing cash.

    Partial-sale proceeds are conservatively withheld until the position's
    final exit. Settlement cash is never recycled during the evaluation:
    archived MLB outcomes do not tell us when the exchange paid the account.
    Equal-time proceeds cannot fund an entry at that same timestamp.
    """
    if not math.isfinite(starting_cash) or starting_cash <= 0:
        raise ValueError("starting_cash must be positive and finite")
    cash = starting_cash
    releases = []
    accepted = []
    peak_locked = 0.0
    locked = 0.0
    for index, row in records.sort_values(["entry_time", "game_pk", "side"]).iterrows():
        when = pd.Timestamp(row.entry_time).value
        while releases and releases[0][0] < when:
            _, _, proceeds, principal = heapq.heappop(releases)
            cash += proceeds
            locked -= principal
        cost = float(row.contracts * row.entry_price + fee(row.contracts, row.entry_price))
        if cost > cash + 1e-9:
            continue
        cash -= cost
        locked += cost
        peak_locked = max(peak_locked, locked)
        accepted.append(index)
        if pd.notna(row.exit_time):
            heapq.heappush(releases, (pd.Timestamp(row.exit_time).value, len(accepted), cost + row.pnl, cost))
    selected = records.loc[accepted].copy()
    pnl = float(selected.pnl.sum())
    return selected, {
        "starting_cash": starting_cash, "ending_equity": starting_cash + pnl,
        "account_return": pnl / starting_cash,
        "cash_rejected_orders": len(records) - len(selected),
        "peak_locked_capital": peak_locked,
        "settlement_cash_reused": False,
        "partial_exit_cash_reused_before_final_exit": False,
        "cash_accounting_scope": "Chronological admission of proposed fills; excludes pending-order reservations and does not regenerate signals after rejection",
    }


def summarize(records: pd.DataFrame, game_dates: dict, start_date, end_date) -> dict:
    """Bootstrap whole calendar days, including zero-trade days, not trades."""
    days = pd.date_range(start_date, end_date, freq="D")
    if len(days) == 0:
        raise ValueError("Empty evaluation window")
    dated = records.copy()
    dated["game_date"] = pd.to_datetime(dated.game_pk.map(game_dates))
    daily = dated.groupby("game_date").pnl.sum().reindex(days, fill_value=0).astype(float)
    values = daily.to_numpy()
    rng = np.random.default_rng(20260927)
    samples = rng.choice(values, size=(10000, len(values)), replace=True).sum(axis=1)
    interval = np.quantile(samples, [.025, .975])
    games = dated.groupby("game_pk").pnl.sum().astype(float)
    # This is settlement/outcome attribution by game day, not an intraday
    # marked-to-market equity curve.
    cumulative = np.r_[0., np.cumsum(values)]
    drawdown = float((np.maximum.accumulate(cumulative) - cumulative).max())
    pnl = float(records.pnl.sum())
    return {
        "trades": len(records), "games_with_trades": int(len(games)),
        "games_in_data": len(game_dates), "calendar_days": len(days),
        "days_with_trades": int(dated.game_date.nunique()),
        "pnl": pnl, "fees": float(records.fees.sum()),
        "day_block_bootstrap_pnl_95pct": interval.tolist(),
        "bootstrap_interpretation": "Descriptive day-resampling uncertainty; excludes model selection and fill-proxy error",
        "pnl_without_best_game": pnl - float(games.nlargest(1).sum()),
        "pnl_without_best_four_games": pnl - float(games.nlargest(4).sum()),
        "worst_game": float(games.min()) if len(games) else 0.0,
        "outcome_attributed_daily_drawdown": drawdown,
        "positive_lower_bound": bool(len(records) and interval[0] > 0),
        "daily_pnl": {str(day.date()): float(value) for day, value in daily.items()},
    }
