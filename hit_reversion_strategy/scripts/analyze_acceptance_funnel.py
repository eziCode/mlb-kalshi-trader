"""Diagnose where causal hit-reversion opportunities leave the entry funnel."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.backtest import AWAY_TRADE_COLUMNS, STATE_COLUMNS, TRADE_COLUMNS
from scripts.research_competing_risks import (
    CORE_EDGE, DEVELOPMENT_END, FIT_END, MODEL_DIR, OUTCOMES,
    RESEARCH_FEATURES, feature_pool, label_outcomes, predict,
)
from scripts.train_reversion_model import CONFIG_PATH, DATA, build_opportunities
from trade_tape_strategy.core import TradeTapeConfig
from trade_tape_strategy.reversion_value import CompetingRisksModel


def main() -> None:
    config = TradeTapeConfig(**json.loads(CONFIG_PATH.read_text()))
    home = pd.read_parquet(DATA / "home_market_trades.parquet", columns=TRADE_COLUMNS)
    away = pd.read_parquet(DATA / "away_market_trades.parquet", columns=AWAY_TRADE_COLUMNS)
    states = pd.read_parquet(DATA / "state_updates.parquet", columns=STATE_COLUMNS)
    for frame in (home, away, states):
        frame["game_date"] = pd.to_datetime(frame.game_date).dt.date
    rows = label_outcomes(build_opportunities(home, away, states, config))
    rows = rows[rows.game_date >= FIT_END].copy()
    loaded = CompetingRisksModel(
        MODEL_DIR, MODEL_DIR / "competing_risks.metadata.json"
    )
    scores = predict(loaded.models, rows)
    gates = pd.DataFrame({
        "positive_lower_bound": scores.expected_pnl_lower_bound.gt(0),
        "reversion_over_half": scores.reversion_probability.gt(.5),
        "profit_over_half": scores.profitability_probability.gt(.5),
        "positive_downside": scores.downside_quartile_pnl.gt(0),
    })
    structural = (
        gates.positive_lower_bound & gates.reversion_over_half
        & gates.profit_over_half & scores.severe_loss_probability.lt(.25)
    )
    core = rows.entry_net_edge.ge(CORE_EDGE)
    expansion_pool = ~core & rows.entry_net_edge.ge(.01)
    expansion_accept = expansion_pool & gates.all(axis=1)
    report = {
        "unique_opportunity_rows": int(len(rows)),
        "core_rows": int(core.sum()),
        "expansion_pool_rows": int(expansion_pool.sum()),
        "expansion_accepted_before_cooldown": int(expansion_accept.sum()),
        "gate_pass_rates_in_expansion_pool": {
            name: float(gates.loc[expansion_pool, name].mean())
            for name in gates
        },
        "first_failed_gate": {},
        "accepted_by_event": rows.loc[expansion_accept].event_type.value_counts().to_dict(),
        "pool_by_event": rows.loc[expansion_pool].event_type.value_counts().to_dict(),
        "accepted_by_edge_band": {},
        "counterfactual_remove_one_gate": {},
        "tail_rule_by_period_event_side": {},
    }
    remaining = expansion_pool.copy()
    for name in gates:
        failed = remaining & ~gates[name]
        report["first_failed_gate"][name] = int(failed.sum())
        remaining &= gates[name]
        subset = rows.loc[failed]
        report["first_failed_gate"][name] = {
            "rows": int(len(subset)),
            "realized_mean_pnl_per_contract": float(subset.pnl_per_contract.mean()),
            "realized_profitable_rate": float(subset.pnl_per_contract.gt(0).mean()),
        }
    for name in gates:
        accepted_without = expansion_pool & gates.drop(columns=name).all(axis=1)
        added = accepted_without & ~gates[name]
        subset = rows.loc[added]
        report["counterfactual_remove_one_gate"][name] = {
            "additional_rows": int(len(subset)),
            "realized_mean_pnl_per_contract": float(subset.pnl_per_contract.mean()),
            "realized_total_pnl_per_contract": float(subset.pnl_per_contract.sum()),
            "realized_profitable_rate": float(subset.pnl_per_contract.gt(0).mean()),
        }
    bands = pd.cut(
        rows.entry_net_edge, [.01, .015, .02, .025, .03, .035, .04],
        right=False,
    )
    for band, index in rows.loc[expansion_pool].groupby(bands, observed=True).groups.items():
        report["accepted_by_edge_band"][str(band)] = {
            "pool": int(len(index)),
            "accepted": int(expansion_accept.loc[index].sum()),
        }
    for period, period_mask in {
        "validation": rows.game_date.lt(pd.Timestamp("2026-06-28").date()),
        "development": rows.game_date.ge(pd.Timestamp("2026-06-28").date())
            & rows.game_date.lt(DEVELOPMENT_END),
        "forward": rows.game_date.ge(DEVELOPMENT_END),
    }.items():
        subset = rows.loc[expansion_pool & structural & period_mask]
        grouped = subset.groupby(["event_type", "side"]).pnl_per_contract.agg(
            ["size", "sum", "mean"]
        )
        report["tail_rule_by_period_event_side"][period] = {
            f"{event}:{side}": values.to_dict()
            for (event, side), values in grouped.iterrows()
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
