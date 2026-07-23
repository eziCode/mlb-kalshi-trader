"""Leak-free chronological backtest of the deployed latency policy."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd

from settlement_value_strategy.research_latency import (
    add_future_market_target,
    period,
)
from settlement_value_strategy.strategy import (
    MispricingConfig,
    _expit,
    _logit,
    mispricing_feature_frame,
    simulate_paired_both,
)


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data/settlement_value"
RESULTS = ROOT / "results"
FOLDS = {
    "april": ("2026-04-01", "2026-05-01"),
    "may": ("2026-05-01", "2026-06-01"),
    "early_june": ("2026-06-01", "2026-06-17"),
    "late_june": ("2026-06-17", "2026-06-28"),
    "early_july": ("2026-06-28", "2026-07-11"),
    "final_holdout": ("2026-07-11", "2026-07-18"),
}


def deployed_config() -> MispricingConfig:
    raw = json.loads((ROOT / "model/live_config.json").read_text())
    config = MispricingConfig(**{
        key: value for key, value in raw.items()
        if key in MispricingConfig.__dataclass_fields__
    })
    # Live execution is capped at $2 even though the research artifact retained
    # its original $10 notional. Both liquidity and fees must use real sizing.
    return replace(config, bet_size=2.0)


def metrics(result, total_games: int) -> dict:
    records = pd.DataFrame(result.records)
    game_pnl = (
        records.groupby("game_pk").pnl.sum()
        if not records.empty else pd.Series(dtype=float)
    )
    top = game_pnl.nlargest(min(4, len(game_pnl))).sum()
    return {
        "games": total_games,
        "orders": result.orders,
        "trades": result.trades,
        "trades_per_game": result.trades / total_games if total_games else 0.0,
        "pnl": result.pnl,
        "capital": result.capital,
        "roi": result.roi,
        "pnl_without_top_four_games": result.pnl - float(top),
    }


def main() -> None:
    frame = pd.read_parquet(DATA / "decision_rows.parquet")
    home = pd.read_parquet(DATA / "execution_trades.parquet")
    away = pd.read_parquet(DATA / "away_execution_trades.parquet")
    labeled = add_future_market_target(frame, home)
    config = deployed_config()
    fold_results: dict[str, dict] = {}
    all_records: list[pd.DataFrame] = []

    for fold_number, (name, (start, end)) in enumerate(FOLDS.items()):
        fit = period(labeled, "2025-01-01", start)
        test = period(labeled, start, end)
        counts = fit.groupby("game_pk").size()
        weights = fit.game_pk.map(1.0 / counts)
        model = CatBoostRegressor(
            iterations=300,
            depth=5,
            learning_rate=.025,
            l2_leaf_reg=40,
            loss_function="Huber:delta=0.05",
            random_seed=117 + fold_number,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(
            mispricing_feature_frame(fit),
            np.clip(fit.future_logit_move.to_numpy(float), -.75, .75),
            sample_weight=weights,
        )
        probability = _expit(
            _logit(test.market_home_price)
            + np.clip(model.predict(mispricing_feature_frame(test)), -.5, .5)
        )
        games = set(test.game_pk)
        result = simulate_paired_both(
            test,
            probability,
            home[home.game_pk.isin(games)],
            away[away.game_pk.isin(games)],
            config,
        )
        fold_results[name] = metrics(result, len(games))
        records = pd.DataFrame(result.records)
        if not records.empty:
            records["evaluation_fold"] = name
            all_records.append(records)

    total_capital = sum(item["capital"] for item in fold_results.values())
    total_pnl = sum(item["pnl"] for item in fold_results.values())
    total_games = sum(item["games"] for item in fold_results.values())
    total_trades = sum(item["trades"] for item in fold_results.values())
    summary = {
        "method": "expanding-window; every model trained before its test fold",
        "config": asdict(config),
        "aggregate": {
            "games": total_games,
            "orders": sum(item["orders"] for item in fold_results.values()),
            "trades": total_trades,
            "trades_per_game": total_trades / total_games,
            "pnl": total_pnl,
            "capital": total_capital,
            "roi": total_pnl / total_capital if total_capital else 0.0,
            "positive_folds": sum(item["pnl"] > 0 for item in fold_results.values()),
        },
        "folds": fold_results,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "live_policy_backtest_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    if all_records:
        pd.concat(all_records, ignore_index=True).to_csv(
            RESULTS / "live_policy_backtest_trades.csv", index=False
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
