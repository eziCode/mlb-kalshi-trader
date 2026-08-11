"""Evaluate a causal competing-risks expansion sleeve for hit reversion."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

from catboost import CatBoostClassifier, CatBoostRegressor, Pool
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.backtest import (  # noqa: E402
    AWAY_TRADE_COLUMNS, STATE_COLUMNS, TRADE_COLUMNS,
    apply_live_paired_execution_prices, apply_publication_latency,
)
from scripts.research_two_stage_value import apply_priority_overlay, metrics  # noqa: E402
from scripts.train_reversion_model import (  # noqa: E402
    CATEGORICAL_FEATURES, CONFIG_PATH, DATA, MODEL_FEATURES,
    build_opportunities,
)
from trade_tape_strategy.core import (  # noqa: E402
    TradeTapeConfig, simulate_trade_tape,
)


FIT_END = pd.Timestamp("2026-06-01").date()
VALIDATION_END = pd.Timestamp("2026-06-28").date()
DEVELOPMENT_END = pd.Timestamp("2026-07-24").date()
RESULT = PROJECT / "artifacts/competing_risks_summary.json"
MODEL_DIR = PROJECT / "models"
METADATA_PATH = MODEL_DIR / "competing_risks.metadata.json"
SEEDS = (3101, 3102, 3103)
CORE_EDGE = 0.04
OUTCOMES = ("fast_reversion", "slow_reversion", "timeout", "settlement")
RESEARCH_FEATURES = MODEL_FEATURES


def feature_pool(frame: pd.DataFrame, label=None) -> Pool:
    return Pool(
        frame.loc[:, RESEARCH_FEATURES], label=label,
        cat_features=list(CATEGORICAL_FEATURES),
    )


def label_outcomes(rows: pd.DataFrame) -> pd.DataFrame:
    result = rows.copy()
    duration = (
        pd.to_datetime(result.exit_time, utc=True)
        - pd.to_datetime(result.entry_time, utc=True)
    ).dt.total_seconds()
    result["duration_seconds"] = duration
    result["risk_outcome"] = np.select(
        [
            result.exit_reason.eq("reversion") & duration.le(30.0),
            result.exit_reason.eq("reversion"),
            result.exit_reason.eq("timeout"),
        ],
        ["fast_reversion", "slow_reversion", "timeout"],
        default="settlement",
    )
    return result


def fit_ensemble(fit: pd.DataFrame):
    models = []
    for seed in SEEDS:
        classifier = CatBoostClassifier(
            iterations=350, depth=5, learning_rate=.025, l2_leaf_reg=40,
            loss_function="MultiClass", random_seed=seed,
            allow_writing_files=False, verbose=False,
        )
        classifier.fit(feature_pool(fit, fit.risk_outcome))
        profitable = CatBoostClassifier(
            iterations=300, depth=5, learning_rate=.025, l2_leaf_reg=40,
            loss_function="Logloss", random_seed=seed + 100,
            allow_writing_files=False, verbose=False,
        )
        profitable.fit(feature_pool(fit, fit.pnl_per_contract.gt(0).astype(int)))
        downside = CatBoostRegressor(
            iterations=325, depth=5, learning_rate=.025, l2_leaf_reg=40,
            loss_function="Quantile:alpha=0.25", random_seed=seed + 200,
            allow_writing_files=False, verbose=False,
        )
        downside.fit(feature_pool(fit, fit.pnl_per_contract))
        severe_loss = CatBoostClassifier(
            iterations=300, depth=5, learning_rate=.025, l2_leaf_reg=40,
            loss_function="Logloss", random_seed=seed + 300,
            allow_writing_files=False, verbose=False,
        )
        severe_loss.fit(feature_pool(
            fit, fit.pnl_per_contract.le(-.05).astype(int)
        ))
        conditional = {}
        for index, outcome in enumerate(OUTCOMES):
            subset = fit[fit.risk_outcome.eq(outcome)]
            if len(subset) < 20:
                raise RuntimeError(f"Insufficient fit rows for {outcome}: {len(subset)}")
            model = CatBoostRegressor(
                iterations=275, depth=4, learning_rate=.025, l2_leaf_reg=40,
                loss_function="Huber:delta=0.05", random_seed=seed + 10 + index,
                allow_writing_files=False, verbose=False,
            )
            model.fit(feature_pool(subset, subset.pnl_per_contract))
            conditional[outcome] = model
        models.append((classifier, profitable, downside, severe_loss, conditional))
    return models


def save_ensemble(models) -> None:
    import hashlib
    hashes = {}
    for seed, (risk, profit, downside, severe_loss, conditional) in zip(SEEDS, models):
        paths = {
            "risk": MODEL_DIR / f"competing_risks_{seed}_risk.cbm",
            "profit": MODEL_DIR / f"competing_risks_{seed}_profit.cbm",
            "downside": MODEL_DIR / f"competing_risks_{seed}_downside.cbm",
            "severe_loss": MODEL_DIR / f"competing_risks_{seed}_severe_loss.cbm",
            **{
                outcome: MODEL_DIR / f"competing_risks_{seed}_{outcome}.cbm"
                for outcome in OUTCOMES
            },
        }
        risk.save_model(str(paths["risk"]))
        profit.save_model(str(paths["profit"]))
        downside.save_model(str(paths["downside"]))
        severe_loss.save_model(str(paths["severe_loss"]))
        for outcome, model in conditional.items():
            model.save_model(str(paths[outcome]))
        hashes[str(seed)] = {
            key: hashlib.sha256(path.read_bytes()).hexdigest()
            for key, path in paths.items()
        }
    METADATA_PATH.write_text(json.dumps({
        "entry_gate": "core_bypass_then_competing_risks",
        "model_features": list(RESEARCH_FEATURES),
        "seeds": list(SEEDS),
        "core_edge": CORE_EDGE,
        "fit_end_exclusive": str(FIT_END),
        "model_sha256": hashes,
    }, indent=2) + "\n")


def predict(models, frame: pd.DataFrame) -> pd.DataFrame:
    expected_by_seed = []
    reversion_by_seed = []
    fast_by_seed = []
    profitable_by_seed = []
    downside_by_seed = []
    severe_loss_by_seed = []
    for classifier, profitable, downside, severe_loss, conditional in models:
        probabilities = classifier.predict_proba(
            frame.loc[:, RESEARCH_FEATURES]
        )
        class_index = {
            str(name): index for index, name in enumerate(classifier.classes_)
        }
        expected = np.zeros(len(frame), dtype=float)
        for outcome in OUTCOMES:
            probability = probabilities[:, class_index[outcome]]
            expected += probability * conditional[outcome].predict(
                frame.loc[:, RESEARCH_FEATURES]
            )
        expected_by_seed.append(expected)
        fast_by_seed.append(probabilities[:, class_index["fast_reversion"]])
        reversion_by_seed.append(
            probabilities[:, class_index["fast_reversion"]]
            + probabilities[:, class_index["slow_reversion"]]
        )
        profitable_by_seed.append(
            profitable.predict_proba(frame.loc[:, RESEARCH_FEATURES])[:, 1]
        )
        downside_by_seed.append(
            downside.predict(frame.loc[:, RESEARCH_FEATURES])
        )
        severe_loss_by_seed.append(
            severe_loss.predict_proba(frame.loc[:, RESEARCH_FEATURES])[:, 1]
        )
    expected_matrix = np.vstack(expected_by_seed)
    return pd.DataFrame({
        "expected_pnl_per_contract": expected_matrix.mean(axis=0),
        "expected_pnl_lower_bound": (
            expected_matrix.mean(axis=0) - expected_matrix.std(axis=0)
        ),
        "fast_reversion_probability": np.vstack(fast_by_seed).mean(axis=0),
        "reversion_probability": np.vstack(reversion_by_seed).mean(axis=0),
        "profitability_probability": np.vstack(profitable_by_seed).mean(axis=0),
        "downside_quartile_pnl": np.vstack(downside_by_seed).mean(axis=0),
        "severe_loss_probability": np.vstack(severe_loss_by_seed).mean(axis=0),
    }, index=frame.index)


def core_mask(frame: pd.DataFrame, config: TradeTapeConfig) -> pd.Series:
    permitted = frame.event_type.isin(config.allowed_event_types)
    for event, maximum_outs in config.maximum_outs_after_by_event.items():
        permitted &= ~frame.event_type.eq(event) | frame.outs.le(maximum_outs)
    return permitted & frame.entry_net_edge.ge(CORE_EDGE)


def select_sleeve(
    frame: pd.DataFrame, predictions: pd.DataFrame, config: TradeTapeConfig,
) -> pd.DataFrame:
    core = core_mask(frame, config)
    # Fixed, interpretable gate: positive one-standard-deviation ensemble
    # lower bound and reversion more likely than non-reversion. No threshold
    # search is performed on validation or holdout periods.
    expansion = (
        predictions.expected_pnl_lower_bound.gt(0.0)
        & predictions.reversion_probability.gt(0.5)
        & predictions.profitability_probability.gt(0.5)
        & predictions.severe_loss_probability.lt(.25)
    )
    selected = frame[core | expansion].copy()
    selected = selected.join(predictions)
    return apply_priority_overlay(
        selected, core.loc[selected.index],
        config.minimum_seconds_between_entries,
    )


class IntegratedScorer:
    """Online scorer whose core bypass exactly matches the live policy."""

    def __init__(self, models, core_edge: float):
        self.models = models
        self.core_edge = float(core_edge)

    def accepts(self, features: dict[str, object]) -> tuple[bool, float]:
        if float(features["entry_net_edge"]) >= self.core_edge:
            return True, float(features["entry_net_edge"])
        frame = pd.DataFrame([features], columns=RESEARCH_FEATURES)
        scores = predict(self.models, frame).iloc[0]
        accepted = bool(
            scores.expected_pnl_lower_bound > 0
            and scores.reversion_probability > .5
            and scores.profitability_probability > .5
            and scores.severe_loss_probability < .25
        )
        return accepted, float(scores.expected_pnl_lower_bound)


def integrated_replay(
    home: pd.DataFrame, away: pd.DataFrame, states: pd.DataFrame,
    config: TradeTapeConfig, models, start, end,
):
    mask = home.game_date >= start
    if end is not None:
        mask &= home.game_date < end
    period_home = home[mask].copy()
    games = set(period_home.game_pk.unique())
    tape = apply_live_paired_execution_prices(
        period_home, away[away.game_pk.isin(games)].copy(),
    )
    updates = apply_publication_latency(
        states[states.game_pk.isin(games)].copy()
    )
    expansion_config = replace(
        config, minimum_edge=.01, direct_value_model_enabled=False,
    )
    result = simulate_trade_tape(
        tape, updates, expansion_config,
        entry_scorer=IntegratedScorer(models, CORE_EDGE),
    )
    records = pd.DataFrame(asdict(record) for record in result.records)
    return result, records, len(games)


def main() -> None:
    config = TradeTapeConfig(**json.loads(CONFIG_PATH.read_text()))
    home = pd.read_parquet(DATA / "home_market_trades.parquet", columns=TRADE_COLUMNS)
    away = pd.read_parquet(DATA / "away_market_trades.parquet", columns=AWAY_TRADE_COLUMNS)
    states = pd.read_parquet(DATA / "state_updates.parquet", columns=STATE_COLUMNS)
    for frame in (home, away, states):
        frame["game_date"] = pd.to_datetime(frame.game_date).dt.date
    rows = label_outcomes(build_opportunities(home, away, states, config))
    fit = rows[rows.game_date < FIT_END].copy()
    models = fit_ensemble(fit)
    save_ensemble(models)
    game_dates = home[["game_pk", "game_date"]].drop_duplicates()
    periods = {
        "validation": (FIT_END, VALIDATION_END),
        "development": (VALIDATION_END, DEVELOPMENT_END),
        "forward": (DEVELOPMENT_END, None),
    }
    selected = {}
    baseline = {}
    scheduled = {}
    for name, (start, end) in periods.items():
        mask = rows.game_date >= start
        game_mask = game_dates.game_date >= start
        if end is not None:
            mask &= rows.game_date < end
            game_mask &= game_dates.game_date < end
        period = rows[mask].copy()
        predictions = predict(models, period)
        selected[name] = select_sleeve(period, predictions, config)
        baseline_frame = period[core_mask(period, config)].copy()
        baseline[name] = apply_priority_overlay(
            baseline_frame,
            pd.Series(True, index=baseline_frame.index),
            config.minimum_seconds_between_entries,
        )
        scheduled[name] = int(game_dates.loc[game_mask, "game_pk"].nunique())
    evaluated_names = ("development", "forward")
    combined = pd.concat([selected[name] for name in evaluated_names])
    baseline_combined = pd.concat([baseline[name] for name in evaluated_names])
    total_games = sum(scheduled[name] for name in evaluated_names)
    summary = {
        "method": "competing-risks ensemble lower-bound expansion sleeve",
        "fit_end_exclusive": str(FIT_END),
        "validation_end_exclusive": str(VALIDATION_END),
        "development_end_exclusive": str(DEVELOPMENT_END),
        "features": list(RESEARCH_FEATURES),
        "outcomes": list(OUTCOMES),
        "fit_rows": len(fit),
        "fit_outcomes": fit.risk_outcome.value_counts().to_dict(),
        "periods": {
            name: {
                "baseline": metrics(baseline[name], scheduled[name]),
                "sleeve": metrics(selected[name], scheduled[name]),
            }
            for name in periods
        },
        "combined_evaluation": {
            "baseline": metrics(baseline_combined, total_games),
            "sleeve": metrics(combined, total_games),
        },
    }
    evaluation = summary["combined_evaluation"]
    gates = {
        "adds_trades": evaluation["sleeve"]["trades"] > evaluation["baseline"]["trades"],
        "adds_pnl": evaluation["sleeve"]["pnl"] > evaluation["baseline"]["pnl"],
        "roi_at_least_five_percent": evaluation["sleeve"]["roi"] >= .05,
        "validation_positive": summary["periods"]["validation"]["sleeve"]["pnl"] > 0,
        "both_evaluation_periods_positive": all(
            summary["periods"][name]["sleeve"]["pnl"] > 0
            for name in evaluated_names
        ),
        "concentration_positive": evaluation["sleeve"]["pnl_without_top_four_games"] > 0,
    }
    summary["gates"] = gates
    integrated = {}
    for name in ("development", "forward"):
        start, end = periods[name]
        result, records, games = integrated_replay(
            home, away, states, config, models, start, end,
        )
        integrated[name] = {
            **metrics(records, games),
            "confirmed_signals": result.confirmed_signals,
            "model_rejected_signals": result.model_rejected_signals,
        }
    integrated_combined = {
        "trades": sum(integrated[name]["trades"] for name in integrated),
        "games": sum(scheduled[name] for name in integrated),
        "pnl": sum(integrated[name]["pnl"] for name in integrated),
        "capital": sum(integrated[name]["capital"] for name in integrated),
    }
    integrated_combined["trades_per_game"] = (
        integrated_combined["trades"] / integrated_combined["games"]
    )
    integrated_combined["roi"] = (
        integrated_combined["pnl"] / integrated_combined["capital"]
    )
    summary["integrated_replay"] = {
        **integrated, "combined": integrated_combined,
    }
    exact_baseline = {
        "trades": 105, "pnl": 22.75, "trades_per_game": 105 / 531,
    }
    integrated_gates = {
        "adds_trades_vs_exact_baseline": integrated_combined["trades"] > 105,
        "adds_pnl_vs_exact_baseline": integrated_combined["pnl"] > 22.75,
        "roi_at_least_five_percent": integrated_combined["roi"] >= .05,
        "both_periods_positive": all(
            integrated[name]["pnl"] > 0 for name in integrated
        ),
    }
    summary["exact_baseline_reference"] = exact_baseline
    summary["integrated_gates"] = integrated_gates
    summary["deployable"] = all(gates.values()) and all(integrated_gates.values())
    RESULT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
