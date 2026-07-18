"""Chronologically tune and evaluate the partial-reversion runner policy."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

from catboost import CatBoostClassifier
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlb_kalshi.empirical_reaction import reversion_feature_frame  # noqa: E402
from mlb_kalshi.runner import (  # noqa: E402
    RunnerPolicyConfig,
    build_runner_outcomes,
    evaluate_runner_strategy,
    prepare_runner_data,
)


DATA_DIR = PROJECT_ROOT / "data/processed/trade_tape"
CANDIDATES_PATH = DATA_DIR / "empirical_reversion_candidates.parquet"
TRADES_PATH = DATA_DIR / "home_market_trades.parquet"
UPDATES_PATH = DATA_DIR / "state_updates.parquet"
MODEL_DIR = PROJECT_ROOT / "models/market_reaction_model"
REVERSION_MODEL_PATH = MODEL_DIR / "empirical_reversion.cbm"
CONFIG_PATH = MODEL_DIR / "empirical_runner_config.json"
STUDY_DIR = PROJECT_ROOT / "studies/empirical_runner"
TUNING_START = pd.Timestamp("2026-06-17").date()
HOLDOUT_START = pd.Timestamp("2026-06-28").date()
HOLDOUT_END = pd.Timestamp("2026-07-11").date()


def result_dict(result) -> dict:
    return {
        "trades": result.trades,
        "scaled_reversions": result.scaled_reversions,
        "full_reversion_exits": result.full_reversion_exits,
        "runner_target_exits": result.runner_target_exits,
        "runner_trailing_exits": result.runner_trailing_exits,
        "runner_state_exits": result.runner_state_exits,
        "runner_settlements": result.runner_settlements,
        "full_settlements": result.full_settlements,
        "fees": result.fees,
        "capital": result.capital,
        "pnl": result.pnl,
        "roi": result.roi,
    }


def accepted_rows(candidates, outcomes, result) -> pd.DataFrame:
    accepted = candidates[
        candidates["candidate_id"].isin(result.accepted_ids)
    ].merge(outcomes, on="candidate_id", suffixes=("", "_outcome"))
    if accepted.empty:
        return accepted
    columns = [
        "candidate_id", "game_pk", "game_date", "event_type", "entry_side",
        "event_end_time", "entry_time", "entry_price", "contracts",
        "probability_residual", "predicted_reversion_probability",
        "reversion_exit_time", "recovery_price", "recovery_contracts",
        "runner_contracts", "runner_exit_time", "runner_exit_price",
        "exit_reason", "fees", "capital", "pnl", "home_win",
    ]
    return accepted.loc[:, columns].sort_values(["game_date", "entry_time"])


def main() -> None:
    candidates = pd.read_parquet(CANDIDATES_PATH)
    trades = pd.read_parquet(TRADES_PATH)
    updates = pd.read_parquet(UPDATES_PATH)
    model = CatBoostClassifier()
    model.load_model(REVERSION_MODEL_PATH)
    candidates["predicted_reversion_probability"] = model.predict_proba(
        reversion_feature_frame(candidates)
    )[:, 1]
    dates = pd.to_datetime(candidates["game_date"]).dt.date
    tuning = candidates[
        (dates >= TUNING_START) & (dates < HOLDOUT_START)
    ].copy()
    holdout = candidates[
        (dates >= HOLDOUT_START) & (dates < HOLDOUT_END)
    ].copy()
    if tuning.empty or holdout.empty:
        raise RuntimeError("Runner tuning or holdout split is empty")

    tuning_games = set(tuning["game_pk"])
    tuning_trades = trades[trades["game_pk"].isin(tuning_games)].copy()
    tuning_updates = updates[updates["game_pk"].isin(tuning_games)].copy()
    prepared_tuning = prepare_runner_data(tuning_trades, tuning_updates)

    grid_rows = []
    for giveback in [0.35, 0.50]:
        for target_multiple in [1.5, 2.0, 3.0]:
            for adverse_move in [0.02, 0.05]:
                exit_config = RunnerPolicyConfig(
                    trailing_giveback_fraction=giveback,
                    trailing_activation_multiple=0.25,
                    second_target_multiple=target_multiple,
                    adverse_state_probability_move=adverse_move,
                )
                outcomes = build_runner_outcomes(
                    tuning,
                    tuning_trades,
                    tuning_updates,
                    exit_config,
                    prepared_data=prepared_tuning,
                )
                for residual in [0.01, 0.03, 0.05, 0.075, 0.10]:
                    for probability in [0.50, 0.60, 0.70, 0.80, 0.90]:
                        config = RunnerPolicyConfig(
                            minimum_probability_residual=residual,
                            minimum_reversion_probability=probability,
                            trailing_giveback_fraction=giveback,
                            trailing_activation_multiple=0.25,
                            second_target_multiple=target_multiple,
                            adverse_state_probability_move=adverse_move,
                        )
                        result = evaluate_runner_strategy(
                            tuning, outcomes, config
                        )
                        grid_rows.append({**asdict(config), **result_dict(result)})

    grid = pd.DataFrame(grid_rows).sort_values(
        ["roi", "pnl", "trades"], ascending=False
    )
    eligible = grid[
        (grid["trades"] >= 30) & (grid["scaled_reversions"] >= 15)
    ]
    selection = eligible.iloc[0] if not eligible.empty else grid.iloc[0]
    selected = RunnerPolicyConfig(
        enabled=False,
        minimum_probability_residual=float(
            selection["minimum_probability_residual"]
        ),
        minimum_reversion_probability=float(
            selection["minimum_reversion_probability"]
        ),
        trailing_giveback_fraction=float(
            selection["trailing_giveback_fraction"]
        ),
        trailing_activation_multiple=float(
            selection["trailing_activation_multiple"]
        ),
        second_target_multiple=float(selection["second_target_multiple"]),
        adverse_state_probability_move=float(
            selection["adverse_state_probability_move"]
        ),
    )

    holdout_games = set(holdout["game_pk"])
    holdout_trades = trades[trades["game_pk"].isin(holdout_games)].copy()
    holdout_updates = updates[updates["game_pk"].isin(holdout_games)].copy()
    prepared_holdout = prepare_runner_data(holdout_trades, holdout_updates)
    holdout_outcomes = build_runner_outcomes(
        holdout,
        holdout_trades,
        holdout_updates,
        selected,
        prepared_data=prepared_holdout,
    )
    holdout_result = evaluate_runner_strategy(
        holdout, holdout_outcomes, selected
    )

    full_exit_config = RunnerPolicyConfig(
        minimum_probability_residual=selected.minimum_probability_residual,
        minimum_reversion_probability=selected.minimum_reversion_probability,
        trailing_giveback_fraction=selected.trailing_giveback_fraction,
        trailing_activation_multiple=selected.trailing_activation_multiple,
        second_target_multiple=selected.second_target_multiple,
        adverse_state_probability_move=selected.adverse_state_probability_move,
        minimum_runner_contracts=1e9,
    )
    tuning_full_outcomes = build_runner_outcomes(
        tuning,
        tuning_trades,
        tuning_updates,
        full_exit_config,
        prepared_data=prepared_tuning,
    )
    holdout_full_outcomes = build_runner_outcomes(
        holdout,
        holdout_trades,
        holdout_updates,
        full_exit_config,
        prepared_data=prepared_holdout,
    )
    tuning_baseline = evaluate_runner_strategy(
        tuning, tuning_full_outcomes, full_exit_config
    )
    holdout_baseline = evaluate_runner_strategy(
        holdout, holdout_full_outcomes, full_exit_config
    )

    validated = bool(
        not eligible.empty
        and float(selection["pnl"]) > 0
        and holdout_result.trades >= 30
        and holdout_result.pnl > 0
        and holdout_result.roi > 0
        and holdout_result.pnl > holdout_baseline.pnl
    )
    selected = RunnerPolicyConfig(
        **{**asdict(selected), "enabled": validated}
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(asdict(selected), indent=2) + "\n")
    grid.to_csv(STUDY_DIR / "tuning_grid.csv", index=False)
    accepted_rows(holdout, holdout_outcomes, holdout_result).to_csv(
        STUDY_DIR / "holdout_trades.csv", index=False
    )
    summary = {
        "data_scope": "2026 only; no 2025 data used",
        "tuning_dates": "2026-06-17 through 2026-06-27",
        "outer_holdout_dates": "2026-06-28 through 2026-07-10",
        "selection_rule": (
            "maximum tuning ROI with >=30 positions and >=15 scaled reversions"
        ),
        "minimum_sample_requirement_met": bool(not eligible.empty),
        "selected_config": asdict(selected),
        "selected_tuning_result": selection.to_dict(),
        "selected_holdout_result": result_dict(holdout_result),
        "same_entry_full_exit_tuning_baseline": result_dict(tuning_baseline),
        "same_entry_full_exit_holdout_baseline": result_dict(holdout_baseline),
        "capital_recovery": (
            "smallest fractional sale at first executable reversion whose net "
            "proceeds cover original position cost and entry fee"
        ),
        "runner_exits": [
            "second price target",
            "activated high-water trailing giveback",
            "adverse completed-pitch fair-probability move",
            "settlement",
        ],
        "execution_model": (
            "every trigger fills only on a strictly later compatible taker-side "
            "trade with sufficient reported size"
        ),
        "time_based_exit": False,
    }
    (STUDY_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print("Top tuning configurations:")
    print(grid.head(12).to_string(index=False, formatters={
        "roi": "{:.2%}".format,
        "pnl": "${:,.2f}".format,
        "fees": "${:,.2f}".format,
    }))
    print("Holdout runner:", json.dumps(result_dict(holdout_result), indent=2))
    print("Holdout full-exit baseline:", json.dumps(
        result_dict(holdout_baseline), indent=2
    ))
    print(f"Saved {CONFIG_PATH} (enabled={selected.enabled})")


if __name__ == "__main__":
    main()
