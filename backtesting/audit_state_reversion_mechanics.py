"""Mechanical audits and tick replays for the state-reversion strategy."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlb_kalshi.hybrid import (  # noqa: E402
    EVENT_BATTING_DIRECTIONS, anchored_event_target, event_category,
)


DATA_DIR = PROJECT_ROOT / "data/processed/trade_tape"
STUDY_DIR = PROJECT_ROOT / "studies/state_reversion"


def logit(values):
    values = np.clip(np.asarray(values, float), 1e-4, 1 - 1e-4)
    return np.log(values / (1 - values))


def audit_state_alignment(updates: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frame = updates.copy().sort_values(["game_pk", "pitch_end_time"])
    frame["category"] = frame["completed_event"].map(event_category)
    frame["expected_batting_direction"] = frame["category"].map(
        EVENT_BATTING_DIRECTIONS
    )
    home_move = frame.fair_after - frame.fair_before
    frame["fair_batting_move"] = np.where(
        frame.completed_event_batting_home.eq(True), home_move, -home_move
    )
    terminal = frame.expected_batting_direction.notna()
    directional = terminal & frame.expected_batting_direction.ne(0)
    frame["direction_violation"] = directional & (
        frame.fair_batting_move * frame.expected_batting_direction < 0
    )
    next_before = frame.groupby("game_pk").fair_before.shift(-1)
    comparable = next_before.notna()
    continuity_error = (frame.fair_after - next_before).abs()
    report = frame.loc[terminal, [
        "game_pk", "pitch_end_time", "completed_event", "category",
        "completed_event_batting_home", "fair_before", "fair_after",
        "fair_batting_move", "expected_batting_direction",
        "direction_violation",
    ]]
    summary = {
        "updates": len(frame), "terminal_events": int(terminal.sum()),
        "directional_terminal_events": int(directional.sum()),
        "direction_violations": int(frame.direction_violation.sum()),
        "direction_violation_rate": float(
            frame.loc[directional, "direction_violation"].mean()
        ) if directional.any() else 0.0,
        "fair_transition_continuity_max_error": float(
            continuity_error[comparable].max()
        ),
        "fair_transition_continuity_errors": int(
            continuity_error[comparable].gt(1e-9).sum()
        ),
    }
    return report, summary


def audit_candidate_arithmetic(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    expected_entry_home = np.where(
        frame.side.eq("yes"), frame.entry_price, 1 - frame.entry_price
    )
    expected_target_contract = np.where(
        frame.side.eq("yes"),
        1 / (1 + np.exp(-(
            logit(frame.target_home_price)
            + frame.signed_logit_residual * (1 - frame.reversion_fraction)
        ))),
        1 - 1 / (1 + np.exp(-(
            logit(frame.target_home_price)
            + frame.signed_logit_residual * (1 - frame.reversion_fraction)
        ))),
    )
    entry_error = np.abs(expected_entry_home - frame.entry_home_price)
    target_error = np.abs(expected_target_contract - frame.target_contract_price)
    settled = frame.exit_reason.eq("settlement")
    expected_pnl = np.where(
        settled,
        np.where(
            (frame.side.eq("yes") & frame.home_win.eq(1))
            | (frame.side.eq("no") & frame.home_win.eq(0)),
            frame.contracts, 0.0,
        ) - frame.contracts * frame.entry_price - frame.entry_fee,
        frame.contracts * (frame.exit_price - frame.entry_price) - frame.fees,
    )
    pnl_error = np.abs(expected_pnl - frame.pnl)
    violations = frame.loc[
        (entry_error > 1e-9) | (target_error > 1e-9) | (pnl_error > 1e-9),
        ["game_pk", "side", "entry_time", "exit_reason", "pnl"],
    ].copy()
    violations["entry_home_error"] = entry_error[violations.index]
    violations["target_contract_error"] = target_error[violations.index]
    violations["pnl_error"] = pnl_error[violations.index]
    summary = {
        "candidates": len(frame), "no_candidates": int(frame.side.eq("no").sum()),
        "maximum_entry_home_error": float(entry_error.max()),
        "maximum_target_contract_error": float(target_error.max()),
        "maximum_pnl_error": float(pnl_error.max()),
        "violations": len(violations),
    }
    return violations, summary


def build_tick_replays(records: pd.DataFrame, trades: pd.DataFrame, updates: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame()
    representatives = pd.concat([
        records[records.side.eq("yes")].nsmallest(1, "pnl"),
        records[records.side.eq("yes")].nlargest(1, "pnl"),
        records[records.side.eq("no")].nsmallest(1, "pnl"),
        records[records.side.eq("no")].nlargest(1, "pnl"),
    ]).drop_duplicates(["game_pk", "entry_time"])
    output = []
    for case_id, row in enumerate(representatives.itertuples(index=False), start=1):
        entry_time = pd.Timestamp(row.entry_time)
        end_time = (
            pd.Timestamp(row.exit_time) if pd.notna(row.exit_time)
            else entry_time + pd.Timedelta(seconds=125)
        )
        tape = trades[
            trades.game_pk.eq(row.game_pk)
            & (trades.created_time >= pd.Timestamp(row.trigger_time) - pd.Timedelta(seconds=3))
            & (trades.created_time <= end_time + pd.Timedelta(seconds=1))
        ].copy().sort_values(["created_time", "trade_id"])
        state = updates[updates.game_pk.eq(row.game_pk)][
            ["pitch_end_time", "fair_after"]
        ].sort_values("pitch_end_time")
        tape = pd.merge_asof(
            tape, state, left_on="created_time", right_on="pitch_end_time",
            direction="backward",
        )
        tape["fair_after"] = tape.fair_after.fillna(row.fair_after)
        tape["dynamic_target"] = anchored_event_target(
            row.target_home_price, row.fair_after, tape.fair_after
        )
        tape["signed_residual"] = logit(tape.yes_price_dollars) - logit(
            tape.dynamic_target
        )
        tape["case_id"] = case_id
        tape["case_side"] = row.side
        tape["case_pnl"] = row.pnl
        tape["is_entry_time"] = tape.created_time.eq(entry_time)
        tape["is_exit_time"] = tape.created_time.eq(end_time)
        output.append(tape)
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def main() -> None:
    updates = pd.read_parquet(DATA_DIR / "state_updates.parquet")
    trades = pd.read_parquet(DATA_DIR / "home_market_trades.parquet")
    for column in ["pitch_start_time", "pitch_end_time"]:
        updates[column] = pd.to_datetime(updates[column], utc=True)
    trades["created_time"] = pd.to_datetime(trades.created_time, utc=True)
    candidates = pd.read_parquet(STUDY_DIR / "deterministic_candidates.parquet")
    records_path = STUDY_DIR / "deterministic_holdout_trades.csv"
    records = pd.read_csv(records_path) if records_path.exists() else pd.DataFrame()
    for column in ["trigger_time", "entry_time", "exit_time"]:
        if column in records:
            records[column] = pd.to_datetime(records[column], utc=True)

    state_report, state_summary = audit_state_alignment(updates)
    arithmetic_report, arithmetic_summary = audit_candidate_arithmetic(candidates)
    replay = build_tick_replays(records, trades, updates)
    candidates["game_date"] = pd.to_datetime(candidates.game_date).dt.date
    candidates["period"] = np.where(
        candidates.game_date >= pd.Timestamp("2026-06-28").date(),
        "holdout", "pre_holdout",
    )
    candidates["latency_bin"] = pd.cut(
        candidates.entry_latency_seconds,
        [0, 2, 4, 6, 8, 10, np.inf],
        labels=["0-2", "2-4", "4-6", "6-8", "8-10", "10+"],
    )
    latency = (
        candidates.groupby(
            ["period", "side", "latency_bin"], observed=True
        )
        .agg(
            candidates=("pnl", "size"), pnl=("pnl", "sum"),
            mean_pnl=("pnl", "mean"), win_rate=("pnl", lambda x: x.gt(0).mean()),
        )
        .reset_index()
    )
    state_report.to_csv(STUDY_DIR / "state_alignment_audit.csv", index=False)
    arithmetic_report.to_csv(STUDY_DIR / "arithmetic_violations.csv", index=False)
    replay.to_csv(STUDY_DIR / "representative_tick_replays.csv", index=False)
    latency.to_csv(STUDY_DIR / "entry_latency_diagnostics.csv", index=False)
    summary = {
        "state_alignment": state_summary,
        "candidate_arithmetic": arithmetic_summary,
        "representative_replay_cases": int(replay.case_id.nunique()) if not replay.empty else 0,
    }
    (STUDY_DIR / "mechanics_audit_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
