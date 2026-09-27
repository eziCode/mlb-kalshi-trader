"""Hold-to-settlement test of refreshed baseball probabilities. No order API."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import date, datetime, timezone
import heapq
import json
import multiprocessing
from pathlib import Path

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd

from research.continuous import Execution, game_frame, NS
from research.passive import fee
from research.slate_study import digest, empty_output
from research.state_refresh import ROOT, home_probability, post_features, probability
from research.validation import summarize

EDGES = (0., .03, .05, .08)
PARTITIONS = {"development": (date(2026, 8, 12), date(2026, 9, 1)),
              "selection": (date(2026, 9, 2), date(2026, 9, 15)),
              "chronological_check": (date(2026, 9, 16), date(2026, 9, 26))}
EVENTS = ROOT / "data/raw/reboot_kalshi/cache/events_2026-08-10_2026-09-26.json"


def settlements(path=EVENTS):
    result = {}
    for event in json.loads(path.read_text()):
        for market in event.get("markets", []):
            if market.get("settlement_ts") and market.get("settlement_value_dollars") is not None:
                value = float(market["settlement_value_dollars"])
                if not 0 <= value <= 1:
                    raise ValueError("Invalid terminal contract value")
                result[market["ticker"]] = {"time": pd.Timestamp(market["settlement_ts"]).value,
                                             "value": value}
    return result


def refreshed_chunk(job):
    chunk, model_path, priors, execution_dict = job
    directory = ROOT / "data/reboot" / chunk
    metadata = json.loads((directory / "state_updates.metadata.json").read_text())
    if metadata.get("state_contract") != "atomic_pitch_or_play_v2":
        raise ValueError("Noncausal state input")
    home, away, updates = [pd.read_parquet(directory / name) for name in
                          ("home_market_trades.parquet", "away_market_trades.parquet", "state_updates.parquet")]
    if not set(updates.game_pk).issubset(priors):
        raise ValueError("Missing frozen team prior")
    model = CatBoostClassifier().load_model(str(model_path))
    updates["fair_after"] = home_probability(model, post_features(updates, priors))
    ag, ug = dict(tuple(away.groupby("game_pk"))), dict(tuple(updates.groupby("game_pk")))
    execution = Execution(**execution_dict)
    frames = [game_frame(h, ag.get(pk, away.iloc[:0]), ug[pk], execution, False)
              for pk, h in home.groupby("game_pk")]
    print(f"Refreshed {chunk}: {len(frames)} games", flush=True)
    return pd.concat(frames, ignore_index=True)


def build(model_directory, execution, workers):
    raw = pd.read_parquet(ROOT / "data/raw/mlb_statcast/2026.parquet", columns=["game_pk", "home_team", "away_team"])
    teams = raw.groupby("game_pk", as_index=False).first()
    ratings = json.loads((model_directory / "prior.json").read_text())["ratings"]
    priors = {g.game_pk: probability(g.home_team, g.away_team, ratings) for g in teams.itertuples(index=False)}
    dataset = json.loads((ROOT / "data/reboot/dataset.json").read_text())
    jobs = [(chunk, model_directory / "model.cbm", priors, asdict(execution)) for chunk in dataset["chunks"]]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        frames = list(pool.map(refreshed_chunk, jobs))
    frame = pd.concat(frames, ignore_index=True).sort_values(["time", "game_pk", "market"]).reset_index(drop=True)
    if frame.duplicated(["time", "game_pk", "market"]).any():
        raise ValueError("Overlapping state/market records")
    return frame


def simulate(frame, edge, terminal, execution=Execution(), starting_cash=100., settlement_delay=60.):
    if starting_cash <= 0 or not np.isfinite(starting_cash) or settlement_delay < 0:
        raise ValueError("Invalid portfolio parameters")
    fields = ["game_pk", "time", "market", "ticker", "buy_limit", "buy_time", "buy_price"]
    arrays = {c: frame[c].to_numpy() for c in fields}
    fair = frame.mid.to_numpy() + frame.fair_gap.to_numpy()
    limits = arrays["buy_limit"]
    costs = np.array([p + fee(1, p, .07) if np.isfinite(p) else np.nan for p in limits])
    signal = np.where(frame.entry_allowed & np.isfinite(fair) & (fair - costs >= edge), fair - costs, -np.inf)
    starts = np.r_[0, 1 + np.flatnonzero((arrays["time"][1:] != arrays["time"][:-1]) |
                                       (arrays["game_pk"][1:] != arrays["game_pk"][:-1])), len(frame)]
    pending, positions, done, events, rows = set(), {}, set(), [], []
    cash, locked, peak, serial = starting_cash, 0., 0., 0
    counter = dict(entry_orders=0, expired_entries=0, cash_rejected_orders=0, settlements=0)

    def advance(now):
        nonlocal cash, locked, serial
        while events and events[0][0] < now:
            stamp, _, kind, pk, data = heapq.heappop(events)
            if kind == "settle":
                position = positions.pop(pk)
                cash += data
                locked -= position["cost"]
                counter["settlements"] += 1
                rows.append({**position["row"], "exit_time": pd.Timestamp(stamp, tz="UTC"),
                    "exit_price": data, "pnl": data - position["cost"], "exit_reason": "exchange_settlement"})
                continue
            pending.remove(pk)
            index, reserve, decision = data
            if not arrays["buy_time"][index]:
                cash += reserve
                locked -= reserve
                counter["expired_entries"] += 1
                continue
            price = float(arrays["buy_price"][index])
            charge = fee(1, price, .07)
            cost = price + charge
            if cost > reserve + 1e-9:
                raise ValueError("Fill exceeds immutable reserved limit")
            cash += reserve - cost
            locked -= reserve - cost
            ticker = arrays["ticker"][index]
            positions[pk] = {"cost": cost, "row": {"game_pk": pk,
                "side": "home_yes" if arrays["market"][index] == 0 else "away_yes", "ticker": ticker,
                "entry_time": pd.Timestamp(stamp, tz="UTC"), "entry_decision_time": pd.Timestamp(decision, tz="UTC"),
                "entry_price": price, "contracts": 1., "fees": charge}}
            done.add(pk)
            result = terminal.get(ticker)
            if result is not None:
                if result["time"] <= stamp:
                    raise ValueError("Execution evidence occurred after exchange settlement")
                serial += 1
                heapq.heappush(events, (int(result["time"] + settlement_delay * NS), serial,
                                       "settle", pk, result["value"]))

    for a, b in zip(starts[:-1], starts[1:]):
        now, pk = int(arrays["time"][a]), int(arrays["game_pk"][a])
        advance(now)
        if pk in pending or pk in done:
            continue
        index = a + int(np.argmax(signal[a:b]))
        if not np.isfinite(signal[index]):
            continue
        reserve = costs[index]
        if reserve > cash + 1e-9:
            counter["cash_rejected_orders"] += 1
            continue
        cash -= reserve
        locked += reserve
        peak = max(peak, locked)
        counter["entry_orders"] += 1
        serial += 1
        pending.add(pk)
        stamp = arrays["buy_time"][index] or now + round((execution.latency + execution.window) * NS)
        heapq.heappush(events, (int(stamp), serial, "entry", pk, (index, reserve, now)))
    advance(np.iinfo(np.int64).max)
    for position in positions.values():
        rows.append({**position["row"], "exit_time": pd.NaT, "exit_price": 0.,
                     "pnl": -position["cost"], "exit_reason": "unresolved_zero_mark"})
    columns = ["game_pk", "side", "ticker", "entry_time", "entry_decision_time", "entry_price",
               "contracts", "fees", "exit_time", "exit_price", "pnl", "exit_reason"]
    trades = pd.DataFrame(rows, columns=columns)
    for name in ("pnl", "fees", "contracts", "entry_price"):
        trades[name] = trades[name].astype(float)
    counter.update(starting_cash=starting_cash, ending_equity=cash,
        unresolved_positions=len(positions), unresolved_cost=locked, peak_reserved_and_invested=peak,
        settlement_delay_seconds=settlement_delay, settlement_cash_reused=True)
    if abs(cash - starting_cash - trades.pnl.sum()) > 1e-7:
        raise ValueError("Settlement account does not reconcile")
    return trades, counter


def run(args):
    empty_output(args.output_dir)
    model_dir, output = args.model_dir.resolve(), args.output_dir
    execution = Execution(latency=args.latency, penalty=args.penalty, state_delay=args.state_delay)
    files = [Path(__file__), ROOT / "research/state_refresh.py", ROOT / "research/STATE_REFRESH_PROTOCOL.md",
             ROOT / "research/continuous.py", ROOT / "research/passive.py", ROOT / "research/validation.py",
             ROOT / "hit_reversion_strategy/scripts/backtest.py", EVENTS,
             ROOT / "data/raw/mlb_statcast/2026.parquet", ROOT / "data/reboot/dataset.json",
             model_dir / "model.cbm", model_dir / "prior.json", model_dir / "manifest.json"]
    for chunk in json.loads((ROOT / "data/reboot/dataset.json").read_text())["chunks"]:
        files.extend((ROOT / "data/reboot" / chunk).glob("*.*"))
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in files}
    manifest = {"started_at": datetime.now(timezone.utc).isoformat(), "hashes": hashes,
                "execution": asdict(execution), "edges": EDGES, "settlement_delay_seconds": 60,
                "partitions": {k: list(map(str, v)) for k, v in PARTITIONS.items()},
                "role": "previously_inspected_chronological_research", "deployment_ready": False}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    frame, terminal = build(model_dir, execution, args.workers), settlements()
    results = {}
    for edge in EDGES:
        key = f"edge_{round(edge * 100):02d}c"
        results[key] = {}
        for name, (start, end) in PARTITIONS.items():
            part = frame[(frame.game_date >= start) & (frame.game_date <= end)]
            trades, counters = simulate(part, edge, terminal, execution)
            dates = part.groupby("game_pk").game_date.first().to_dict()
            metric = summarize(trades, dates, start, end)
            metric.update(counters, entries_per_game=len(trades) / len(dates),
                game_coverage=trades.game_pk.nunique() / len(dates),
                every_game_traded=trades.game_pk.nunique() == len(dates))
            results[key][name] = metric
            trades.to_csv(output / f"{key}_{name}.csv", index=False)
            print(f"{key} {name}: {len(trades)}/{len(dates)} games, PnL ${metric['pnl']:.4f}", flush=True)
    for name, expected in hashes.items():
        if digest(ROOT / name) != expected:
            raise RuntimeError(f"Input changed during replay: {name}")
    (output / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--latency", type=float, default=.68)
    parser.add_argument("--penalty", type=float, default=.01)
    parser.add_argument("--state-delay", type=float, default=2.)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
