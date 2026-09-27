"""Chronological high-activity strategy research. Contains no order API.

python -m research.continuous develop
python -m research.continuous check
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timezone
import heapq
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from research.reboot import ROOT, MODELS, digest
from research.validation import summarize
from scripts.backtest import apply_publication_latency
from trade_tape_strategy.hybrid import anchored_event_target
from trade_tape_strategy.strategy import taker_fee

NS = 1_000_000_000
FEATURES = ["mid", "spread", "return_10", "return_30", "return_120",
            "other_gap", "flow_30", "volume_30", "fair_gap", "play_gap",
            "state_age", "play_age", "inning", "score_diff", "outs", "home_side"]
FAMILIES = ["revert_30", "revert_120", "momentum_30", "momentum_120",
            "cross_market", "state_value", "all_play", "learned"]
TRAIN_END = date(2026, 9, 1)
SELECT_END = date(2026, 9, 15)
FINAL_END = date(2026, 9, 26)


@dataclass(frozen=True)
class Execution:
    latency: float = .68
    window: float = .25
    penalty: float = .01
    participation: float = .1
    state_delay: float = 2.
    step: int = 5
    quote_age: float = 10.

    def __post_init__(self):
        vals = np.array(list(asdict(self).values()), dtype=float)
        if not np.isfinite(vals).all() or self.latency < 0 or self.window <= 0:
            raise ValueError("Invalid execution timing")
        if not 0 < self.participation <= 1 or not 0 <= self.penalty < .5:
            raise ValueError("Invalid cost or participation")
        if self.state_delay < 0 or self.quote_age <= 0 or self.step <= self.latency + self.window:
            raise ValueError("Decisions must be farther apart than the pending order window")


@dataclass(frozen=True)
class Policy:
    family: str
    hold: int
    edge: float

    @property
    def name(self):
        return f"{self.family}_{self.hold}s_{round(self.edge * 100):02d}c"


def policies():
    return [Policy(f, h, e) for f in FAMILIES for h in (60, 300) for e in (0., .01, .03)] + [
        Policy("activity_control", h, -10.) for h in (60, 300)]


def nanos(values):
    return pd.DatetimeIndex(values).as_unit("ns").asi8


def prior_values(times, values, decisions, max_age=np.inf):
    """Strictly past observations, including for equal-timestamp packets."""
    out = np.full(len(decisions), np.nan)
    if len(times):
        idx = np.searchsorted(times, decisions, side="left") - 1
        valid = (idx >= 0) & ((decisions - times[np.maximum(idx, 0)]) / NS <= max_age)
        out[valid] = values[idx[valid]]
    return out


def print_sides(tape):
    tape = tape.sort_values(["created_time", "trade_id"])
    result = {}
    for side in ("yes", "no"):
        part = tape[tape.taker_outcome_side.eq(side)]
        result[side] = (nanos(part.created_time), part.yes_price_dollars.to_numpy(float),
                        part.count_fp.to_numpy(float))
    return result


def proxy_quotes(sides, times, age):
    ask = prior_values(*sides["yes"][:2], times, age)
    bid = prior_values(*sides["no"][:2], times, age)
    valid = (ask >= bid) & (ask - bid <= .08 + 1e-9) & (bid > 0) & (ask < 1)
    return np.where(valid, bid, np.nan), np.where(valid, ask, np.nan)


def fill_evidence(sides, decisions, limits, execution, buy):
    """First compatible evidence; no order is rescored using future data.

    Returned evidence is an outcome, never a feature or submission criterion.
    One whole contract needs sufficient side-specific printed participation.
    """
    times, prices, volume = sides["yes" if buy else "no"]
    arrival = decisions + round(execution.latency * NS)
    expiry = arrival + round(execution.window * NS)
    filled_at = np.zeros(len(decisions), dtype=np.int64)
    paid = np.full(len(decisions), np.nan)
    left = np.searchsorted(times, arrival, side="right")
    right = np.searchsorted(times, expiry, side="right")
    for i in np.flatnonzero((right > left) & np.isfinite(limits)):
        for j in range(left[i], right[i]):
            price = prices[j] + (execution.penalty if buy else -execution.penalty)
            within = price <= limits[i] + 1e-9 if buy else price >= limits[i] - 1e-9
            if 0 < price < 1 and within and volume[j] * execution.participation >= 1 - 1e-9:
                filled_at[i], paid[i] = times[j], price
                break
    return filled_at, paid


def causal_states(updates, execution):
    delayed = apply_publication_latency(updates)
    delayed["event_available_time"] += pd.Timedelta(seconds=execution.state_delay)
    delayed = delayed.sort_values(["event_available_time", "pitch_end_time", "pitch_number"])
    # Availability order can differ from baseball order. Reject stale arrivals.
    latest, keep = -1, []
    for i, stamp in zip(delayed.index, nanos(delayed.pitch_end_time)):
        if stamp > latest:
            keep.append(i)
            latest = stamp
    return delayed.loc[keep]


def game_frame(home, away, updates, execution, with_labels):
    if home.empty or updates.empty:
        return pd.DataFrame()
    pk, game_date = int(home.game_pk.iloc[0]), home.game_date.iloc[0]
    raw_start = int(nanos(updates.pitch_start_time).min())
    # Domain boundary only: it never determines whether an open position exits.
    stop = max(int(nanos(home.created_time).max()),
               int(nanos(away.created_time).max()) if len(away) else 0)
    start = (raw_start // (execution.step * NS) + 1) * execution.step * NS
    times = np.arange(start, stop + 1, execution.step * NS, dtype=np.int64)
    state = causal_states(updates, execution)
    available = nanos(state.event_available_time)
    state_age = (times - prior_values(available, available, times)) / NS
    inning = prior_values(available, state.inning_after.to_numpy(), times)
    atomic = prior_values(available, state.atomic_play_input.to_numpy(float), times)
    valid_state = (state_age <= 120) & (atomic == 1)
    sides = [print_sides(home), print_sides(away)]
    quotes = [proxy_quotes(s, times, execution.quote_age) for s in sides]
    rows = []
    for market, (tape, s) in enumerate(zip((home, away), sides)):
        if tape.empty:
            continue
        bid, ask = quotes[market]
        mid = (bid + ask) / 2
        other_mid = sum(quotes[1 - market]) / 2
        fair = prior_values(available, state.fair_after.to_numpy(), times)
        if market:
            fair = 1 - fair
        fair = np.where(valid_state, fair, np.nan)
        terminal = state[state.completed_event.notna() & state.atomic_play_input]
        pre_bid, pre_ask = proxy_quotes(s, nanos(terminal.pitch_start_time), execution.quote_age)
        before, after = terminal.fair_before.to_numpy(), terminal.fair_after.to_numpy()
        if market:
            before, after = 1 - before, 1 - after
        target = anchored_event_target((pre_bid + pre_ask) / 2, before, after)
        play_times = nanos(terminal.event_available_time)
        play_age = (times - prior_values(play_times, play_times, times)) / NS
        play_target = prior_values(play_times, target, times, 60.)
        # Once a newer pitch is observed, an old play target is no longer current.
        play_end = prior_values(play_times, nanos(terminal.pitch_end_time), times)
        state_end = prior_values(available, nanos(state.pitch_end_time), times)
        play_target = np.where(play_end == state_end, play_target, np.nan)
        ordered = tape.sort_values(["created_time", "trade_id"])
        trade_times = nanos(ordered.created_time)
        quantities = ordered.count_fp.to_numpy(float)
        signed = quantities * np.where(ordered.taker_outcome_side.eq("yes"), 1, -1)
        now = np.searchsorted(trade_times, times, side="left")
        prev = np.searchsorted(trade_times, times - 30 * NS, side="left")
        total = np.r_[0., quantities.cumsum()]
        flow = np.r_[0., signed.cumsum()]
        volume = total[now] - total[prev]
        buy_limit = np.minimum(.99, ask + execution.penalty)
        sell_limit = np.maximum(.01, bid - execution.penalty)
        buy_time, buy_price = fill_evidence(s, times, buy_limit, execution, True)
        sell_time, sell_price = fill_evidence(s, times, sell_limit, execution, False)
        frame = pd.DataFrame({"game_pk": pk, "game_date": game_date, "market": market,
            "ticker": str(tape.market_ticker.iloc[0]), "time": times,
            "outcome": int(home.home_win.iloc[0]) if not market else 1 - int(home.home_win.iloc[0]),
            "mid": mid, "bid": bid, "ask": ask, "spread": ask - bid,
            "other_gap": 1 - other_mid - mid, "flow_30": (flow[now] - flow[prev]) / np.maximum(volume, 1),
            "volume_30": np.log1p(volume), "fair_gap": fair - mid, "play_gap": play_target - mid,
            "state_age": state_age, "play_age": play_age, "inning": inning,
            "score_diff": prior_values(available, state.score_diff_after.to_numpy(), times) * (-1 if market else 1),
            "outs": prior_values(available, state.outs_when_up_after.to_numpy(), times), "home_side": 1 - market,
            "entry_allowed": valid_state & (inning <= 7) & (inning >= 1) & (ask >= .05) & (ask <= .95),
            "buy_limit": buy_limit, "sell_limit": sell_limit,
            "buy_time": buy_time, "buy_price": buy_price, "sell_time": sell_time, "sell_price": sell_price})
        for lookback in (10, 30, 120):
            past_bid, past_ask = proxy_quotes(s, times - lookback * NS, execution.quote_age)
            frame[f"return_{lookback}"] = mid - (past_bid + past_ask) / 2
        if with_labels:
            for horizon in (60, 300):
                future_bid, _ = proxy_quotes(s, times + horizon * NS, execution.quote_age)
                # Do not extrapolate a training label beyond the archive boundary.
                frame[f"label_{horizon}"] = np.where(times + horizon * NS <= stop, future_bid - mid, np.nan)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_chunk(job):
    chunk, start, end, execution_dict, labels = job
    execution = Execution(**execution_dict)
    directory = ROOT / "data/reboot" / chunk
    metadata = json.loads((directory / "state_updates.metadata.json").read_text())
    if metadata.get("state_contract") != "atomic_pitch_or_play_v2":
        raise ValueError("Noncausal state input")
    for key, name in (("model_sha256", "local_win_expectancy.cbm"), ("prior_sha256", "mlb_pregame_prior.json")):
        if metadata[key] != digest(MODELS / name):
            raise ValueError("Frozen model/state provenance mismatch")
    filters = [("game_date", ">=", start), ("game_date", "<=", end)]
    home, away, updates = [pd.read_parquet(directory / name, filters=filters) for name in (
        "home_market_trades.parquet", "away_market_trades.parquet", "state_updates.parquet")]
    ag = dict(tuple(away.groupby("game_pk")))
    ug = dict(tuple(updates.groupby("game_pk")))
    frames = [game_frame(h, ag.get(pk, away.iloc[:0]), ug[pk], execution, labels)
              for pk, h in home.groupby("game_pk")]
    print(f"Built {chunk}: {len(frames)} games", flush=True)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build(start, end, execution, workers, labels=False):
    dataset = json.loads((ROOT / "data/reboot/dataset.json").read_text())
    chunks = [c for c in dataset["chunks"] if date.fromisoformat(c[:10]) <= end
              and date.fromisoformat(c[11:]) >= start]
    jobs = [(c, start, end, asdict(execution), labels) for c in chunks]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        frames = list(pool.map(build_chunk, jobs))
    frame = pd.concat(frames, ignore_index=True).sort_values(["time", "game_pk", "market"]).reset_index(drop=True)
    if frame.duplicated(["time", "game_pk", "market"]).any():
        raise ValueError("Overlapping game data")
    return frame


def predict(frame, models):
    for horizon, model in models.items():
        frame[f"prediction_{horizon}"] = model.predict(frame[FEATURES])


def signals(frame, policy, execution):
    family = policy.family
    if family.startswith("revert_"):
        delta = -frame[f"return_{family.split('_')[1]}"].to_numpy()
    elif family.startswith("momentum_"):
        delta = frame[f"return_{family.split('_')[1]}"].to_numpy()
    else:
        column = {"cross_market": "other_gap", "state_value": "fair_gap", "all_play": "play_gap",
                  "learned": f"prediction_{policy.hold}", "activity_control": "fair_gap"}[family]
        delta = frame[column].to_numpy()
    price = frame.ask.to_numpy()
    # The learned label already uses a future bid; other targets are midpoints.
    exit_spread = 0 if family == "learned" else frame.spread.to_numpy() / 2
    edge = frame.mid.to_numpy() + delta - price - exit_spread - .14 * price * (1 - price) - 2 * execution.penalty
    return np.where(frame.entry_allowed.to_numpy() & np.isfinite(edge) & (edge >= policy.edge), edge, -np.inf)


TRADE_COLUMNS = ["game_pk", "side", "ticker", "entry_time", "exit_time", "entry_price", "exit_price",
                 "contracts", "fees", "pnl", "exit_reason", "entry_decision_time"]


def simulate(frame, policy, execution, starting_cash=100.):
    """Chronological shared portfolio with actual pending cash reservations.

    Evidence arrays contain counterfactual outcomes for every tick. They are
    read only AFTER deciding and reserving an order using causal features.
    """
    if not np.isfinite(starting_cash) or starting_cash <= 0:
        raise ValueError("Invalid bankroll")
    edge = signals(frame, policy, execution)
    arrays = {c: frame[c].to_numpy() for c in ("game_pk", "time", "market", "ticker", "outcome",
        "buy_limit", "sell_limit", "buy_time", "buy_price", "sell_time", "sell_price")}
    # Consecutive rows form one game's decision, with independent team markets.
    starts = np.r_[0, 1 + np.flatnonzero((arrays["time"][1:] != arrays["time"][:-1]) |
                                       (arrays["game_pk"][1:] != arrays["game_pk"][:-1])), len(frame)]
    positions, pending, events, rows = {}, set(), [], []
    cash, locked, peak_locked = starting_cash, 0., 0.
    counter = {"entry_orders": 0, "exit_orders": 0, "expired_entries": 0, "expired_exits": 0,
               "cash_rejected_orders": 0}
    serial = 0

    def close(pk, stamp, price, reason):
        nonlocal cash, locked
        pos = positions.pop(pk)
        exit_fee = taker_fee(1, price) if reason != "settlement" else 0.
        proceeds = price - exit_fee
        if reason != "settlement":
            cash += proceeds
            locked -= pos["cost"]
        rows.append({"game_pk": pk, "side": "home_yes" if pos["market"] == 0 else "away_yes",
            "ticker": pos["ticker"], "entry_time": pd.Timestamp(pos["entry_time"], tz="UTC"),
            "exit_time": pd.Timestamp(stamp, tz="UTC") if stamp else pd.NaT,
            "entry_price": pos["price"], "exit_price": price, "contracts": 1.,
            "fees": pos["fee"] + exit_fee, "pnl": proceeds - pos["cost"], "exit_reason": reason,
            "entry_decision_time": pd.Timestamp(pos["decision"], tz="UTC")})

    def advance(now):
        nonlocal cash, locked
        while events and events[0][0] < now:
            stamp, _, kind, pk, data = heapq.heappop(events)
            pending.remove(pk)
            if kind == "buy":
                reserve, index, decision = data
                if arrays["buy_time"][index]:
                    price = arrays["buy_price"][index]
                    fee = taker_fee(1, price)
                    cost = price + fee
                    cash += reserve - cost
                    locked -= reserve - cost
                    positions[pk] = {"price": price, "fee": fee, "cost": cost,
                        "entry_time": stamp, "decision": decision, "market": arrays["market"][index],
                        "ticker": arrays["ticker"][index], "outcome": arrays["outcome"][index]}
                else:
                    cash += reserve
                    locked -= reserve
                    counter["expired_entries"] += 1
            elif arrays["sell_time"][data]:
                close(pk, stamp, arrays["sell_price"][data], "timed_sale")
            else:
                counter["expired_exits"] += 1

    for a, b in zip(starts[:-1], starts[1:]):
        now, pk = int(arrays["time"][a]), int(arrays["game_pk"][a])
        advance(now)
        if pk in pending:
            continue
        if pk in positions:
            pos = positions[pk]
            if now < pos["entry_time"] + policy.hold * NS:
                continue
            candidates = [i for i in range(a, b) if arrays["market"][i] == pos["market"]
                          and np.isfinite(arrays["sell_limit"][i])]
            if not candidates:
                continue
            index = candidates[0]
            when = arrays["sell_time"][index] or now + round((execution.latency + execution.window) * NS)
            kind, data = "sell", index
            counter["exit_orders"] += 1
        else:
            index = a + int(np.argmax(edge[a:b]))
            if not np.isfinite(edge[index]):
                continue
            limit = arrays["buy_limit"][index]
            reserve = limit + taker_fee(1, limit)
            if reserve > cash + 1e-9:
                counter["cash_rejected_orders"] += 1
                continue
            cash -= reserve
            locked += reserve
            peak_locked = max(peak_locked, locked)
            when = arrays["buy_time"][index] or now + round((execution.latency + execution.window) * NS)
            kind, data = "buy", (reserve, index, now)
            counter["entry_orders"] += 1
        serial += 1
        pending.add(pk)
        heapq.heappush(events, (int(when), serial, kind, pk, data))
    advance(np.iinfo(np.int64).max)
    for pk in list(positions):
        close(pk, 0, float(positions[pk]["outcome"]), "settlement")
    result = pd.DataFrame(rows, columns=TRADE_COLUMNS)
    for c in ("pnl", "fees", "contracts", "entry_price"):
        result[c] = result[c].astype(float)
    counter.update({"starting_cash": starting_cash, "peak_reserved_and_invested": peak_locked,
                    "ending_equity": starting_cash + float(result.pnl.sum()),
                    "settlement_cash_reused": False, "pending_cash_reserved": True})
    return result, counter


def metrics(frame, trades, counters, start, end):
    dates = frame.groupby("game_pk").game_date.first().to_dict()
    base = summarize(trades, dates, start, end)
    base.update(counters)
    base.update({"entries_per_game": len(trades) / len(dates),
        "game_coverage": trades.game_pk.nunique() / len(dates),
        "every_game_traded": trades.game_pk.nunique() == len(dates),
        "settlements": int(trades.exit_reason.eq("settlement").sum()),
        "mean_net_cents_per_entry": 100 * float(trades.pnl.mean()) if len(trades) else 0.})
    return base


def input_manifest(start, end):
    dataset = ROOT / "data/reboot/dataset.json"
    files = [dataset, Path(__file__), ROOT / "research/validation.py", ROOT / "research/CONTINUOUS_PROTOCOL.md",
             ROOT / "hit_reversion_strategy/scripts/backtest.py", ROOT / "hit_reversion_strategy/trade_tape_strategy/strategy.py",
             ROOT / "hit_reversion_strategy/trade_tape_strategy/hybrid.py", MODELS / "event_observation_latency.json",
             MODELS / "local_win_expectancy.cbm", MODELS / "mlb_pregame_prior.json"]
    for chunk in json.loads(dataset.read_text())["chunks"]:
        if date.fromisoformat(chunk[:10]) <= end and date.fromisoformat(chunk[11:]) >= start:
            files.extend(sorted((dataset.parent / chunk).glob("*.*")))
    return {str(p.relative_to(ROOT)): digest(p) for p in files}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def verify_hashes(hashes):
    if any(digest(ROOT / name) != expected for name, expected in hashes.items()):
        raise RuntimeError("Inputs changed during the research run")


def train_models(frame, output):
    models, diagnostics = {}, {}
    development = frame[frame.game_date <= TRAIN_END]
    for horizon in (60, 300):
        sample = development[development.entry_allowed & development[f"label_{horizon}"].notna()
                             & development.mid.notna()]
        counts = sample.groupby("game_pk").game_pk.transform("size")
        weight = 1 / counts.to_numpy(float)
        weight /= weight.mean()
        model = CatBoostRegressor(iterations=250, depth=4, learning_rate=.04, l2_leaf_reg=20,
                                  loss_function="RMSE", random_seed=20260927, thread_count=4,
                                  verbose=False, allow_writing_files=False)
        model.fit(sample[FEATURES], sample[f"label_{horizon}"], sample_weight=weight)
        model.save_model(str(output / f"markout_{horizon}.cbm"))
        models[horizon] = model
        diagnostics[str(horizon)] = {"training_rows": len(sample), "training_games": sample.game_pk.nunique(),
                                    "features": FEATURES, "parameters": model.get_params()}
    return models, diagnostics


def develop(args):
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose an empty output directory; research trials are never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    execution = Execution()
    hashes = input_manifest(date(2026, 8, 12), SELECT_END)
    manifest = {"started_at": datetime.now(timezone.utc).isoformat(), "hashes": hashes,
        "execution": asdict(execution), "candidates": [asdict(p) for p in policies()],
        "partitions": {"development": ["2026-08-12", str(TRAIN_END)],
                       "selection": ["2026-09-02", str(SELECT_END)],
                       "final_check": ["2026-09-16", str(FINAL_END)]},
        "status": "chronological_research_not_untouched_holdout", "deployment_ready": False}
    write_json(out / "manifest.json", manifest)
    frame = build(date(2026, 8, 12), SELECT_END, execution, args.workers, labels=True)
    models, diagnostics = train_models(frame, out)
    predict(frame, models)
    results = {}
    for policy in policies():
        results[policy.name] = {"policy": asdict(policy)}
        for name, start, end in (("development", date(2026, 8, 12), TRAIN_END),
                                 ("selection", date(2026, 9, 2), SELECT_END)):
            part = frame[(frame.game_date >= start) & (frame.game_date <= end)]
            trades, counters = simulate(part, policy, execution)
            result = metrics(part, trades, counters, start, end)
            results[policy.name][name] = result
        print(f"{policy.name}: dev ${results[policy.name]['development']['pnl']:.2f}; "
              f"select ${result['pnl']:.2f}, {result['entries_per_game']:.2f}/game, "
              f"{result['game_coverage']:.0%} coverage", flush=True)
    eligible, active, rest = [], [], []
    for name, result in results.items():
        if result["policy"]["family"] == "activity_control":
            continue
        activity = all(result[p]["entries_per_game"] >= 1 and result[p]["game_coverage"] >= .8
                       for p in ("development", "selection"))
        passing = activity and all(result[p]["pnl"] > 0 for p in ("development", "selection"))
        rest.append(name)
        if activity:
            active.append(name)
        if passing:
            eligible.append(name)
    candidate = max(eligible or active or rest, key=lambda name: results[name]["selection"]["day_block_bootstrap_pnl_95pct"][0])
    selection = {"frozen_at": datetime.now(timezone.utc).isoformat(), "candidate": candidate,
        "policy": results[candidate]["policy"], "development_selection_pass": candidate in eligible,
        "eligible_candidates": eligible, "activity_candidates": active,
        "fallback": not bool(eligible), "deployment_ready": False,
        "model_hashes": {str(h): digest(out / f"markout_{h}.cbm") for h in (60, 300)}}
    for horizon in (60, 300):
        part = frame[(frame.game_date > TRAIN_END) & frame[f"label_{horizon}"].notna() & frame.mid.notna()]
        error = part[f"prediction_{horizon}"] - part[f"label_{horizon}"]
        diagnostics[str(horizon)]["selection_rmse"] = float(np.sqrt(np.mean(error ** 2)))
        diagnostics[str(horizon)]["zero_change_rmse"] = float(np.sqrt(np.mean(part[f"label_{horizon}"] ** 2)))
    verify_hashes(hashes)
    write_json(out / "development_selection.json", results)
    write_json(out / "model_diagnostics.json", diagnostics)
    write_json(out / "selection.json", selection)
    print(f"Frozen research candidate: {candidate}; passed development/selection: {candidate in eligible}", flush=True)


STRESSES = {"reference": {}, "latency_1_5s": {"latency": 1.5}, "latency_3s": {"latency": 3.},
    "two_cent_cost": {"penalty": .02}, "one_percent_volume": {"participation": .01},
    "fifty_ms_window": {"window": .05}, "publication_plus_2s": {"state_delay": 4.}}


def check(args):
    out = args.output_dir
    # Final checks must use the exact feature/execution implementation and
    # development inputs that produced the frozen selection.
    verify_hashes(json.loads((out / "manifest.json").read_text())["hashes"])
    selection_path = out / "selection.json"
    selection = json.loads(selection_path.read_text())
    if (out / "final_check.json").exists():
        raise ValueError("Final check already exists; do not retune on it")
    models = {}
    for h in (60, 300):
        path = out / f"markout_{h}.cbm"
        if digest(path) != selection["model_hashes"][str(h)]:
            raise ValueError("Selected model changed")
        models[h] = CatBoostRegressor().load_model(str(path))
    policy = Policy(**selection["policy"])
    hashes = input_manifest(date(2026, 9, 16), FINAL_END)
    manifest = {"started_at": datetime.now(timezone.utc).isoformat(), "hashes": hashes,
                "selection_sha256": digest(selection_path), "stresses": STRESSES}
    write_json(out / "final_manifest.json", manifest)
    results = {}
    for name, changes in STRESSES.items():
        execution = replace(Execution(), **changes)
        frame = build(date(2026, 9, 16), FINAL_END, execution, args.workers)
        predict(frame, models)
        trades, counters = simulate(frame, policy, execution)
        results[name] = metrics(frame, trades, counters, date(2026, 9, 16), FINAL_END)
        trades.to_csv(out / f"{name}_trades.csv", index=False)
        print(f"Final {name}: {len(trades)} entries, ${results[name]['pnl']:.2f}, "
              f"{results[name]['game_coverage']:.0%} coverage", flush=True)
    verify_hashes(hashes)
    if digest(selection_path) != manifest["selection_sha256"]:
        raise RuntimeError("Selection changed during final check")
    write_json(out / "final_check.json", {"candidate": selection["candidate"], "deployment_ready": False,
        "development_selection_pass": selection["development_selection_pass"], "scenarios": results})
    report(out)


def report(out):
    selection = json.loads((out / "selection.json").read_text())
    trials = json.loads((out / "development_selection.json").read_text())
    final = json.loads((out / "final_check.json").read_text())
    lines = ["# Continuous strategy research", "", f"Frozen candidate: **{selection['candidate']}**.", "",
        f"Development/selection passed: **{selection['development_selection_pass']}**. Live remains disabled.", "",
        "One contract per entry; $100 initial cash per partition. Entries are simulated print-based fills, not verified fills.",
        "These previously examined dates are chronological checks, not an untouched holdout.", "",
        "## Frozen candidate", "", "| Partition/scenario | Games | Entries | Entries/game | Games traded | Net PnL | Fees | Day-bootstrap 95% |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    rows = {k: trials[selection["candidate"]][k] for k in ("development", "selection")}
    rows.update({f"final/{k}": v for k, v in final["scenarios"].items()})
    for name, r in rows.items():
        lo, hi = r["day_block_bootstrap_pnl_95pct"]
        lines.append(f"| {name} | {r['games_in_data']} | {r['trades']} | {r['entries_per_game']:.2f} | "
            f"{r['games_with_trades']} ({r['game_coverage']:.1%}) | ${r['pnl']:.2f} | ${r['fees']:.2f} | [${lo:.2f}, ${hi:.2f}] |")
    lines += ["", "## All trials", "", "Every declared candidate and both forced-activity controls appear below.", "",
        "| Candidate | Dev PnL | Selection PnL | Selection entries/game | Selection game coverage |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for name, r in trials.items():
        d, s = r["development"], r["selection"]
        lines.append(f"| {name} | ${d['pnl']:.2f} | ${s['pnl']:.2f} | {s['entries_per_game']:.2f} | {s['game_coverage']:.1%} |")
    lines += ["", "## Interpretation limits", "",
        "The price proxies are recent same-side trades, not historical bid/ask quotes. Print-based fills cannot establish queue, depth, or availability at order arrival. Maker fills are never inferred.", "",
        "Final MLB archives can contain corrections. State reception is modeled; the terminal latency profile came from one July sample. Final game boundaries delimit the available archive and unfilled exits settle.", "",
        "Bootstrap intervals describe day variation under these assumptions. They do not correct multiple testing, earlier inspection of this period, or execution-model error. The final period contains only 11 days.", "",
        "Cash is reserved for submissions, all games share one chronological portfolio, and settlement proceeds are withheld. Each partition starts with a fresh $100; partition returns must not be added as a compounded live path.", "",
        "The frequency goal is reported as both entries/game and game coverage. No failed candidate is enabled or described as profitable live."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["develop", "check"])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "research/results/continuous")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    (develop if args.command == "develop" else check)(args)


if __name__ == "__main__":
    main()
