"""Freeze, replay, and periodically score selective maker experiments."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

from research.fill_quality import write_report
from research.maker_portfolio import CashAccount, PortfolioClock
from research.record import BookSequence
from research.selective_maker import AnchoredForecast, SelectiveConfig, SelectiveGame, candidates
from research.settle_forward import observed_settlements
from research.slate_study import ROOT, digest, empty_output, snapshot
from research.sportsbook import EASTERN, OddsTape, timestamp


SOURCE_NAMES = ("selective_study.py", "selective_maker.py", "sportsbook.py", "fill_quality.py",
                "maker_portfolio.py", "maker_study.py", "passive.py", "record.py", "slate_study.py",
                "settle_forward.py", "forward_value.py", "state_refresh.py", "market_correction.py",
                "NEXT_EXPERIMENT.md")


def configuration_digest(policies):
    return hashlib.sha256(json.dumps(policies, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def freeze(output, model_dir=None):
    empty_output(output)
    files = [ROOT / "research" / name for name in SOURCE_NAMES]
    files.append(ROOT / "settlement_value_strategy/play_eligibility.py")
    model_files = []
    if model_dir:
        model_files = [model_dir / name for name in ("model.cbm", "prior.json", "manifest.json")]
        training = json.loads((model_dir / "manifest.json").read_text())
        if training["cutoff_exclusive"] >= datetime.now(timezone.utc).date().isoformat():
            raise ValueError("Baseball training cutoff is not historical")
    policies = {name: asdict(config) for name, config in candidates().items()}
    protocol = {"frozen_at": datetime.now(timezone.utc).isoformat(), "version": 1,
                "policies": policies, "configuration_sha256": configuration_digest(policies),
                "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in files},
                "model_dir": str(model_dir.resolve()) if model_dir else None,
                "model_hashes": {str(p.resolve()): digest(p) for p in model_files},
                "allowed_latencies": [.68, 1.5, 3.], "starting_cash": 100.,
                "validation_rule": "Capture starts after freeze; game dates must be later than freeze's Eastern date",
                "actual_exchange_orders": False, "deployment_ready": False}
    path = output / "protocol.json"
    path.write_text(json.dumps(protocol, indent=2, allow_nan=False))
    for source in files:
        archived = output / "source" / source.relative_to(ROOT)
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, archived)
        if digest(archived) != protocol["source_hashes"][str(source.relative_to(ROOT))]:
            raise RuntimeError("Source changed during protocol freeze")
    return path


def verify(protocol_path):
    spec = json.loads(protocol_path.read_text())
    if spec.get("version") != 1 or configuration_digest(spec["policies"]) != spec["configuration_sha256"]:
        raise ValueError("Frozen configuration mismatch")
    for name, expected in spec["source_hashes"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    for name, expected in spec["model_hashes"].items():
        if digest(Path(name)) != expected:
            raise ValueError("Frozen model changed")
    return spec


def reconcile_ledger(game):
    """Independently rebuild signed inventory and realized PnL from all fills."""
    position = basis = realized = fees = 0.
    previous_time = -math.inf
    for fill in game["fills"]:
        quantity, price, charge = fill["quantity"], fill["price"], fill["fee"]
        if (not all(math.isfinite(v) for v in (quantity, price, charge, fill["received_seconds"]))
                or quantity <= 0 or not 0 < price < 1 or charge < 0):
            raise ValueError("Invalid fill ledger")
        if fill["received_seconds"] < previous_time:
            raise ValueError("Fill ledger clock moved backwards")
        previous_time = fill["received_seconds"]
        if fill.get("ticker") != game["ticker"] or fill["side"] not in {"yes", "no"}:
            raise ValueError("Fill belongs to a different market or side")
        sign = 1 if fill["side"] == "yes" else -1
        if fill["action"] == "sell":
            if position * sign <= 0 or quantity > abs(position) + 1e-9:
                raise ValueError("Sell would create an unreported position")
            closing, opening = quantity, 0.
            realized += quantity * (price - basis) - charge
            position -= sign * quantity
        elif fill["action"] == "buy":
            closing = min(quantity, abs(position)) if position * sign < 0 else 0.
            opening = quantity - closing
            realized += closing * (1 - price - basis) - charge * closing / quantity
            position += sign * closing
            if opening > 1e-9:
                old = abs(position)
                basis = (old * basis + opening * price + charge * opening / quantity) / (old + opening)
                position += sign * opening
        else:
            raise ValueError("Unknown fill action")
        if abs(position) < 1e-9:
            position, basis = 0., 0.
        fees += charge
        for expected, actual in ((opening, fill["opening_quantity"]), (closing, fill["closing_quantity"]),
                                 (position, fill["position_after"]), (realized, fill["realized_pnl_after"])):
            if abs(expected - actual) > 1e-7:
                raise ValueError("Fill accounting does not reconcile")
    for expected, actual in ((position, game["inventory"]), (basis, game["inventory_cost_per_contract"]),
                             (realized, game["realized_pnl"]), (fees, game["fees"])):
        if abs(expected - actual) > 1e-7:
            raise ValueError("Game accounting does not reconcile with fill ledger")


def settlement_mark(game, terminals, first_wall, first_monotonic):
    reconcile_ledger(game)
    value = game["inventory_liquidation_mark"]
    terminal = terminals.get(game["ticker"])
    settled = False
    if terminal is not None and game["fills"]:
        latest = max(f["received_seconds"] for f in game["fills"])
        if (first_wall + latest >= terminal["exchange_time"]
                or first_monotonic + int(latest * 1e9) >= terminal["observed_monotonic_ns"]):
            raise ValueError("A fill occurs after settlement")
    if abs(game["inventory"]) > 1e-9 and terminal:
        payout = terminal["value"] if game["inventory"] > 0 else 1 - terminal["value"]
        value = abs(game["inventory"]) * payout
        settled = True
    pnl = game["realized_pnl"] + value - abs(game["inventory"]) * game["inventory_cost_per_contract"]
    return {"net_shadow_pnl": pnl, "closed_shadow_pnl": game["realized_pnl"], "inventory_value": value,
            "position_settled": settled, "unsettled_quantity": 0. if settled else abs(game["inventory"]),
            "mark_source": "observed_exchange_settlement" if settled else "depth_checked_bid_mark" if game["inventory"] else "closed_inventory",
            "settlement_observed_at": terminal["observed_at"] if settled else None}


def validate_capture_role(spec, first_record, slate, role):
    if role not in {"development", "validation"}:
        raise ValueError("Unknown evaluation role")
    if role == "validation":
        if timestamp(first_record["recorded_at"]) <= timestamp(spec["frozen_at"]):
            raise ValueError("Validation capture predates the frozen protocol")
        freeze_day = datetime.fromtimestamp(timestamp(spec["frozen_at"]), EASTERN).date()
        if any(datetime.fromtimestamp(timestamp(g["scheduled_time"]), EASTERN).date() <= freeze_day for g in slate["games"]):
            raise ValueError("Validation requires games on a later date; today's data are development")


def evaluate(capture, slate_path, protocol_path, output, *, role="development", odds=None, settlements=None, latencies=(.68, 1.5, 3.)):
    spec = verify(protocol_path)
    if not latencies or len(set(latencies)) != len(latencies) or any(x not in spec["allowed_latencies"] for x in latencies):
        raise ValueError("Only distinct frozen latency scenarios are permitted")
    slate = json.loads(slate_path.read_text())
    if not slate["games"] or len({g["game_pk"] for g in slate["games"]}) != len(slate["games"]):
        raise ValueError("Empty or ambiguous slate")
    if slate.get("scheduled_game_count", len(slate["games"])) < len(slate["games"]):
        raise ValueError("Scheduled game denominator is smaller than the mapped slate")
    with gzip.open(capture, "rt") as stream:
        first_row = json.loads(next(stream))
    if first_row["type"] != "capture_start":
        raise ValueError("Replay must begin with capture_start")
    validate_capture_role(spec, first_row, slate, role)
    empty_output(output)
    inputs = [capture, slate_path, protocol_path] + ([odds] if odds else []) + ([settlements] if settlements else [])
    hashes = {str(p.resolve()): digest(p) for p in inputs}
    manifest = {"started_at": datetime.now(timezone.utc).isoformat(), "role": role, "input_hashes": hashes,
                "source_hashes": spec["source_hashes"], "protocol": str(protocol_path.resolve()),
                "latencies": list(latencies), "actual_exchange_orders": False, "deployment_ready": False}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    tape = OddsTape.read(odds, slate) if odds else None
    terminals, settlement_errors = observed_settlements(settlements) if settlements else ({}, 0)
    model = ratings = None
    if spec["model_dir"]:
        from catboost import CatBoostClassifier
        model_dir = Path(spec["model_dir"])
        prior = json.loads((model_dir / "prior.json").read_text())
        if any(g["scheduled_time"][:10] < prior["cutoff_exclusive"] for g in slate["games"]):
            raise ValueError("Baseball prior could contain future outcomes")
        model = CatBoostClassifier().load_model(str(model_dir / "model.cbm"))
        ratings = prior["ratings"]
    engines, clocks, by_game, by_ticker = {}, {}, {}, {}
    seen_tickers = set()
    for game in slate["games"]:
        pair = [game["market_ticker"], game["away_market_ticker"]]
        if pair[0] == pair[1] or any(t in seen_tickers for t in pair):
            raise ValueError("Ambiguous game/market identity")
        seen_tickers.update(pair)
    for name, config in spec["policies"].items():
        for delay in latencies:
            key = f"{name}_latency_{delay}"
            account = CashAccount.funded(spec["starting_cash"])
            policy = replace(SelectiveConfig(**config), submission_seconds=delay, cancellation_seconds=delay)
            cohort = []
            for game in slate["games"]:
                forecast = None
                if policy.reference_mode == "sportsbook_anchor" and model is not None:
                    from research.state_refresh import probability
                    from research.sportsbook import team_code
                    forecast = AnchoredForecast(game["game_pk"], model,
                        probability(team_code(game["home_code"]), team_code(game["away_code"]), ratings), tape)
                engine = SelectiveGame(game, policy, account, forecast)
                cohort.append(engine)
                by_game.setdefault(game["game_pk"], []).append(engine)
                for ticker in (engine.ticker, engine.peer_ticker):
                    by_ticker.setdefault(ticker, []).append(engine)
            engines[key] = cohort
            clocks[key] = PortfolioClock(cohort, account)
    first_ns = int(first_row["recorded_monotonic_ns"])
    first_wall = timestamp(first_row["recorded_at"])
    sequence = BookSequence()
    previous, last, ended, complete, error = first_ns, 0., False, False, None
    paths = {key: [] for key in clocks}
    next_mark = 0.
    with gzip.open(capture, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            stamp = int(row["recorded_monotonic_ns"])
            if stamp < previous:
                raise ValueError("Observation clock moved backwards")
            previous = stamp
            now = (stamp - first_ns) / 1e9
            if abs(timestamp(row["recorded_at"]) - first_wall - now) > 1.:
                error = "Receipt wall clock diverged from monotonic replay clock"
                break
            if row["type"] == "connection_gap" or (ended and row["type"] == "connection_start"):
                error = "Connection gap; exposed orders remain indeterminate"
                break
            if ended:
                complete = complete or row["type"] == "capture_end"
                continue
            try:
                for clock in clocks.values():
                    clock.advance_before(now)
                if row["type"] == "kalshi_message":
                    message = row["message"]
                    if message.get("type") == "error":
                        raise ValueError("Recorded subscription error")
                    sequence.observe(message)
                    body = message.get("msg") or {}
                    if message.get("type") == "trade":
                        if body.get("ts_ms") is None:
                            raise ValueError("Trade lacks exchange timestamp")
                        body["exchange_elapsed_seconds"] = float(body["ts_ms"]) / 1000 - first_wall
                    for engine in by_ticker.get(body.get("market_ticker"), []):
                        engine.observe(now, message)
                elif row["type"] == "mlb_observation":
                    for engine in by_game.get(row["game_pk"], []):
                        engine.baseball(now, row, first_wall)
                elif row["type"] == "connection_end":
                    ended = True
                elif row["type"] == "capture_end":
                    complete = True
            except ValueError as exc:
                error = str(exc)
                break
            last = now
            for clock in clocks.values():
                clock.measure()
            if now >= next_mark:
                for key, clock in clocks.items():
                    paths[key].append({"received_seconds": now, "equity": clock.summary()["equity"]})
                next_mark = now + 60.
    if error is None and not ended:
        for clock in clocks.values():
            clock.advance_before(last, inclusive=True)
    results = {}
    for key, cohort in engines.items():
        directory = output / key
        directory.mkdir()
        games = []
        for engine in cohort:
            game = engine.summary()
            game.update(fills=engine.fills, fill_markouts=engine.markouts, capture_complete=complete,
                        observed_seconds=last, continuity_error=error,
                        replay_valid=error is None and engine.book is not None and engine.peer_book is not None)
            game["settlement"] = settlement_mark(game, terminals, first_wall, first_ns)
            games.append(game)
            (directory / f"{engine.game_pk}.json").write_text(json.dumps(game, indent=2, allow_nan=False))
        portfolio = clocks[key].summary()
        paths[key].append({"received_seconds": last, "equity": portfolio["equity"]})
        peak, drawdown = spec["starting_cash"], 0.
        for mark in paths[key]:
            peak = max(peak, mark["equity"])
            drawdown = max(drawdown, peak - mark["equity"])
        net = sum(g["settlement"]["net_shadow_pnl"] for g in games)
        adjusted = portfolio["free_cash"] + portfolio["reserved_cash"] + sum(g["settlement"]["inventory_value"] for g in games)
        if abs(adjusted - spec["starting_cash"] - net) > 1e-6:
            raise ValueError("Settlement-adjusted shared account does not reconcile")
        entries = sum(g["entry_orders_filled"] for g in games)
        traded = sum(g["entry_orders_filled"] > 0 for g in games)
        attempts = sum(g["opening_order_attempts"] for g in games)
        opening_attempts_filled = sum(g["opening_attempts_filled"] for g in games)
        outstanding = sum(g["outstanding_orders"] for g in games)
        unsettled = sum(g["settlement"]["unsettled_quantity"] for g in games)
        valid = all(g["replay_valid"] for g in games)
        results[key] = {"games": len(games), "games_with_entries": traded, "entry_orders_filled": entries,
            "entries_per_mapped_game": entries / len(games), "mapped_game_coverage": traded / len(games),
            "scheduled_game_coverage": traded / slate.get("scheduled_game_count", len(games)),
            "opening_order_attempts": attempts, "opening_attempts_filled": opening_attempts_filled,
            "opening_fill_rate": opening_attempts_filled / attempts if attempts else None,
            "reducing_orders_with_opening_fills": sum(g["reducing_orders_with_opening_fills"] for g in games),
            "fees": sum(g["fees"] for g in games), "net_shadow_pnl": net,
            "closed_shadow_pnl": sum(g["realized_pnl"] for g in games),
            "settled_positions": sum(g["settlement"]["position_settled"] for g in games),
            "unsettled_quantity": unsettled, "outstanding_orders": outstanding,
            "all_replays_valid": valid, "games_observed_live": sum(g["game_observed_live"] for g in games),
            "games_observed_final": sum(g["game_observed_final"] for g in games),
            "complete_policy_evaluation": bool(complete and valid and not outstanding and unsettled < 1e-9
                and all(g["game_observed_final"] for g in games)),
            "portfolio": portfolio, "settlement_adjusted_equity": adjusted,
            "sampled_marked_drawdown_60s": drawdown, "equity_samples": paths[key]}
    summary = {"role": role, "capture_complete": complete, "continuity_error": error,
        "observed_seconds": last, "scheduled_game_count": slate.get("scheduled_game_count", len(slate["games"])),
        "unmapped_game_pks": slate.get("unmapped_game_pks", []), "candidates": results,
        "sportsbook_snapshots": len(tape.times) if tape else 0,
        "sportsbook_rejections": dict(tape.rejected) if tape else {},
        "settlement_observation_errors": settlement_errors, "cash_recycled_from_settlement_during_replay": False,
        "actual_exchange_orders": False, "deployment_ready": False}
    if any(digest(Path(p)) != expected for p, expected in hashes.items()):
        raise RuntimeError("Frozen replay inputs changed")
    verify(protocol_path)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    write_report(output, output / "fill_quality")
    lines = ["# Selective maker experiment", "", f"Role: {role}. All fills are modeled shadow fills.", "",
             "| Scenario | Entries | Games entered / mapped | Net shadow PnL incl. marks | Fees | Unsettled qty |",
             "|---|---:|---:|---:|---:|---:|"]
    for key, row in results.items():
        lines.append(f"| {key} | {row['entry_orders_filled']} | {row['games_with_entries']} / {row['games']} | "
                     f"${row['net_shadow_pnl']:+.4f} | ${row['fees']:.4f} | {row['unsettled_quantity']:.2f} |")
    lines += ["", "Unsettled inventory uses conservative visible-depth marks. Pending orders remain exposed.",
              "The activity target is one entry in each game; attempts and contracts are not entries.",
              "No result is accepted automatically, and a zero-trade prefix provides no profitability evidence."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({key: {k: r[k] for k in ("entry_orders_filled", "games_with_entries", "net_shadow_pnl", "all_replays_valid")}
                      for key, r in results.items()}), flush=True)
    return summary


def watch(args):
    empty_output(args.output_dir)
    expected = digest(args.protocol)
    deadline, index = time.monotonic() + args.duration, 0
    while time.monotonic() < deadline:
        if digest(args.protocol) != expected:
            raise RuntimeError("Frozen protocol changed")
        verify(args.protocol)
        index += 1
        prefix = args.output_dir / f"prefix_{index:03d}"
        capture, metadata = snapshot(args.capture, prefix)
        odds = snapshot(args.odds, prefix / "odds")[0] if args.odds else None
        settlements = snapshot(args.settlements, prefix / "settlements")[0] if args.settlements else None
        result = evaluate(capture, args.slate, args.protocol, prefix / "replay", role=args.role,
                          odds=odds, settlements=settlements, latencies=args.latencies)
        if result["continuity_error"] or (metadata["capture_complete"] and all(r["complete_policy_evaluation"] for r in result["candidates"].values())):
            break
        next_run = min(time.monotonic() + args.interval, deadline)
        while time.monotonic() < next_run:
            time.sleep(min(30., max(0., next_run - time.monotonic())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "evaluate", "watch"])
    parser.add_argument("capture", type=Path, nargs="?")
    parser.add_argument("--slate", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--odds", type=Path)
    parser.add_argument("--settlements", type=Path)
    parser.add_argument("--role", choices=["development", "validation"], default="development")
    parser.add_argument("--latencies", type=float, nargs="+", default=[.68, 1.5, 3.])
    parser.add_argument("--interval", type=float, default=900.)
    parser.add_argument("--duration", type=float, default=21600.)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        print(freeze(args.output_dir, args.model_dir))
    elif not args.capture or not args.slate or not args.protocol:
        parser.error("evaluate/watch require capture, --slate and --protocol")
    elif args.command == "evaluate":
        evaluate(args.capture, args.slate, args.protocol, args.output_dir, role=args.role,
                 odds=args.odds, settlements=args.settlements, latencies=args.latencies)
    else:
        if not math.isfinite(args.duration) or not math.isfinite(args.interval) or not 60 <= args.interval <= 3600 or not 60 <= args.duration <= 86400:
            parser.error("Invalid watch interval or duration")
        watch(args)


if __name__ == "__main__":
    main()
