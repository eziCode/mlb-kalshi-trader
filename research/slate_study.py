"""Freeze capture prefixes and replay the declared passive slate candidates.

Examples:
  python -m research.slate_study snapshot LIVE_CAPTURE --output-dir SNAPSHOT_DIR
  python -m research.slate_study evaluate SNAPSHOT_DIR/capture.jsonl.gz --slate SLATE_JSON --output-dir RESULTS
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import zlib

from research.maker_study import MakerStudyReplay, candidates
from research.maker_portfolio import CashAccount, PortfolioClock, PortfolioGame
from research.record import BookSequence

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def empty_output(path):
    if path.exists() and any(path.iterdir()):
        raise ValueError("Choose an empty output directory; completed results are retained")
    path.mkdir(parents=True, exist_ok=True)


def snapshot(source, destination):
    """Freeze complete JSON lines from an actively written gzip file.

    Read exactly the byte count observed at start, never the growing tail. A
    completed gzip envelope around a prefix does not make it a complete game.
    """
    empty_output(destination)
    count, remaining = 0, source.stat().st_size
    original_size = remaining
    decoder, pending = zlib.decompressobj(16 + zlib.MAX_WBITS), b""
    last_type = None
    target = destination / "capture.jsonl.gz"
    with source.open("rb") as raw, gzip.open(target, "wb") as output:
        while remaining:
            block = raw.read(min(1024 * 1024, remaining))
            if not block:
                raise ValueError("Source shrank while taking the snapshot")
            remaining -= len(block)
            pending += decoder.decompress(block)
            rows = pending.split(b"\n")
            pending = rows.pop()
            for line in rows:
                row = json.loads(line)
                last_type = row["type"]
                output.write(line + b"\n")
                count += 1
    metadata = {"source": str(source), "source_bytes_at_start": original_size,
        "snapshotted_at": datetime.now(timezone.utc).isoformat(), "complete_records": count,
        "capture_complete": bool(decoder.eof and last_type == "capture_end"),
        "partial_trailing_json_bytes_omitted": len(pending), "sha256": digest(target)}
    (destination / "snapshot.json").write_text(json.dumps(metadata, indent=2))
    return target, metadata


def evaluate(capture, slate_path, output, latency=.68, penalty=.01, shared_cash=None, policy=None):
    empty_output(output)
    slate = json.loads(slate_path.read_text())
    policies = {name: replace(config, submission_seconds=latency, cancellation_seconds=latency,
                              liquidation_penalty=penalty) for name, config in candidates().items()}
    if policy is not None:
        policies = {policy: policies[policy]}
    files = [Path(__file__), ROOT / "research/maker_study.py", ROOT / "research/passive.py",
             ROOT / "research/record.py", ROOT / "research/maker_portfolio.py",
             ROOT / "research/PASSIVE_SLATE_PROTOCOL.md"]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
    manifest = {"started_at": datetime.now(timezone.utc).isoformat(), "capture_sha256": digest(capture),
                "slate_sha256": digest(slate_path), "source_hashes": hashes,
                "policies": {name: asdict(config) for name, config in policies.items()},
                "shared_starting_cash": shared_cash,
                "role": "development", "deployment_ready": False}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    engines = {name: {} for name in policies}
    accounts = {name: CashAccount.funded(shared_cash) for name in policies} if shared_cash is not None else {}
    by_game, by_ticker = {}, {}
    for game in slate["games"]:
        pk, home, away = game["game_pk"], game["market_ticker"], game["away_market_ticker"]
        cohort = []
        for name, config in policies.items():
            engine = (PortfolioGame(home, away, pk, config, accounts[name]) if accounts
                      else MakerStudyReplay(home, away, pk, config))
            engines[name][pk] = engine
            cohort.append(engine)
        by_game[pk] = cohort
        for ticker in (home, away):
            if ticker in by_ticker:
                raise ValueError("A ticker was mapped to multiple games")
            by_ticker[ticker] = cohort
    clocks = {name: PortfolioClock(list(games.values()), accounts[name])
              for name, games in engines.items()} if accounts else {}
    sequence = BookSequence()
    first = previous = first_wall = None
    gap, ended, complete, rows = None, False, False, 0
    last_good_when = 0.
    with gzip.open(capture, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            stamp = int(row["recorded_monotonic_ns"])
            if previous is not None and stamp < previous:
                raise ValueError("Observation clock moved backwards")
            first = stamp if first is None else first
            if first_wall is None and row.get("recorded_at"):
                first_wall = datetime.fromisoformat(row["recorded_at"]).timestamp() - (stamp - first) / 1e9
            previous = stamp
            when = (stamp - first) / 1e9
            rows += 1
            if row["type"] == "connection_gap" or (ended and row["type"] == "connection_start"):
                gap = "Connection gap; subsequent queue and exposed-order outcomes are indeterminate"
                break
            if ended:
                # MLB requests can finish after the book connection closes.
                # They cannot extend executable book time or fill old orders.
                complete = complete or row["type"] == "capture_end"
                continue
            try:
                for clock in clocks.values():
                    clock.advance_before(when)
                if row["type"] == "connection_end":
                    for cohort in by_game.values():
                        for engine in cohort:
                            engine.advance_clock(when)
                    ended = True
                elif row["type"] == "capture_end":
                    complete = True
                elif row["type"] == "mlb_observation":
                    for engine in by_game.get(row["game_pk"], []):
                        engine.baseball(when, row, first_wall)
                elif row["type"] == "kalshi_message":
                    message = row["message"]
                    if message.get("type") == "error":
                        raise ValueError("Recorded subscription error")
                    sequence.observe(message)
                    body = message.get("msg") or {}
                    if message.get("type") == "trade":
                        if first_wall is None or body.get("ts_ms") is None:
                            raise ValueError("Trade lacks an exchange/receipt timestamp")
                        body["exchange_elapsed_seconds"] = float(body["ts_ms"]) / 1000 - first_wall
                    for engine in by_ticker.get(body.get("market_ticker"), []):
                        engine.observe(when, message)
            except ValueError as error:
                gap = str(error)
                break
            last_good_when = when
            for clock in clocks.values():
                clock.measure()
            if rows % 100000 == 0:
                print(f"Processed {rows:,} records, {when / 60:.1f} captured minutes", flush=True)
    # No future book observations are fabricated for an unfinished prefix.
    # Timers can advance only through its last known connected observation.
    if gap is None and not ended:
        if clocks:
            for clock in clocks.values():
                clock.advance_before(last_good_when, inclusive=True)
        else:
            for cohort in by_game.values():
                for engine in cohort:
                    engine.advance_clock(last_good_when)
    results = {}
    for name, games in engines.items():
        directory = output / name
        directory.mkdir()
        summaries = []
        for pk, engine in games.items():
            summary = engine.summary()
            summary.update(capture_complete=complete, continuity_error=gap,
                replay_valid=gap is None and engine.book is not None,
                observed_seconds=last_good_when,
                account_scope=("Game PnL attribution within a shared cash account" if accounts
                               else "Independent single-game diagnostic; not a shared portfolio"))
            summaries.append(summary)
            (directory / f"{pk}.json").write_text(json.dumps({**summary, "fills": engine.fills,
                                                            "fill_markouts": engine.markouts}, indent=2, allow_nan=False))
        results[name] = {"games": len(games), "games_with_entries": sum(r["entry_orders_filled"] > 0 for r in summaries),
            "games_observed_live": sum(r["game_observed_live"] for r in summaries),
            "games_observed_final": sum(r["game_observed_final"] for r in summaries),
            "entry_orders_filled": sum(r["entry_orders_filled"] for r in summaries),
            "maker_fill_events": sum(r["maker_fill_events"] for r in summaries),
            "taker_fill_events": sum(r["taker_fill_events"] for r in summaries),
            "fees": sum(r["fees"] for r in summaries),
            "sum_independent_game_realized_pnl": sum(r["realized_pnl"] for r in summaries),
            "sum_independent_game_liquidation_marked_pnl": sum(r["liquidation_marked_pnl"] for r in summaries),
            "games_with_open_inventory": sum(abs(r["inventory"]) > 1e-9 for r in summaries),
            "all_replays_valid": all(r["replay_valid"] for r in summaries),
            "capture_complete": complete, "continuity_error": gap}
        if clocks:
            results[name]["portfolio"] = clocks[name].summary()
            for suffix in ("realized_pnl", "liquidation_marked_pnl"):
                results[name][f"sum_game_{suffix}"] = results[name].pop(f"sum_independent_game_{suffix}")
    if digest(capture) != manifest["capture_sha256"] or digest(slate_path) != manifest["slate_sha256"]:
        raise RuntimeError("Capture or slate changed during replay")
    if any(digest(ROOT / name) != value for name, value in hashes.items()):
        raise RuntimeError("Source changed during replay")
    (output / "summary.json").write_text(json.dumps({"role": "development", "deployment_ready": False,
        "capture_complete": complete, "observed_seconds": last_good_when, "continuity_error": gap,
        "candidates": results}, indent=2, allow_nan=False))
    for name, result in results.items():
        marked = (result["portfolio"]["liquidation_marked_pnl"] if clocks
                  else result["sum_independent_game_liquidation_marked_pnl"])
        print(f"{name}: {result['entry_orders_filled']} entry orders, "
              f"{result['games_with_entries']}/{result['games']} games, "
              f"marked sum ${marked:.4f}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["snapshot", "evaluate"])
    parser.add_argument("capture", type=Path)
    parser.add_argument("--slate", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--latency", type=float, choices=[.68, 1.5, 3.], default=.68)
    parser.add_argument("--liquidation-penalty", type=float, choices=[.01, .02], default=.01)
    parser.add_argument("--shared-cash", type=float, help="Use one bankroll across all games")
    parser.add_argument("--policy", choices=list(candidates()), help="Replay one declared policy")
    args = parser.parse_args()
    if args.command == "snapshot":
        path, metadata = snapshot(args.capture, args.output_dir)
        print(json.dumps({"path": str(path), **metadata}, indent=2))
    else:
        if args.slate is None:
            parser.error("evaluate requires --slate")
        evaluate(args.capture, args.slate, args.output_dir, args.latency, args.liquidation_penalty,
                 args.shared_cash, args.policy)


if __name__ == "__main__":
    main()
