"""Reconcile held shadow positions against prospectively observed settlements."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from research.slate_study import digest, empty_output


def observed_settlements(path):
    result, previous, observation_errors = {}, None, 0
    with gzip.open(path, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            now = int(row["recorded_monotonic_ns"])
            if previous is not None and now < previous:
                raise ValueError("Settlement journal clock moved backwards")
            previous = now
            if row["type"] == "market_settlement_error":
                observation_errors += 1
                continue
            if row["type"] != "market_settlement_observation" or row.get("unchanged"):
                continue
            market = row["market"]
            if market.get("status") != "finalized":
                continue
            if market.get("settlement_ts") is None or market.get("settlement_value_dollars") is None:
                observation_errors += 1
                continue
            value = float(market["settlement_value_dollars"])
            exchange_time = datetime.fromisoformat(market["settlement_ts"].replace("Z", "+00:00")).timestamp()
            known_at = datetime.fromisoformat(row["recorded_at"]).timestamp()
            if not 0 <= value <= 1 or exchange_time > known_at:
                raise ValueError("Invalid finalized settlement observation")
            ticker = market["ticker"]
            terminal = {"value": value, "exchange_time": exchange_time,
                        "observed_monotonic_ns": now, "observed_at": row["recorded_at"]}
            if ticker in result and any(result[ticker][k] != terminal[k] for k in ("value", "exchange_time")):
                raise ValueError("Conflicting finalized settlement observations")
            result.setdefault(ticker, terminal)
    return result, observation_errors


def reconcile(game, terminals, first_wall, first_monotonic):
    position = abs(game["inventory"])
    result = {"game_pk": game["game_pk"], "quantity": position,
              "outstanding_orders": game["outstanding_orders"], "settled": False,
              "pnl": game["liquidation_marked_pnl"], "mark_source": "last_visible_bid_less_exit_fees"}
    if position < 1e-9:
        result["mark_source"] = "no_inventory"
        return result
    fills = game["fills"]
    if any(f["action"] != "buy" or abs(f["quantity"] - f["opening_quantity"]) > 1e-9 for f in fills):
        raise ValueError("This reconciliation requires held opening positions")
    if abs(sum(f["quantity"] for f in fills) - position) > 1e-7:
        raise ValueError("Fill quantities do not reconcile with held inventory")
    cost = sum(f["quantity"] * f["price"] + f["fee"] for f in fills)
    if abs(cost - position * game["inventory_cost_per_contract"]) > 1e-7:
        raise ValueError("Fill costs do not reconcile with inventory basis")
    tickers = {f["ticker"] for f in fills}
    if len(tickers) != 1:
        raise ValueError("Position is not attributable to one contract market")
    ticker = next(iter(tickers))
    result["ticker"] = ticker
    terminal = terminals.get(ticker)
    if terminal is None:
        return result
    latest_fill = max(f["received_seconds"] for f in fills)
    if first_wall + latest_fill >= terminal["exchange_time"]:
        raise ValueError("Shadow entry occurred after exchange settlement")
    if first_monotonic + round(latest_fill * 1e9) >= terminal["observed_monotonic_ns"]:
        raise ValueError("Settlement observation precedes the acquired inventory")
    proceeds = position * terminal["value"]
    result.update(settled=True, settlement_value=terminal["value"], settlement_proceeds=proceeds,
                  settlement_observed_at=terminal["observed_at"], mark_source="observed_exchange_settlement",
                  pnl=game["realized_pnl"] + proceeds - position * game["inventory_cost_per_contract"])
    return result


def run(args):
    empty_output(args.output_dir)
    replay_manifest = json.loads((args.replay_dir / "manifest.json").read_text())
    expected = replay_manifest["hashes"].get(str(args.capture.resolve()))
    if expected is None or digest(args.capture) != expected:
        raise ValueError("Capture does not match the frozen forward replay")
    game_files = sorted(p for p in args.replay_dir.glob("*.json") if p.stem.isdigit())
    files = [args.capture, args.settlements, args.replay_dir / "summary.json",
             args.replay_dir / "manifest.json", Path(__file__), *game_files]
    hashes = {str(p.resolve()): digest(p) for p in files}
    (args.output_dir / "manifest.json").write_text(json.dumps({"started_at": datetime.now(timezone.utc).isoformat(),
        "hashes": hashes, "role": "shadow_settlement_accounting", "actual_exchange_orders": False}, indent=2))
    with gzip.open(args.capture, "rt") as stream:
        first = json.loads(next(stream))
    first_wall = datetime.fromisoformat(first["recorded_at"]).timestamp()
    terminals, errors = observed_settlements(args.settlements)
    games = [json.loads(path.read_text()) for path in game_files]
    positions = [reconcile(g, terminals, first_wall, first["recorded_monotonic_ns"]) for g in games]
    original = json.loads((args.replay_dir / "summary.json").read_text())
    if len(games) != original["games"] or len({g["game_pk"] for g in games}) != len(games):
        raise ValueError("Per-game reports do not cover the original replay")
    pnl = sum(p["pnl"] for p in positions)
    all_settled = all(p["quantity"] < 1e-9 or p["settled"] for p in positions)
    pending = sum(p["outstanding_orders"] for p in positions)
    result = {"shadow_pnl_including_unsettled_marks": pnl,
        "settled_shadow_pnl": sum(p["pnl"] for p in positions if p["settled"]),
        "starting_cash": original["portfolio"]["starting_cash"],
        "shadow_equity": original["portfolio"]["starting_cash"] + pnl,
        "games": len(games), "games_with_entries": original["games_with_entries"],
        "scheduled_game_count": original.get("scheduled_game_count", len(games)),
        "unmapped_game_pks": original.get("unmapped_game_pks", []),
        "settled_positions": sum(p["settled"] for p in positions),
        "unsettled_positions": sum(p["quantity"] > 1e-9 and not p["settled"] for p in positions),
        "outstanding_orders": pending, "settlement_observation_errors": errors,
        "all_existing_positions_settled": all_settled,
        "complete_policy_evaluation": bool(all_settled and pending == 0 and original["capture_complete"]
            and original["all_replays_valid"] and original["games_observed_final"] == len(games)),
        "evaluation_scope": "Mapped games; unmapped scheduled games remain outside scored results",
        "cash_recycled_during_replay": False, "actual_exchange_orders": False,
        "deployment_ready": False, "positions": positions}
    if any(digest(Path(p)) != expected for p, expected in hashes.items()):
        raise RuntimeError("Settlement reconciliation inputs changed")
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in result.items() if k != "positions"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--settlements", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
