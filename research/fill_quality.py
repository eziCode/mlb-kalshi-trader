"""Describe passive fill quality without turning future marks into signals."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random

from research.slate_study import digest, empty_output


HORIZONS = (5, 30, 60)


def clustered_interval(values, minimum_games=10, repetitions=1000):
    """Descriptive game-cluster bootstrap of a quantity-weighted mean.

    This does not correct for candidate selection or execution-model error.
    A few games with many fills are still a few independent game outcomes.
    """
    if len(values) < minimum_games:
        return None
    pairs = list(values.values())
    rng = random.Random(270926)
    estimates = []
    for _ in range(repetitions):
        sample = rng.choices(pairs, k=len(pairs))
        weight = sum(q for _, q in sample)
        if weight:
            estimates.append(sum(p for p, _ in sample) / weight)
    estimates.sort()
    if not estimates:
        return None
    return [estimates[int(.025 * (len(estimates) - 1))], estimates[int(.975 * (len(estimates) - 1))]]


def summarize_rows(rows):
    quantity = sum(row["quantity"] for row in rows)
    result = {"games": len({r["game_pk"] for r in rows}),
              "entry_orders": len({(r["game_pk"], r["order_id"]) for r in rows}),
              "opening_fill_events": len(rows), "opening_contracts": quantity,
              "entry_fees": sum(r["entry_fee"] for r in rows), "horizons": {}}
    for horizon in HORIZONS:
        eligible = [r for r in rows if horizon in r["marks"] and r["marks"][horizon].get("valid_book")]
        q = sum(r["quantity"] for r in eligible)
        metric = {"observations": len(eligible), "missing_or_invalid_observations": len(rows) - len(eligible),
                  "opening_contracts": q, "games": len({r["game_pk"] for r in eligible})}
        for name in ("mid_change_per_contract", "liquidation_pnl_per_contract", "post_fill_mid_change_per_contract"):
            usable = [r for r in eligible if r["marks"][horizon].get(name) is not None]
            denominator = sum(r["quantity"] for r in usable)
            totals = defaultdict(lambda: [0., 0.])
            for row in usable:
                value = row["marks"][horizon][name]
                if not math.isfinite(value):
                    raise ValueError("Nonfinite fill diagnostic")
                totals[row["game_pk"]][0] += value * row["quantity"]
                totals[row["game_pk"]][1] += row["quantity"]
            metric[name] = {"weighted_mean": sum(p for p, _ in totals.values()) / denominator if denominator else None,
                            "game_cluster_interval_95": clustered_interval(totals),
                            "games": len(totals), "opening_contracts": denominator}
        metric["depth_checked_observations"] = sum(bool(r["marks"][horizon].get("depth_checked")) for r in eligible)
        metric["top_bid_proxy_observations"] = len(eligible) - metric["depth_checked_observations"]
        result["horizons"][str(horizon)] = metric
    return result


def opening_rows(game):
    marks = defaultdict(dict)
    for mark in game.get("fill_markouts", []):
        key = mark["fill_index"], mark["horizon_seconds"]
        if key[1] in marks[key[0]]:
            raise ValueError("Duplicate fill/horizon observation")
        marks[key[0]][key[1]] = mark
    rows = []
    for index, fill in enumerate(game.get("fills", [])):
        quantity = fill.get("opening_quantity", fill["quantity"] - fill.get("closing_quantity", 0))
        if quantity <= 1e-9 or fill.get("liquidity_role") != "maker":
            continue
        if not 0 < quantity <= fill["quantity"] or not math.isfinite(fill["fee"]):
            raise ValueError("Invalid opening quantity or fee")
        features = fill.get("features") or {}
        spread = features.get("spread")
        regime = features.get("regime") or ("inning_break" if features.get("break_age") is not None else "active_play")
        mechanism = "cancel_race" if fill.get("cancel_pending") else ("trade_through" if fill.get("traded_through") else "queue")
        rows.append({"game_pk": game["game_pk"], "fill_index": index, "order_id": fill["order_id"],
                     "quantity": quantity, "entry_fee": fill["fee"] * quantity / fill["quantity"],
                     "regime": regime, "mechanism": mechanism,
                     "spread_bucket": ("unknown" if spread is None else "under_3c" if spread < .03 - 1e-9
                                       else "3_to_5c" if spread <= .05 + 1e-9 else "over_5c"),
                     "marks": marks[index]})
    return rows


def analyze_games(games):
    if len({g["game_pk"] for g in games}) != len(games):
        raise ValueError("Repeated game reports would double-count overlapping capture prefixes")
    rows = [r for game in games for r in opening_rows(game)]
    groups = {}
    for dimension in ("regime", "mechanism", "spread_bucket"):
        values = sorted({r[dimension] for r in rows})
        groups[dimension] = {value: summarize_rows([r for r in rows if r[dimension] == value]) for value in values}
    return {"all_openings": summarize_rows(rows), "groups": groups,
            "mapped_games": len(games), "zero_entry_games": sum(g["entry_orders_filled"] == 0 for g in games),
            "all_replays_valid": all(g.get("replay_valid", False) for g in games),
            "interpretation": "Future marks describe fills; they are not executable exits or training features",
            "uncertainty": "Game-cluster descriptive intervals require ten games; no selection or fill-model correction"}


def load_candidates(root):
    summary = json.loads((root / "summary.json").read_text())
    names = list(summary["candidates"]) if "candidates" in summary else ["frozen"]
    return summary, {name: [json.loads(p.read_text()) for p in sorted((root / name if name != "frozen" else root).glob("*.json"))
                            if p.stem.isdigit()] for name in names}


def write_report(root, output):
    empty_output(output)
    summary, candidates = load_candidates(root)
    files = list(root.rglob("*.json"))
    hashes = {str(p.resolve()): digest(p) for p in files}
    analyses = {name: analyze_games(games) for name, games in candidates.items()}
    (output / "summary.json").write_text(json.dumps({"created_at": datetime.now(timezone.utc).isoformat(),
        "source_hashes": hashes, "deployment_ready": False, "candidates": analyses}, indent=2, allow_nan=False))
    fields = ["candidate", "game_pk", "entry_orders", "opening_contracts", "fees", "realized_shadow_pnl",
              "marked_shadow_pnl", "inventory", "outstanding_orders", "replay_valid"]
    with (output / "per_game.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, games in candidates.items():
            for g in games:
                writer.writerow(dict(candidate=name, game_pk=g["game_pk"], entry_orders=g["entry_orders_filled"],
                    opening_contracts=g["entry_contracts"], fees=g["fees"], realized_shadow_pnl=g["realized_pnl"],
                    marked_shadow_pnl=g["liquidation_marked_pnl"], inventory=g["inventory"],
                    outstanding_orders=g["outstanding_orders"], replay_valid=g["replay_valid"]))
    lines = ["# Passive fill diagnostics", "", "Shadow observations only. Missing horizons remain missing.", "",
             "| Candidate | Games entered / mapped | Entry orders | Opening contracts | 30s midpoint less entry |",
             "|---|---:|---:|---:|---:|"]
    for name, report in analyses.items():
        all_rows = report["all_openings"]
        mean = all_rows["horizons"]["30"]["mid_change_per_contract"]["weighted_mean"]
        mean_text = "missing" if mean is None else f"{100 * mean:+.3f} cents"
        lines.append(f"| {name} | {report['mapped_games']-report['zero_entry_games']} / {report['mapped_games']} | "
                     f"{all_rows['entry_orders']} | {all_rows['opening_contracts']:.2f} | {mean_text} |")
    lines += ["", "A future midpoint is not a fill. Older replay files use a top-bid exit proxy;",
              "the JSON explicitly distinguishes those marks from depth-checked marks.",
              "Game-cluster intervals are descriptive and do not account for strategy selection.",
              "A prefix with no entries provides no evidence about trading profitability."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    if any(digest(Path(p)) != expected for p, expected in hashes.items()):
        raise RuntimeError("Diagnostic inputs changed during analysis")
    return analyses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = write_report(args.replay_dir, args.output_dir)
    print(json.dumps({name: r["all_openings"]["entry_orders"] for name, r in result.items()}, indent=2))


if __name__ == "__main__":
    main()
