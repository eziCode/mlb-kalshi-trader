"""Paper-only forward scorer for prepared decision rows supplied as JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from settlement_value_strategy.predict import MispricingPredictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        help="JSONL decision stream; omit to read newline-delimited JSON on stdin",
    )
    parser.add_argument(
        "--log", type=Path,
        default=Path(__file__).resolve().parent / "results/paper_decisions.jsonl",
    )
    args = parser.parse_args()
    predictor = MispricingPredictor()
    source = args.input.open() if args.input else sys.stdin
    args.log.parent.mkdir(parents=True, exist_ok=True)
    seen_games: set[int] = set()
    try:
        with args.log.open("a") as output:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                game_pk = int(row["game_pk"])
                decision = predictor.decision(row)
                decision.update({
                    "game_pk": game_pk,
                    "signal_time": row.get("signal_time"),
                    "market_home_price": row["market_home_price"],
                    "paper_order": bool(decision["eligible"] and game_pk not in seen_games),
                })
                if decision["paper_order"]:
                    seen_games.add(game_pk)
                encoded = json.dumps(decision, sort_keys=True)
                output.write(encoded + "\n")
                output.flush()
                print(encoded, flush=True)
    finally:
        if args.input:
            source.close()


if __name__ == "__main__":
    main()

