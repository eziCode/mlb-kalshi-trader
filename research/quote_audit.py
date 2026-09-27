"""Describe paired quote observations in the old runtime's decision logs.

This counts sampled price dislocations, not fills or arbitrage profits.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def audit(directory):
    frames, hashes = [], {}
    columns = ["decision_time", "home_ask", "away_ask", "home_ask_size", "away_ask_size"]
    for path in sorted(directory.glob("settlement_value_decisions_v2_*.csv")):
        frame = pd.read_csv(path, usecols=columns)
        if frame.empty:
            continue
        frame["market"] = path.stem.removeprefix("settlement_value_decisions_v2_")
        frames.append(frame)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not frames:
        raise ValueError("No paired quote logs found")
    frame = pd.concat(frames, ignore_index=True)
    valid = (frame.home_ask.between(.0001, .9999) & frame.away_ask.between(.0001, .9999)
             & (frame.home_ask_size >= 1) & (frame.away_ask_size >= 1))
    ask_sum = frame.home_ask + frame.away_ask
    # Round each one-contract leg's fee upwards to a centicent.
    import numpy as np
    fees = sum(np.ceil(.07 * frame[c] * (1 - frame[c]) * 10000 - 1e-9) / 10000
               for c in ("home_ask", "away_ask"))
    apparent = valid & (ask_sum + fees < 1)
    times = pd.to_datetime(frame.decision_time, utc=True, format="mixed")
    return {"quote_rows": len(frame), "markets": frame.market.nunique(),
        "start": times.min().isoformat(), "end": times.max().isoformat(),
        "apparent_dislocation_rows": int(apparent.sum()),
        "markets_with_apparent_dislocations": frame.loc[apparent, "market"].nunique(),
        "raw_ask_sum_quantiles": {str(k): v for k, v in ask_sum.quantile([0, .001, .01, .5, .99, 1]).items()},
        "source_hashes": hashes, "deployment_ready": False,
        "limitations": ["Old-policy decision samples, not continuous books",
            "Quote age, synchronized availability, leg execution, and contract-equivalence rules are not established",
            "These observations cannot estimate full-market opportunity frequency or realized profit"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=Path("live_logs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path")
    result = audit(args.log_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in result.items() if k != "source_hashes"}, indent=2))


if __name__ == "__main__":
    main()
