"""Apply idempotent, unit-preserving feature preprocessing."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlb_kalshi.strategy import validate_market_prices  # noqa: E402


TRAIN_PATH = PROJECT_ROOT / "data/processed/train/training_dataset.parquet"
TEST_PATH = PROJECT_ROOT / "data/processed/test/test_dataset.parquet"


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Keep raw units so historical, replay, and live features are identical."""
    result = df.copy()
    if "inning_topbot" in result:
        encoded = result["inning_topbot"].replace({
            "Top": 0, "top": 0, "Bot": 1, "bot": 1,
        })
        result["inning_topbot"] = pd.to_numeric(encoded, errors="coerce")
    if "runner_state" in result:
        state = result["runner_state"].fillna("000").astype(str).str.zfill(3)
        result["runner_on_first"] = pd.to_numeric(state.str[0], errors="coerce")
        result["runner_on_second"] = pd.to_numeric(state.str[1], errors="coerce")
        result["runner_on_third"] = pd.to_numeric(state.str[2], errors="coerce")
        result = result.drop(columns="runner_state")

    result["yes_bid_close"] = pd.to_numeric(
        result["yes_bid_close"], errors="coerce"
    )
    result["yes_ask_close"] = pd.to_numeric(
        result["yes_ask_close"], errors="coerce"
    )
    result["kalshi_price"] = (
        result["yes_bid_close"] + result["yes_ask_close"]
    ) / 2.0
    result["spread"] = result["yes_ask_close"] - result["yes_bid_close"]
    validate_market_prices(result)
    return result


def main() -> None:
    train = preprocess(pd.read_parquet(TRAIN_PATH))
    test = preprocess(pd.read_parquet(TEST_PATH))
    train.to_parquet(TRAIN_PATH, index=False)
    test.to_parquet(TEST_PATH, index=False)
    print(f"Saved raw-unit train rows: {len(train):,}")
    print(f"Saved raw-unit test rows:  {len(test):,}")


if __name__ == "__main__":
    main()
