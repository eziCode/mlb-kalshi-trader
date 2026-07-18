"""Build one causal empirical-market-reaction row per completed 2026 hit."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlb_kalshi.empirical_reaction import logit  # noqa: E402


DATA_DIR = PROJECT_ROOT / "data/processed/trade_tape"
OUTPUT_PATH = DATA_DIR / "empirical_reaction_events.parquet"
REACTION_SECONDS = 5.0
MAXIMUM_PRE_TRADE_AGE_SECONDS = 5.0


def window_stats(
    prices: np.ndarray,
    sizes: np.ndarray,
    taker_sides: np.ndarray,
    start: int,
    stop: int,
    batting_home: bool,
) -> dict:
    selected_prices = prices[start:stop]
    selected_sizes = sizes[start:stop]
    selected_sides = taker_sides[start:stop]
    total = float(selected_sizes.sum())
    signed = np.where(selected_sides == "yes", selected_sizes, -selected_sizes)
    if not batting_home:
        signed = -signed
    return {
        "trade_count": int(stop - start),
        "volume": total,
        "flow_imbalance": float(signed.sum() / total) if total else 0.0,
        "volatility": float(np.std(selected_prices)) if len(selected_prices) > 1 else 0.0,
    }


def main() -> None:
    trades = pd.read_parquet(DATA_DIR / "home_market_trades.parquet")
    updates = pd.read_parquet(DATA_DIR / "state_updates.parquet")
    trades["created_time"] = pd.to_datetime(trades["created_time"], utc=True)
    updates["pitch_start_time"] = pd.to_datetime(
        updates["pitch_start_time"], utc=True
    )
    updates["pitch_end_time"] = pd.to_datetime(
        updates["pitch_end_time"], utc=True
    )
    trades_by_game = {
        int(game_pk): game.sort_values(["created_time", "trade_id"])
        for game_pk, game in trades.groupby("game_pk", sort=False)
    }
    rows = []
    for game_pk, game_updates in updates.groupby("game_pk", sort=False):
        game_pk = int(game_pk)
        game_trades = trades_by_game.get(game_pk)
        if game_trades is None or game_trades.empty:
            continue
        game_updates = game_updates.sort_values("pitch_end_time").reset_index(drop=True)
        times = game_trades["created_time"].array.as_unit("ns").asi8
        prices = game_trades["yes_price_dollars"].to_numpy(dtype=float)
        sizes = game_trades["count_fp"].to_numpy(dtype=float)
        taker_sides = game_trades["taker_outcome_side"].astype(str).to_numpy()
        home_win = int(game_trades["home_win"].iloc[-1])
        game_end = game_updates["pitch_end_time"].max()

        for update_index, hit in game_updates[game_updates["is_hit"]].iterrows():
            start_ns = pd.Timestamp(hit.pitch_start_time).value
            end_ns = pd.Timestamp(hit.pitch_end_time).value
            pre_index = int(np.searchsorted(times, start_ns, side="left") - 1)
            if pre_index < 0:
                continue
            pre_age = (start_ns - times[pre_index]) / 1e9
            if pre_age > MAXIMUM_PRE_TRADE_AGE_SECONDS:
                continue
            post_stop_ns = end_ns + int(REACTION_SECONDS * 1e9)
            post_start = int(np.searchsorted(times, end_ns, side="left"))
            post_stop = int(np.searchsorted(times, post_stop_ns, side="right"))
            if post_start >= post_stop:
                continue
            pre_start = int(np.searchsorted(
                times, start_ns - int(5e9), side="left"
            ))
            pre_stop = pre_index + 1
            batting_home = bool(hit.completed_event_batting_home)
            pre_home = float(prices[pre_index])
            post_weights = sizes[post_start:post_stop]
            post_home = float(np.average(
                prices[post_start:post_stop], weights=post_weights
            ))
            pre_batting = pre_home if batting_home else 1.0 - pre_home
            post_batting = post_home if batting_home else 1.0 - post_home
            pre_stats = window_stats(
                prices, sizes, taker_sides, pre_start, pre_stop, batting_home
            )
            post_stats = window_stats(
                prices, sizes, taker_sides, post_start, post_stop, batting_home
            )
            next_updates = game_updates[
                game_updates["pitch_end_time"] > hit.pitch_end_time
            ]
            valid_until = (
                next_updates.iloc[0]["pitch_end_time"]
                if not next_updates.empty else game_end
            )
            rows.append({
                "event_id": len(rows),
                "game_pk": game_pk,
                "game_date": hit.game_date,
                "market_ticker": hit.market_ticker,
                "home_win": home_win,
                "at_bat_number": int(hit.at_bat_number),
                "pitch_number": int(hit.pitch_number),
                "event_type": str(hit.completed_event),
                "batting_home": int(batting_home),
                "pitch_start_time": hit.pitch_start_time,
                "event_end_time": hit.pitch_end_time,
                "decision_time": pd.Timestamp(hit.pitch_end_time)
                + pd.Timedelta(seconds=REACTION_SECONDS),
                "valid_until": valid_until,
                "game_end_time": game_end,
                "pre_trade_time": game_trades.iloc[pre_index]["created_time"],
                "pre_trade_age_seconds": pre_age,
                "pre_home_price": pre_home,
                "post_home_price": post_home,
                "pre_batting_price": pre_batting,
                "post_batting_price": post_batting,
                "actual_batting_logit_move": float(
                    logit(post_batting) - logit(pre_batting)
                ),
                "inning": int(hit.inning_after),
                "batting_score_diff": float(
                    hit.score_diff_after if batting_home
                    else -hit.score_diff_after
                ),
                "outs_after": int(hit.outs_when_up_after),
                "runner_on_first_after": int(hit.runner_on_first_after),
                "runner_on_second_after": int(hit.runner_on_second_after),
                "runner_on_third_after": int(hit.runner_on_third_after),
                "pitch_duration_seconds": (
                    hit.pitch_end_time - hit.pitch_start_time
                ).total_seconds(),
                "pre_trade_count_5s": pre_stats["trade_count"],
                "pre_volume_5s": pre_stats["volume"],
                "pre_flow_imbalance_5s": pre_stats["flow_imbalance"],
                "pre_volatility_5s": pre_stats["volatility"],
                "post_trade_count_5s": post_stats["trade_count"],
                "post_volume_5s": post_stats["volume"],
                "post_flow_imbalance_5s": post_stats["flow_imbalance"],
                "post_volatility_5s": post_stats["volatility"],
            })
    frame = pd.DataFrame(rows).sort_values(
        ["game_date", "event_end_time", "game_pk"]
    ).reset_index(drop=True)
    frame["event_id"] = np.arange(len(frame), dtype=int)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUTPUT_PATH, index=False)
    print(
        f"Saved {len(frame):,} empirical hit reactions across "
        f"{frame['game_pk'].nunique():,} games to {OUTPUT_PATH}"
    )
    print(
        f"Dates: {frame['game_date'].min()} through {frame['game_date'].max()}"
    )


if __name__ == "__main__":
    main()
