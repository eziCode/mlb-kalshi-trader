"""Prospective shadow IOC replay of the frozen market-correction candidate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import heapq
import json
import math
from pathlib import Path

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd

from research.maker_portfolio import CashAccount, PortfolioClock, PortfolioGame
from research.maker_study import StudyConfig
from research.market_correction import logit, logistic
from research.market_check import verify_run, verify_model_directory
from research.passive import fee
from research.record import BookSequence
from research.slate_study import digest, empty_output
from research.state_refresh import ROOT, home_probability, post_features, probability
from settlement_value_strategy.play_eligibility import incomplete_ball_in_play_reason


def observed_state(observation):
    """Only fields present together in a received MLB snapshot are available."""
    score = observation.get("linescore") or {}
    play = observation.get("currentPlay") or {}
    pitches = [p for p in play.get("playEvents", []) if p.get("isPitch")]
    if pitches and incomplete_ball_in_play_reason(play, pitches[-1].get("pitchNumber", 0)):
        return None
    required = ("currentInning", "isTopInning", "outs", "balls", "strikes", "teams", "offense")
    if any(name not in score for name in required):
        return None
    home = not bool(score["isTopInning"])
    offense = score["offense"]
    teams = (observation.get("gameData") or {}).get("teams") or {}
    batting_team = (teams.get("home" if home else "away") or {}).get("id")
    if batting_team is None or (offense.get("team") or {}).get("id") != batting_team:
        return None
    if not (1 <= score["currentInning"] <= 30 and 0 <= score["outs"] <= 3
            and 0 <= score["balls"] <= 3 and 0 <= score["strikes"] <= 2):
        return None
    runs = [(score["teams"].get(side) or {}).get("runs") for side in ("home", "away")]
    if any(value is None for value in runs):
        return None
    return {"inning_after": score["currentInning"], "inning_topbot_after": int(home),
            "outs_when_up_after": score["outs"], "score_diff_after": runs[0] - runs[1],
            "balls_after": score["balls"], "strikes_after": score["strikes"],
            **{f"runner_on_{base}_after": int(bool(offense.get(base))) for base in ("first", "second", "third")}}


class ForwardValueGame(PortfolioGame):
    def __init__(self, game, account, baseball_model, prior, correction, latency=.68, penalty=.01):
        config = StudyConfig(decision_seconds=5., submission_seconds=latency, cancellation_seconds=latency)
        super().__init__(game["market_ticker"], game["away_market_ticker"], game["game_pk"], config, account)
        self.baseball_model, self.prior, self.correction = baseball_model, prior, correction
        self.penalty = penalty
        self.entry_filled = False
        self.current_state, self.current_probability, self.state_seen = None, None, -math.inf
        self.state_identity = None
        self.first_pitch, self.anchor_offset = None, None
        self.anchor_available_at = math.inf
        self.pregame_trades = []
        initial = pd.DataFrame([{ "game_pk": self.game_pk, "inning_after": 1., "inning_topbot_after": 0,
            "outs_when_up_after": 0, "score_diff_after": 0., "balls_after": 0, "strikes_after": 0,
            "runner_on_first_after": 0., "runner_on_second_after": 0., "runner_on_third_after": 0.}])
        self.initial_probability = float(home_probability(baseball_model, post_features(initial, {self.game_pk: prior}))[0])
        self.counters.update(value_orders=0, value_misses=0, value_partial_fills=0, invalid_state_observations=0)

    def baseball(self, now, row, first_wall):
        self.advance_clock(now)
        self.feed_seen = now
        observation = row.get("observation")
        if observation is None:
            self.decide()
            return
        status = ((observation.get("gameData") or {}).get("status") or {}).get("abstractGameState")
        self.game_live, self.game_final = status == "Live", status == "Final"
        if self.game_live and self.first_live_seen is None:
            self.first_live_seen = now
        if self.game_final and self.final_seen is None:
            self.final_seen = now
        plays = list(observation.get("recentPlays") or []) + [observation.get("currentPlay") or {}]
        if self.first_pitch is None and first_wall is not None:
            starts = [p.get("startTime") for play in plays if play.get("atBatIndex") == 0
                      for p in play.get("playEvents", []) if p.get("isPitch") and p.get("startTime")]
            if starts:
                self.first_pitch = min(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() for s in starts) - first_wall
                eligible = [(stamp, price, received) for stamp, price, received in self.pregame_trades
                            if 0 < self.first_pitch - stamp <= 300]
                if eligible:
                    _, price, received = max(eligible, key=lambda item: (item[0], item[2]))
                    self.anchor_offset = float(logit(price) - logit(self.initial_probability))
                    self.anchor_available_at = received + 2.
                self.pregame_trades.clear()
        state = observed_state(observation)
        play = observation.get("currentPlay") or {}
        events = play.get("playEvents") or []
        identity = (play.get("atBatIndex"), tuple((p.get("index"), p.get("endTime")) for p in events))
        if state is None:
            self.current_state, self.current_probability = None, None
            self.counters["invalid_state_observations"] += 1
        elif state != self.current_state or identity != self.state_identity:
            self.current_state, self.state_seen = state, now
            self.state_identity = identity
            features = post_features(pd.DataFrame([{**state, "game_pk": self.game_pk}]), {self.game_pk: self.prior})
            self.current_probability = float(home_probability(self.baseball_model, features)[0])
        self.decide()

    def estimate(self):
        if (self.current_probability is None or self.anchor_offset is None or self.peer_book is None
                or self.now <= self.anchor_available_at or not self.game_live
                or self.now - self.feed_seen > 30 or self.now - self.state_seen > 120):
            return None
        inning = self.current_state["inning_after"]
        if not 1 <= inning <= 7:
            return None
        books = (self.book, self.peer_book)
        if any(b.bid("yes") <= 0 or b.bid("no") <= 0 or not 0 <= b.ask("yes") - b.bid("yes") <= .08 + 1e-9 for b in books):
            return None
        mid, peer_mid = [(b.bid("yes") + b.ask("yes")) / 2 for b in books]
        anchored = logistic(logit(self.current_probability) + self.anchor_offset)
        base = logit(mid)
        features = np.array([base, logit(anchored) - base, base * inning / 9, 1., 1 - mid - peer_mid])
        prediction = float(logistic(base + (features / np.array(self.correction["scales"])) @
                                    np.array(self.correction["coefficients"])))
        return prediction, {"home_probability": prediction, "home_mid": mid, "away_mid": peer_mid,
                            "baseball_probability": self.current_probability, "anchored_probability": float(anchored),
                            "state_age": self.now - self.state_seen, "state": dict(self.current_state)}

    def decide(self):
        if self.book is None or self.now < self.last_decision + self.config.decision_seconds:
            return
        self.last_decision = self.now
        if self.orders or self.entry_filled:
            return
        estimate = self.estimate()
        if estimate is None:
            return
        p, features = estimate
        choices = []
        for side, market, fair in (("yes", self.book, p), ("no", self.peer_book, 1-p)):
            ask = market.ask("yes")
            limit = min(.99, ask + self.penalty)
            edge = fair - limit - fee(1, limit, .07)
            if .05 <= ask <= .95 and edge >= .01 - 1e-9:
                choices.append((edge, side, limit))
        if not choices:
            return
        _, side, limit = max(choices)
        reserve = limit + fee(1, limit, .07) + .01
        if reserve > self.cash + 1e-9:
            self.counters["cash_rejections"] += 1
            return
        self.cash -= reserve
        self.reserved += reserve
        order_id = self.serial + 1
        self.orders[order_id] = dict(side=side, price=limit, remaining=1., reserved=reserve,
            submitted_at=self.now, active=False, cancelling=False, features=features)
        self.schedule(self.now + self.config.submission_seconds, "value_ioc", order_id)
        self.counters["value_orders"] += 1

    def advance(self, now):
        while self.events and self.events[0][0] <= now:
            stamp, _, kind, order_id = heapq.heappop(self.events)
            self.now = stamp
            if kind != "value_ioc":
                raise ValueError("Unexpected event in frozen value policy")
            self.execute(order_id)
        self.now = now

    def execute(self, order_id):
        order = self.orders[order_id]
        market = self.book if order["side"] == "yes" else self.peer_book
        filled = 0.
        for no_price, displayed in sorted(market.levels["no"].items(), reverse=True):
            price = 1 - no_price + self.penalty
            if price > order["price"] + 1e-9 or not 0 < price < 1:
                continue
            quantity = min(displayed, order["remaining"])
            if quantity <= 1e-9:
                continue
            charge = fee(quantity, price, .07)
            cost = quantity * price + charge
            if cost > order["reserved"] + 1e-9:
                raise ValueError("Observed IOC cost exceeds reserved cash")
            previous = abs(self.position)
            self.basis = (previous * self.basis + cost) / (previous + quantity)
            self.position += quantity * (1 if order["side"] == "yes" else -1)
            self.position_since = self.now if self.position_since is None else self.position_since
            self.reserved -= cost
            order["reserved"] -= cost
            order["remaining"] -= quantity
            self.total_fees += charge
            self.peak_inventory = max(self.peak_inventory, abs(self.position))
            filled += quantity
            self.entry_filled = True
            self.fills.append(dict(order_id=order_id, received_seconds=self.now, side=order["side"],
                ticker=self.ticker if order["side"] == "yes" else self.peer_ticker, action="buy",
                liquidity_role="taker", price=price, quantity=quantity, fee=charge, opening_quantity=quantity,
                closing_quantity=0., cancel_pending=False, traded_through=False, position_after=self.position,
                realized_pnl_after=self.realized, submitted_at=order["submitted_at"], features=order["features"]))
            if order["remaining"] <= 1e-9:
                break
        self.counters["value_misses"] += int(filled <= 1e-9)
        self.counters["value_partial_fills"] += int(1e-9 < filled < 1 - 1e-9)
        self.remove(order_id)

    def observe(self, now, message):
        body = message.get("msg") or {}
        if (message.get("type") == "trade" and body.get("market_ticker") == self.ticker
                and self.first_pitch is None and not body.get("is_block_trade")):
            stamp = body.get("exchange_elapsed_seconds")
            if stamp is not None and stamp <= now:
                self.pregame_trades.append((stamp, float(body["yes_price_dollars"]), now))
                self.pregame_trades = [(t, p, r) for t, p, r in self.pregame_trades if now - t <= 600]
        super().observe(now, message)

    def summary(self):
        result = super().summary()
        if self.position < 0:
            value, missing = self.peer_book.liquidation_value("yes", abs(self.position))
            result["inventory_liquidation_mark"], result["inventory_without_bid_depth"] = value, missing
            result["liquidation_marked_pnl"] = self.realized + value - abs(self.position) * self.basis
        result.update(policy="frozen_corrected_01c_observed_book_ioc", first_pitch_observed=self.first_pitch is not None,
                      pregame_anchor_available=self.anchor_offset is not None,
                      status="prospective_book_shadow_not_verified_fills")
        return result


def run(args):
    frozen, selection = verify_run(args.frozen_run)
    verify_model_directory(args.model_dir, frozen)
    if selection["candidate"] != "corrected_01c":
        raise ValueError("This forward protocol only supports the declared frozen candidate")
    output = args.output_dir
    empty_output(output)
    paths = [args.capture, args.slate, args.model_dir / "model.cbm", args.model_dir / "prior.json",
             args.frozen_run / "correction.json", args.frozen_run / "selection.json", Path(__file__),
             ROOT / "research/FORWARD_VALUE_PROTOCOL.md", ROOT / "research/maker_portfolio.py",
             ROOT / "research/maker_study.py", ROOT / "research/passive.py", ROOT / "research/record.py",
             ROOT / "research/market_check.py", ROOT / "research/market_correction.py", ROOT / "research/state_refresh.py",
             ROOT / "settlement_value_strategy/play_eligibility.py"]
    hashes = {str(path.resolve()): digest(path) for path in paths}
    (output / "manifest.json").write_text(json.dumps({"started_at": datetime.now(timezone.utc).isoformat(),
        "hashes": hashes, "candidate": selection["candidate"], "latency": args.latency,
        "penalty": args.penalty, "role": "prospective_frozen_policy_shadow", "deployment_ready": False}, indent=2))
    model = CatBoostClassifier().load_model(str(args.model_dir / "model.cbm"))
    correction = json.loads((args.frozen_run / "correction.json").read_text())
    ratings = json.loads((args.model_dir / "prior.json").read_text())["ratings"]
    account = CashAccount.funded(100.)
    aliases = {"CHW": "CWS", "ARI": "AZ"}
    games, tickers = {}, {}
    for game in json.loads(args.slate.read_text())["games"]:
        home = game["home_code"] if game["home_code"] in ratings else aliases.get(game["home_code"], game["home_code"])
        away = game["away_code"] if game["away_code"] in ratings else aliases.get(game["away_code"], game["away_code"])
        engine = ForwardValueGame(game, account, model, probability(home, away, ratings), correction, args.latency, args.penalty)
        games[game["game_pk"]] = engine
        for ticker in (engine.ticker, engine.peer_ticker):
            if ticker in tickers:
                raise ValueError("Ambiguous ticker mapping")
            tickers[ticker] = engine
    clock, sequence = PortfolioClock(list(games.values()), account), BookSequence()
    first = previous = first_wall = None
    error, complete, ended, last = None, False, False, 0.
    with gzip.open(args.capture, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            stamp = row["recorded_monotonic_ns"]
            if previous is not None and stamp < previous:
                raise ValueError("Observation clock moved backwards")
            first = stamp if first is None else first
            first_wall = datetime.fromisoformat(row["recorded_at"]).timestamp() if first_wall is None else first_wall
            previous = stamp
            now = (stamp - first) / 1e9
            if row["type"] == "connection_gap" or (ended and row["type"] == "connection_start"):
                error = "Connection gap; subsequent execution is indeterminate"
                break
            if ended:
                complete = complete or row["type"] == "capture_end"
                continue
            try:
                clock.advance_before(now)
                if row["type"] == "kalshi_message":
                    message = row["message"]
                    if message.get("type") == "error":
                        raise ValueError("Subscription error")
                    sequence.observe(message)
                    body = message.get("msg") or {}
                    if message.get("type") == "trade":
                        if body.get("ts_ms") is None:
                            raise ValueError("Missing exchange trade timestamp")
                        body["exchange_elapsed_seconds"] = body["ts_ms"] / 1000 - first_wall
                    if body.get("market_ticker") in tickers:
                        tickers[body["market_ticker"]].observe(now, message)
                elif row["type"] == "mlb_observation" and row["game_pk"] in games:
                    games[row["game_pk"]].baseball(now, row, first_wall)
                elif row["type"] == "capture_end":
                    complete = True
                elif row["type"] == "connection_end":
                    ended = True
            except ValueError as exc:
                error = str(exc)
                break
            clock.measure()
            last = now
    if error is None and not ended:
        clock.advance_before(last, inclusive=True)
    summaries = []
    for pk, engine in games.items():
        result = engine.summary()
        result.update(replay_valid=error is None and engine.book is not None and engine.peer_book is not None,
                      capture_complete=complete, continuity_error=error)
        summaries.append(result)
        (output / f"{pk}.json").write_text(json.dumps({**result, "fills": engine.fills}, indent=2, allow_nan=False))
    summary = {"portfolio": clock.summary(), "games": len(games),
        "games_observed_live": sum(r["game_observed_live"] for r in summaries),
        "games_observed_final": sum(r["game_observed_final"] for r in summaries),
        "games_with_entries": sum(r["entry_orders_filled"] > 0 for r in summaries),
        "entry_orders_filled": sum(r["entry_orders_filled"] for r in summaries),
        "entry_contracts": sum(r["entry_contracts"] for r in summaries),
        "fees": sum(r["fees"] for r in summaries), "continuity_error": error,
        "all_replays_valid": all(r["replay_valid"] for r in summaries), "capture_complete": complete,
        "observed_seconds": last, "deployment_ready": False}
    if any(digest(Path(path)) != expected for path, expected in hashes.items()):
        raise RuntimeError("Frozen forward inputs changed during replay")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--slate", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--frozen-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--latency", type=float, choices=[.68, 1.5, 3.], default=.68)
    parser.add_argument("--penalty", type=float, choices=[.01, .02], default=.01)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
