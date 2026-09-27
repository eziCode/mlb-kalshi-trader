"""Separate, fixed selective maker policies using only received observations."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, replace
from datetime import datetime
import math

from research.maker_portfolio import PortfolioGame
from research.maker_study import StudyConfig
from research.passive import Book, fee


@dataclass(frozen=True)
class SelectiveConfig(StudyConfig):
    lifetime_seconds: float = 10.
    minimum_spread: float = .02
    improve_when_spread: float = 2.  # Join; no assumed queue jump.
    minimum_estimated_edge: float = .005  # After two modeled maker fees.
    maximum_book_age: float = 30.
    movement_window: float = 10.
    maximum_mid_range: float = .03
    maximum_peer_disagreement: float = .03
    maximum_queue: float = 50.
    reentry_cooldown: float = 5.
    maximum_entry_orders: int = 20
    maximum_game_loss: float = 3.
    reference_mode: str = "paired_books"

    def __post_init__(self):
        super().__post_init__()
        for name in ("maximum_book_age", "movement_window", "maximum_mid_range", "maximum_peer_disagreement",
                     "maximum_queue", "reentry_cooldown", "maximum_entry_orders", "maximum_game_loss"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"Invalid {name}")
        if self.reference_mode not in {"paired_books", "sportsbook_anchor"}:
            raise ValueError("Unknown selective reference")
        if self.contracts != 1 or int(self.maximum_entry_orders) != self.maximum_entry_orders:
            raise ValueError("Protocol requires one-contract quotes and an integer entry cap")


def candidates(include_sportsbook=True):
    result = {"selective_live": SelectiveConfig(), "selective_break": SelectiveConfig(entry_regime="break")}
    if include_sportsbook:
        result["sportsbook_anchor"] = SelectiveConfig(reference_mode="sportsbook_anchor")
    return result


class AnchoredForecast:
    """Frozen baseball model shifted by a consensus known before first pitch."""
    def __init__(self, game_pk, model, prior, tape):
        import pandas as pd
        from research.state_refresh import home_probability, post_features
        self.game_pk, self.model, self.prior, self.tape = game_pk, model, prior, tape
        initial = {"game_pk": game_pk, "inning_after": 1., "inning_topbot_after": 0,
                   "outs_when_up_after": 0, "score_diff_after": 0., "balls_after": 0, "strikes_after": 0,
                   "runner_on_first_after": 0., "runner_on_second_after": 0., "runner_on_third_after": 0.}
        self.initial = float(home_probability(model, post_features(pd.DataFrame([initial]), {game_pk: prior}))[0])
        self.first_pitch, self.anchor, self.current = None, None, None
        self.state, self.identity, self.state_seen = None, None, -math.inf
        self.reason = "first_pitch_not_observed"

    def observe(self, now, observation, first_wall):
        import pandas as pd
        from research.forward_value import observed_state
        from research.state_refresh import home_probability, post_features
        plays = list(observation.get("recentPlays") or []) + [observation.get("currentPlay") or {}]
        if self.first_pitch is None and first_wall is not None:
            starts = [p["startTime"] for play in plays if play.get("atBatIndex") == 0
                      for p in play.get("playEvents", []) if p.get("isPitch") and p.get("startTime")]
            if starts:
                stamp = min(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() for s in starts)
                if stamp > first_wall + now:
                    raise ValueError("First pitch source clock is in the future")
                self.first_pitch = stamp
                reference = (self.tape.reference(self.game_pk, stamp, strictly_before=True) if self.tape
                             else {"valid": False, "reason": "sportsbook_feed_missing"})
                self.anchor = reference if reference["valid"] else None
                self.reason = None if self.anchor else reference["reason"]
        state = observed_state(observation)
        play = observation.get("currentPlay") or {}
        identity = (play.get("atBatIndex"), tuple((p.get("index"), p.get("endTime")) for p in play.get("playEvents", [])))
        if state is None:
            self.current, self.state = None, None
        elif state != self.state or identity != self.identity:
            self.state, self.identity, self.state_seen = state, identity, now
            features = post_features(pd.DataFrame([{**state, "game_pk": self.game_pk}]), {self.game_pk: self.prior})
            self.current = float(home_probability(self.model, features)[0])

    def probability(self, now):
        from research.market_correction import logit, logistic
        if self.anchor is None or self.current is None or now - self.state_seen > 120:
            return None
        if not 1 <= self.state["inning_after"] <= 7:
            return None
        return float(logistic(logit(self.current) + logit(self.anchor["home_probability"]) - logit(self.initial)))


class SelectiveGame(PortfolioGame):
    def __init__(self, game, config, account, forecast=None):
        super().__init__(game["market_ticker"], game["away_market_ticker"], game["game_pk"], config, account)
        self.forecast = forecast
        self.book_seen, self.peer_seen = -math.inf, -math.inf
        self.history = deque()
        self.inning, self.phase, self.state_valid = None, None, False
        self.last_flattened = -math.inf
        self.opening_order_ids = set()
        self.attempts, self.rejections = {}, Counter()

    def baseball(self, now, row, first_wall):
        self.advance_clock(now)  # No timer may see the newly arriving state early.
        observation = row.get("observation")
        if observation is not None:
            from research.forward_value import observed_state
            score = observation.get("linescore") or {}
            self.inning, self.phase = score.get("currentInning"), str(score.get("inningState", "")).lower()
            state = observed_state(observation)
            self.state_valid = state is not None and 1 <= state["inning_after"] <= 7
            if self.forecast:
                self.forecast.observe(now, observation, first_wall)
        super().baseball(now, row, first_wall)

    def opening_allowed(self):
        return (super().opening_allowed() and self.state_valid
                and len(self.opening_order_ids) < self.config.maximum_entry_orders
                and self.now - self.last_flattened >= self.config.reentry_cooldown
                and self.realized > -self.config.maximum_game_loss)

    def features(self, side, price):
        result = super().features(side, price)
        cutoff = self.now - self.config.movement_window
        while len(self.history) > 1 and self.history[1][0] <= cutoff:
            self.history.popleft()
        values = [v for _, v in self.history]
        result.update(mid_range=max(values) - min(values) if values else None,
                      book_age=self.now - self.book_seen if math.isfinite(self.book_seen) else None,
                      peer_age=self.now - self.peer_seen if math.isfinite(self.peer_seen) else None,
                      inning=self.inning, phase=self.phase,
                      regime="inning_break" if self.phase in {"middle", "end"} else "active_play",
                      reference_mode=self.config.reference_mode)
        reference = None
        if self.config.reference_mode == "sportsbook_anchor":
            reference = self.forecast.probability(self.now) if self.forecast else None
            result["sportsbook_anchor"] = self.forecast.anchor if self.forecast else None
        elif result["peer_fair"] is not None:
            # Consensus among already received prices, never a future mark.
            reference = sorted([result["mid"], result["microprice"], result["peer_fair"]])[1]
        result["fair_home_probability"] = reference
        if reference is not None:
            fair = reference if side == "yes" else 1 - reference
            result["estimated_roundtrip_edge"] = fair - price - fee(1, price, self.config.maker_fee_rate) - fee(1, 1-fair, self.config.maker_fee_rate)
        return result

    def signal_allows(self, side, price, features):
        reason = None
        if features["peer_age"] is None or features["peer_age"] > self.config.maximum_book_age:
            reason = "peer_book_stale_or_missing"
        elif features["peer_fair"] is None or abs(features["mid"] - features["peer_fair"]) > self.config.maximum_peer_disagreement + 1e-9:
            reason = "paired_books_disagree"
        elif features["mid_range"] is None or features["mid_range"] > self.config.maximum_mid_range + 1e-9:
            reason = "rapid_price_movement"
        elif features["queue_at_decision"] > self.config.maximum_queue:
            reason = "queue_too_deep"
        elif features["fair_home_probability"] is None:
            reason = (self.forecast.reason if self.forecast and self.forecast.reason else "reference_unavailable")
        elif features["estimated_roundtrip_edge"] < self.config.minimum_estimated_edge - 1e-9:
            reason = "insufficient_edge_after_maker_fees"
        if reason:
            self.rejections[reason] += 1
        return reason is None

    def decide(self):
        if self.book is None or self.now < self.last_decision + self.config.decision_seconds:
            return
        valid = (self.now - self.book_seen <= self.config.maximum_book_age
                 and 0 < self.book.bid("yes") <= self.book.ask("yes") < 1)
        if not valid:
            self.last_decision = self.now
            for key, order in self.orders.items():
                self.cancel(key, order)
            self.rejections["own_book_stale_or_invalid"] += 1
            return
        before = set(self.orders)
        super().decide()
        for key in self.orders.keys() - before:
            order = self.orders[key]
            self.attempts[key] = {"order_id": key, "side": order["side"], "price": order["price"],
                "quantity": order["remaining"], "submitted_at": order["submitted_at"],
                "submitted_as_reducing": order["submitted_as_reducing"], "status": "pending",
                "features": dict(order["features"]), "filled_quantity": 0.}

    def remove(self, order_id):
        order = self.orders[order_id]
        if order_id in self.attempts:
            self.attempts[order_id].update(status="filled" if order["remaining"] <= 1e-9 else
                "cancelled" if order["cancelling"] else "post_only_rejected", ended_at=self.now,
                arrived_at=order.get("arrived_at"), initial_queue_ahead=order.get("initial_queue_ahead"))
        super().remove(order_id)

    def fill(self, order_id, quantity, through):
        order = dict(self.orders[order_id])
        side = order["side"]
        mid = (self.book.bid(side) + self.book.ask(side)) / 2
        previous = self.position
        super().fill(order_id, quantity, through)
        row = self.fills[-1]
        row.update(ticker=self.ticker, fill_mid=mid,
                   spread_capture_per_contract=mid - row["price"],
                   queue_consumed_before_fill=max(0., order["initial_queue_ahead"] - order["queue_ahead"]))
        if row["opening_quantity"] > 1e-9:
            self.opening_order_ids.add(order_id)
        if abs(previous) > 1e-9 and abs(self.position) < 1e-9:
            self.last_flattened = self.now
        if order_id in self.attempts:
            self.attempts[order_id]["filled_quantity"] += quantity

    def execute_liquidation(self):
        previous = self.position
        start = len(self.fills)
        super().execute_liquidation()
        for row in self.fills[start:]:
            row["ticker"] = self.ticker
        if abs(previous) > 1e-9 and abs(self.position) < 1e-9:
            self.last_flattened = self.now

    def depth_mark(self, side, quantity):
        adjusted = Book({"yes_dollars_fp": [], "no_dollars_fp": []})
        adjusted.levels[side] = {p: max(0., q - self.used_depth[side].get(p, 0.))
                                 for p, q in self.book.levels[side].items()}
        return adjusted.liquidation_value(side, quantity)

    def mark(self, fill_index, horizon):
        row = self.fills[fill_index]
        quantity = row["opening_quantity"]
        if quantity <= 1e-9:
            return
        valid = bool(self.book and self.now - self.book_seen <= self.config.maximum_book_age
                     and 0 < self.book.bid(row["side"]) <= self.book.ask(row["side"]) < 1)
        result = {"fill_index": fill_index, "horizon_seconds": horizon, "observed_seconds": self.now,
                  "opening_quantity": quantity, "valid_book": valid, "depth_checked": True,
                  "mid_change_per_contract": None, "post_fill_mid_change_per_contract": None,
                  "liquidation_pnl_per_contract": None, "unfilled_mark_quantity": None}
        if valid:
            mid = (self.book.bid(row["side"]) + self.book.ask(row["side"])) / 2
            proceeds, missing = self.depth_mark(row["side"], quantity)
            result.update(mid_change_per_contract=mid - row["price"],
                          post_fill_mid_change_per_contract=mid - row["fill_mid"],
                          liquidation_pnl_per_contract=proceeds / quantity - row["price"] - row["fee"] / row["quantity"],
                          unfilled_mark_quantity=missing)
        self.markouts.append(result)

    def observe(self, now, message):
        self.advance_clock(now)
        kind, body = message.get("type"), message.get("msg", {})
        own, peer = body.get("market_ticker") == self.ticker, body.get("market_ticker") == self.peer_ticker
        if kind in {"orderbook_snapshot", "orderbook_delta"} and (own or peer):
            book = self.book if own else self.peer_book
            if kind == "orderbook_snapshot":
                if book is not None:
                    raise ValueError("Unexpected replacement snapshot")
                book = Book(body)
            else:
                if book is None:
                    raise ValueError("Delta before snapshot")
                book.update(body)
            if own:
                self.book, self.book_seen = book, now
                mid = (book.bid("yes") + book.ask("yes")) / 2
                self.history.append((now, mid))
                if kind == "orderbook_delta" and round(float(body["price_dollars"]), 4) not in book.levels[body["side"]]:
                    self.used_depth[body["side"]].pop(round(float(body["price_dollars"]), 4), None)
            else:
                self.peer_book, self.peer_seen = book, now
        elif kind == "trade" and own:
            self.trade(body)
        self.decide()

    def summary(self):
        result = super().summary()
        if abs(self.position) > 1e-9 and self.book:
            value, missing = self.depth_mark("yes" if self.position > 0 else "no", abs(self.position))
            result.update(inventory_liquidation_mark=value, inventory_without_bid_depth=missing,
                          liquidation_marked_pnl=self.realized + value - abs(self.position) * self.basis)
        attempts = [dict(a) for a in self.attempts.values()]
        for attempt in attempts:
            order = self.orders.get(attempt["order_id"])
            if order:
                attempt.update(status="cancelling" if order["cancelling"] else "resting" if order["active"] else "pending",
                               arrived_at=order.get("arrived_at"), initial_queue_ahead=order.get("initial_queue_ahead"))
        result.update(policy="selective_maker_v1", rejections=dict(self.rejections),
                      opening_order_attempts=sum(not a["submitted_as_reducing"] for a in attempts),
                      opening_attempts_filled=sum(a["order_id"] in self.opening_order_ids and not a["submitted_as_reducing"] for a in attempts),
                      reducing_orders_with_opening_fills=sum(a["order_id"] in self.opening_order_ids and a["submitted_as_reducing"] for a in attempts),
                      reducing_order_attempts=sum(a["submitted_as_reducing"] for a in attempts),
                      order_attempts=attempts, first_pitch_observed=self.game_started,
                      sportsbook_anchor=self.forecast.anchor if self.forecast else None,
                      sportsbook_status=self.forecast.reason if self.forecast else "not_configured")
        return result
