"""Prospective passive strategy variants with observed baseball regimes.

Research only: never submits orders. See PASSIVE_SLATE_PROTOCOL.md.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime
import heapq
import math

from research.passive import Book, MakerConfig, PassiveReplay, fee


@dataclass(frozen=True)
class StudyConfig(MakerConfig):
    lifetime_seconds: float = 60.
    maximum_inventory_seconds: float = 30.
    minimum_spread: float = .01
    improve_when_spread: float = 2.
    price_filter: str = "none"
    entry_regime: str = "live"
    minimum_estimated_edge: float = .005
    opening_break_seconds: float = 45.
    maximum_feed_age: float = 30.
    liquidation_penalty: float = .01

    def __post_init__(self):
        MakerConfig(**{f.name: getattr(self, f.name) for f in fields(MakerConfig)})
        if self.price_filter not in {"none", "microprice", "peer"}:
            raise ValueError("Unknown price filter")
        if self.entry_regime not in {"live", "break"}:
            raise ValueError("Unknown baseball regime")
        for name in ("minimum_estimated_edge", "opening_break_seconds", "maximum_feed_age", "liquidation_penalty"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"Invalid {name}")


def candidates():
    return {f"{quote}_{signal}_{regime}": StudyConfig(
        minimum_spread=.01 if quote == "join" else .02,
        improve_when_spread=2. if quote == "join" else .03,
        price_filter=signal, entry_regime=regime)
        for quote in ("join", "improve") for signal in ("none", "microprice", "peer")
        for regime in ("live", "break")}


class MakerStudyReplay(PassiveReplay):
    def __init__(self, ticker, peer_ticker, game_pk, config=StudyConfig()):
        super().__init__(ticker, config)
        self.peer_ticker, self.peer_book, self.game_pk = peer_ticker, None, game_pk
        self.game_live, self.feed_seen = False, -math.inf
        self.game_final, self.first_live_seen, self.final_seen = False, None, None
        self.break_key, self.break_since = None, None
        self.liquidation = None
        self.used_depth = {"yes": {}, "no": {}}
        self.markouts = []
        self.counters.update(liquidation_orders=0, liquidation_fills=0, liquidation_misses=0,
                             opening_signal_rejections=0, opening_regime_rejections=0)

    def baseball(self, now, row, first_wall):
        self.advance_clock(now)
        if row.get("game_pk") != self.game_pk:
            self.decide()
            return
        self.feed_seen = now
        observation = row.get("observation")
        if observation is not None:
            status = (observation.get("gameData") or {}).get("status") or {}
            self.game_live = str(status.get("abstractGameState", "")).lower() == "live"
            self.game_final = str(status.get("abstractGameState", "")).lower() == "final"
            if self.game_live and self.first_live_seen is None:
                self.first_live_seen = now
            if self.game_final and self.final_seen is None:
                self.final_seen = now
            score = observation.get("linescore") or {}
            phase = str(score.get("inningState", "")).lower()
            key = (score.get("currentInning"), phase)
            play = observation.get("currentPlay") or {}
            is_break = phase in {"middle", "end"} and (play.get("count") or {}).get("outs") == 3
            if is_break:
                if key != self.break_key:
                    end = (play.get("about") or {}).get("endTime")
                    # An old break seen on a delayed first poll does not get a
                    # fresh 45 seconds. A missing source clock disables entry.
                    self.break_since = None
                    if end and first_wall is not None:
                        source = datetime.fromisoformat(end.replace("Z", "+00:00")).timestamp() - first_wall
                        if source <= now:
                            self.break_since = source
                    self.break_key = key
            else:
                self.break_key, self.break_since = None, None
        self.decide()

    def opening_allowed(self):
        fresh = self.game_live and self.now - self.feed_seen <= self.config.maximum_feed_age
        if self.config.entry_regime == "live":
            return fresh
        return fresh and self.break_since is not None and 0 <= self.now - self.break_since <= self.config.opening_break_seconds

    def features(self, side, price):
        yes_bid, yes_ask = self.book.bid("yes"), self.book.ask("yes")
        yes_size = self.book.levels["yes"].get(yes_bid, 0.)
        no_size = self.book.levels["no"].get(self.book.bid("no"), 0.)
        mid = (yes_bid + yes_ask) / 2
        micro = (yes_ask * yes_size + yes_bid * no_size) / max(yes_size + no_size, 1e-9)
        peer_mid = None
        if self.peer_book and self.peer_book.bid("yes") > 0 and self.peer_book.bid("no") > 0:
            peer_mid = 1 - (self.peer_book.bid("yes") + self.peer_book.ask("yes")) / 2
        return {"mid": mid, "microprice": micro, "peer_fair": peer_mid,
            "spread": yes_ask - yes_bid, "yes_size": yes_size, "no_size": no_size,
            "queue_at_decision": self.book.levels[side].get(price, 0.),
            "feed_age": self.now - self.feed_seen,
            "break_age": self.now - self.break_since if self.break_since is not None else None,
            "side": side, "price": price}

    def signal_allows(self, side, price, features):
        mode = self.config.price_filter
        if mode == "none":
            return True
        fair = features["microprice"] if mode == "microprice" else features["peer_fair"]
        if fair is None:
            return False
        fair = fair if side == "yes" else 1 - fair
        return fair - price - fee(1, price, self.config.maker_fee_rate) >= self.config.minimum_estimated_edge - 1e-9

    def cancel(self, order_id, order):
        if not order["cancelling"]:
            order["cancelling"] = True
            self.schedule(self.now + self.config.cancellation_seconds, "cancel", order_id)
            self.counters["cancellations"] += 1

    def decide(self):
        c = self.config
        if self.book is None or self.now < self.last_decision + c.decision_seconds:
            return
        self.last_decision = self.now
        expired = self.position_since is not None and self.now - self.position_since >= c.maximum_inventory_seconds
        if self.liquidation is not None:
            return
        if expired:
            for order_id, order in self.orders.items():
                self.cancel(order_id, order)
            if not self.orders:
                side = "yes" if self.position > 0 else "no"
                limit = self.book.bid(side) - c.liquidation_penalty
                if limit <= 0:
                    return
                self.liquidation = {"side": side, "quantity": abs(self.position), "limit": limit,
                                    "submitted_at": self.now}
                self.schedule(self.now + c.submission_seconds, "liquidate", -1)
                self.counters["liquidation_orders"] += 1
            return
        spread = self.book.ask("yes") - self.book.bid("yes")
        valid = spread >= c.minimum_spread - 1e-9 and .05 <= self.book.bid("yes") <= .95
        can_open = self.opening_allowed()
        for side, sign in (("yes", 1), ("no", -1)):
            reducing = self.position * sign < 0
            flat = abs(self.position) < 1e-9
            price = self.book.bid(side) + (c.tick if spread >= c.improve_when_spread - 1e-9 else 0.)
            price = round(price, 4)
            feature = self.features(side, price)
            good_signal = self.signal_allows(side, price, feature)
            # Risk-reducing quotes remain available even when the entry
            # filter or baseball phase no longer permits a new position.
            desired = (reducing or (flat and valid and can_open and good_signal))
            desired = desired and 0 < price < self.book.ask(side) - 1e-9
            if flat and valid and not can_open:
                self.counters["opening_regime_rejections"] += 1
            elif flat and valid and not good_signal:
                self.counters["opening_signal_rejections"] += 1
            existing = [(i, o) for i, o in self.orders.items() if o["side"] == side]
            for order_id, order in existing:
                excessive_close = reducing and order["remaining"] > abs(self.position) + 1e-9
                if (not desired or excessive_close or abs(order["price"] - price) > 1e-9
                        or self.now - order["submitted_at"] >= c.lifetime_seconds):
                    self.cancel(order_id, order)
            if existing or not desired:
                continue
            quantity = min(c.contracts, abs(self.position)) if reducing else c.contracts
            reserve = quantity * price + fee(quantity, price, c.maker_fee_rate) + math.ceil(quantity * 100) * .0001
            if reserve > self.cash + 1e-9:
                self.counters["cash_rejections"] += 1
                continue
            self.cash -= reserve
            self.reserved += reserve
            order_id = self.serial + 1
            self.orders[order_id] = {"side": side, "price": price, "remaining": quantity,
                "reserved": reserve, "submitted_at": self.now, "active": False,
                "cancelling": False, "queue_ahead": None, "features": feature,
                "submitted_as_reducing": reducing}
            self.schedule(self.now + c.submission_seconds, "submit", order_id)
            self.counters["submissions"] += 1

    def advance(self, now):
        while self.events and self.events[0][0] <= now:
            stamp, _, kind, order_id = heapq.heappop(self.events)
            self.now = stamp
            if kind == "liquidate":
                self.execute_liquidation()
                continue
            if kind.startswith("mark_"):
                self.mark(order_id, int(kind.split("_")[1]))
                continue
            order = self.orders.get(order_id)
            if order is None:
                continue
            if kind == "cancel":
                self.remove(order_id)
            elif self.book is None or order["price"] >= self.book.ask(order["side"]) - 1e-9:
                self.counters["post_only_rejections"] += 1
                self.remove(order_id)
            else:
                order["active"], order["arrived_at"] = True, stamp
                order["queue_ahead"] = self.book.levels[order["side"]].get(order["price"], 0.)
                order["initial_queue_ahead"] = order["queue_ahead"]
        self.now = now

    def fill(self, order_id, quantity, through):
        order = dict(self.orders[order_id])
        super().fill(order_id, quantity, through)
        row = self.fills[-1]
        row.update(order_id=order_id, action="buy", liquidity_role="maker", features=order["features"],
                   submitted_at=order["submitted_at"], arrived_at=order["arrived_at"],
                   initial_queue_ahead=order["initial_queue_ahead"],
                   queue_wait_seconds=self.now - order["arrived_at"],
                   submitted_as_reducing=order["submitted_as_reducing"],
                   opening_quantity=quantity - row["closing_quantity"])
        for horizon in (5, 30, 60):
            self.schedule(self.now + horizon, f"mark_{horizon}", len(self.fills) - 1)

    def execute_liquidation(self):
        order = self.liquidation
        self.liquidation = None
        if order is None or self.book is None:
            return
        side = order["side"]
        if (side == "yes" and self.position <= 0) or (side == "no" and self.position >= 0):
            raise ValueError("Liquidation would open or reverse a position")
        remaining = min(order["quantity"], abs(self.position))
        filled = 0.
        for quoted_price, displayed in sorted(self.book.levels[side].items(), reverse=True):
            price = quoted_price - self.config.liquidation_penalty
            if price < order["limit"] - 1e-9 or price <= 0:
                continue
            available = max(0., displayed - self.used_depth[side].get(quoted_price, 0.))
            quantity = min(available, remaining)
            if quantity <= 1e-9:
                continue
            charge = fee(quantity, price, .07)
            self.cash += quantity * price - charge
            self.realized += quantity * (price - self.basis) - charge
            self.total_fees += charge
            self.position += (-1 if side == "yes" else 1) * quantity
            remaining -= quantity
            filled += quantity
            self.used_depth[side][quoted_price] = self.used_depth[side].get(quoted_price, 0.) + quantity
            self.fills.append({"received_seconds": self.now, "side": side, "action": "sell",
                "price": price, "quantity": quantity, "opening_quantity": 0., "closing_quantity": quantity,
                "fee": charge, "liquidity_role": "taker", "cancel_pending": False, "traded_through": False,
                "submitted_at": order["submitted_at"], "position_after": self.position,
                "realized_pnl_after": self.realized})
            if remaining <= 1e-9:
                break
        if abs(self.position) <= 1e-9:
            self.position, self.basis, self.position_since = 0., 0., None
        self.counters["liquidation_fills" if filled else "liquidation_misses"] += 1

    def mark(self, fill_index, horizon):
        if self.book is None:
            return
        row = self.fills[fill_index]
        side, quantity = row["side"], row["quantity"]
        bid, ask = self.book.bid(side), self.book.ask(side)
        mid = (bid + ask) / 2
        self.markouts.append({"fill_index": fill_index, "horizon_seconds": horizon,
            "observed_seconds": self.now, "side": side, "opening_quantity": row["opening_quantity"],
            "mid_change_per_contract": mid - row["price"],
            "liquidation_pnl_per_contract": bid - row["price"] - row["fee"] / quantity - fee(quantity, bid, .07) / quantity,
            "valid_book": 0 < bid <= ask < 1})

    def observe(self, now, message):
        self.advance_clock(now)
        kind, body = message.get("type"), message.get("msg", {})
        if body.get("market_ticker") == self.peer_ticker:
            if kind == "orderbook_snapshot":
                if self.peer_book is not None:
                    raise ValueError("Unexpected peer snapshot")
                self.peer_book = Book(body)
            elif kind == "orderbook_delta":
                if self.peer_book is None:
                    raise ValueError("Peer delta before snapshot")
                self.peer_book.update(body)
        # The parent advances no additional ticks because the clock is already
        # current, and handles immutable fills before new entry decisions.
        super().observe(now, message)
        if body.get("market_ticker") == self.ticker and kind == "orderbook_delta":
            side, price = body["side"], round(float(body["price_dollars"]), 4)
            if price not in self.book.levels[side]:
                self.used_depth[side].pop(price, None)

    def summary(self):
        result = super().summary()
        result["game_pk"] = self.game_pk
        result.update(game_observed_live=self.first_live_seen is not None,
                      game_observed_final=self.final_seen is not None,
                      first_live_seen_seconds=self.first_live_seen,
                      final_seen_seconds=self.final_seen)
        result["entry_orders_filled"] = len({f["order_id"] for f in self.fills
                                             if f["opening_quantity"] > 1e-9})
        result["maker_fill_events"] = sum(f["liquidity_role"] == "maker" for f in self.fills)
        result["taker_fill_events"] = sum(f["liquidity_role"] == "taker" for f in self.fills)
        result["fills_during_cancellation"] = sum(f["cancel_pending"] for f in self.fills)
        result["markouts"] = {}
        for horizon in (5, 30, 60):
            rows = [m for m in self.markouts if m["horizon_seconds"] == horizon and m["valid_book"] and m["opening_quantity"] > 0]
            weight = sum(m["opening_quantity"] for m in rows)
            result["markouts"][str(horizon)] = {"observations": len(rows), "opening_contracts": weight,
                "weighted_mean_mid_change": sum(m["mid_change_per_contract"] * m["opening_quantity"] for m in rows) / weight if weight else None,
                "weighted_mean_liquidation_pnl": sum(m["liquidation_pnl_per_contract"] * m["opening_quantity"] for m in rows) / weight if weight else None}
        return result
