"""Queue-aware passive shadow replay of prospective order books and trades.

This is an execution diagnostic, not a fill guarantee or a live trader. It has
no credentials or order-submission path. Price-touch fills are not assumed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import gzip
import heapq
import json
import math
from pathlib import Path

from research.record import BookSequence


@dataclass(frozen=True)
class MakerConfig:
    submission_seconds: float = .68
    cancellation_seconds: float = .68
    decision_seconds: float = 1.
    lifetime_seconds: float = 10.
    minimum_spread: float = .02
    improve_when_spread: float = .03
    tick: float = .01
    contracts: float = 1.
    starting_cash: float = 100.
    # KXMLBGAME has maker multiplier 1 in the July 7, 2026 schedule.
    maker_fee_rate: float = .0175
    maximum_inventory_seconds: float = 60.

    def __post_init__(self):
        if not all(math.isfinite(v) and v >= 0 for v in asdict(self).values()):
            raise ValueError("All parameters must be finite and nonnegative")
        if min(self.decision_seconds, self.lifetime_seconds, self.tick, self.contracts, self.starting_cash) <= 0:
            raise ValueError("Decision interval, lifetime, tick, size, and cash must be positive")
        if not self.tick <= self.minimum_spread < 1 or self.improve_when_spread < self.minimum_spread:
            raise ValueError("Invalid spread thresholds")


class Book:
    def __init__(self, message):
        self.levels = {"yes": {}, "no": {}}
        for side in self.levels:
            for p, q in message[f"{side}_dollars_fp"]:
                price, quantity = round(float(p), 4), float(q)
                if not 0 < price < 1 or not math.isfinite(quantity) or quantity < 0:
                    raise ValueError("Invalid snapshot price or quantity")
                if quantity > 0:
                    self.levels[side][price] = quantity

    def update(self, message):
        side = message["side"]
        price = round(float(message["price_dollars"]), 4)
        change = float(message["delta_fp"])
        if side not in self.levels or not 0 < price < 1 or not math.isfinite(change):
            raise ValueError("Invalid depth update")
        quantity = self.levels[side].get(price, 0.) + change
        if quantity < -1e-6:
            raise ValueError("Negative book depth; cannot reconstruct this stream")
        if quantity <= 1e-6:
            self.levels[side].pop(price, None)
        else:
            self.levels[side][price] = quantity

    def bid(self, side):
        return max(self.levels[side], default=0.)

    def ask(self, side):
        return 1 - self.bid("no" if side == "yes" else "yes")

    def liquidation_value(self, side, quantity):
        """Walk visible bids; insufficient depth contributes zero to the mark."""
        value, fees = 0., 0.
        for price, size in sorted(self.levels[side].items(), reverse=True):
            used = min(size, quantity)
            value += used * price
            fees += fee(used, price, .07)
            quantity -= used
            if quantity <= 1e-9:
                break
        return value - fees, quantity


def fee(quantity, price, rate):
    principal = quantity * price
    raw = quantity * price * (1 - price) * rate
    return max(0., math.ceil((principal + raw) * 10000 - 1e-9) / 10000 - principal)


class PassiveReplay:
    """Single-market two-sided inventory strategy on local reception time.

    Queue ahead starts at ALL displayed size at order arrival. Book-size
    decreases do not improve our queue. Only compatible public trades consume
    it. A trade through our price fills us, including while cancellation is in
    flight. Post-only orders crossing at arrival are rejected. No favorable
    future observation can undo an exposed quote.
    """
    def __init__(self, ticker, config=MakerConfig()):
        self.ticker, self.config = ticker, config
        self.book = None
        self.orders, self.events, self.fills = {}, [], []
        self.cash = config.starting_cash
        self.position, self.basis, self.realized = 0., 0., 0.
        self.reserved, self.peak_inventory, self.total_fees = 0., 0., 0.
        self.now, self.last_decision, self.position_since = 0., -math.inf, None
        self.serial = 0
        self.seen_trades = set()
        self.counters = dict(submissions=0, cancellations=0, post_only_rejections=0,
                             cash_rejections=0, traded_through_fills=0, queue_fills=0,
                             stale_inventory_pauses=0)

    def schedule(self, when, kind, order_id):
        self.serial += 1
        heapq.heappush(self.events, (when, self.serial, kind, order_id))

    def remove(self, order_id):
        order = self.orders.pop(order_id)
        self.cash += order["reserved"]
        self.reserved -= order["reserved"]

    def advance(self, now):
        # Equal-time market observations cannot retroactively influence an
        # order that should already have arrived. Use the preceding book.
        while self.events and self.events[0][0] <= now:
            stamp, _, kind, order_id = heapq.heappop(self.events)
            order = self.orders.get(order_id)
            if order is None:
                continue
            if kind == "cancel":
                self.remove(order_id)
            elif self.book is None or order["price"] >= self.book.ask(order["side"]) - 1e-9:
                self.counters["post_only_rejections"] += 1
                self.remove(order_id)
            else:
                order["active"] = True
                order["arrived_at"] = stamp
                order["queue_ahead"] = self.book.levels[order["side"]].get(order["price"], 0.)
        self.now = now

    def decide(self):
        c = self.config
        if self.book is None or self.now < self.last_decision + c.decision_seconds:
            return
        self.last_decision = self.now
        spread = self.book.ask("yes") - self.book.bid("yes")
        valid = spread >= c.minimum_spread - 1e-9 and .05 <= self.book.bid("yes") <= .95
        stale_inventory = self.position_since is not None and self.now - self.position_since > c.maximum_inventory_seconds
        if stale_inventory:
            self.counters["stale_inventory_pauses"] += 1
        for side, sign in (("yes", 1), ("no", -1)):
            # After an inventory fill, quote only the reducing side. Do not
            # turn an unpaired loss into more exposure through averaging down.
            desired = valid and (abs(self.position) < 1e-9 or self.position * sign < 0)
            if stale_inventory:
                desired = self.position * sign < 0
            price = self.book.bid(side)
            if spread >= c.improve_when_spread - 1e-9:
                price += c.tick
            price = round(price, 4)
            desired = desired and 0 < price < self.book.ask(side) - 1e-9
            existing = [(i, o) for i, o in self.orders.items() if o["side"] == side]
            for order_id, order in existing:
                expired = self.now - order["submitted_at"] >= c.lifetime_seconds
                if (not desired or abs(order["price"] - price) > 1e-9 or expired) and not order["cancelling"]:
                    order["cancelling"] = True
                    self.schedule(self.now + c.cancellation_seconds, "cancel", order_id)
                    self.counters["cancellations"] += 1
            if existing or not desired:
                continue
            quantity = min(c.contracts, abs(self.position)) if self.position * sign < 0 else c.contracts
            # Reserve extra rounding room if fees apply to fragmented fills.
            rounding_room = math.ceil(quantity * 100) * .0001 if c.maker_fee_rate else 0.
            reserve = quantity * price + fee(quantity, price, c.maker_fee_rate) + rounding_room
            if reserve > self.cash + 1e-9:
                self.counters["cash_rejections"] += 1
                continue
            self.cash -= reserve
            self.reserved += reserve
            order_id = self.serial + 1
            self.orders[order_id] = {"side": side, "price": price, "remaining": quantity,
                "reserved": reserve, "submitted_at": self.now, "active": False,
                "cancelling": False, "queue_ahead": None}
            self.schedule(self.now + c.submission_seconds, "submit", order_id)
            self.counters["submissions"] += 1

    def advance_clock(self, now):
        # Timers keep running between messages. Quiet books must not extend a
        # quote's lifetime until the next trade happens to arrive.
        if self.book is not None:
            while self.last_decision + self.config.decision_seconds < now - 1e-9:
                self.advance(self.last_decision + self.config.decision_seconds)
                self.decide()
        self.advance(now)

    def fill(self, order_id, quantity, through):
        order = self.orders[order_id]
        price, sign = order["price"], (1 if order["side"] == "yes" else -1)
        released = order["reserved"] * quantity / order["remaining"]
        charge = fee(quantity, price, self.config.maker_fee_rate)
        self.cash += released
        self.reserved -= released
        order["reserved"] -= released
        order["remaining"] -= quantity
        self.total_fees += charge
        closing = min(quantity, abs(self.position)) if self.position * sign < 0 else 0.
        opening = quantity - closing
        if closing:
            self.cash += closing * (1 - price) - charge * closing / quantity
            self.realized += closing * (1 - price - self.basis) - charge * closing / quantity
            self.position += sign * closing
        if opening > 1e-9:
            previous = abs(self.position)
            cost = opening * price + charge * opening / quantity
            self.cash -= cost
            self.basis = (previous * self.basis + cost) / (previous + opening)
            self.position += sign * opening
            if previous < 1e-9:
                self.position_since = self.now
        if abs(self.position) < 1e-9:
            self.position, self.basis, self.position_since = 0., 0., None
        self.peak_inventory = max(self.peak_inventory, abs(self.position))
        self.fills.append({"received_seconds": self.now, "side": order["side"], "price": price,
            "quantity": quantity, "closing_quantity": closing, "fee": charge,
            "cancel_pending": order["cancelling"], "traded_through": through,
            "position_after": self.position, "realized_pnl_after": self.realized})
        self.counters["traded_through_fills" if through else "queue_fills"] += 1
        if order["remaining"] <= 1e-9:
            self.remove(order_id)

    def trade(self, message):
        identity = message.get("trade_id")
        if not identity or identity in self.seen_trades or message.get("is_block_trade"):
            return
        self.seen_trades.add(identity)
        taker = message.get("taker_outcome_side", message.get("taker_side"))
        if taker not in ("yes", "no"):
            raise ValueError("Unknown trade side")
        resting_side = "no" if taker == "yes" else "yes"
        price = float(message[f"{resting_side}_price_dollars"])
        volume = float(message["count_fp"])
        if not 0 < price < 1 or not math.isfinite(volume) or volume <= 0:
            raise ValueError("Invalid public trade")
        for order_id, order in list(self.orders.items()):
            execution_time = message.get("exchange_elapsed_seconds", self.now)
            if (not order["active"] or order["side"] != resting_side
                    or min(self.now, execution_time) <= order["arrived_at"]):
                continue
            through = price < order["price"] - 1e-9
            if through:
                # Adverse trade-throughs cannot be silently dropped because
                # the public book no longer contains our hypothetical quote.
                used = min(order["remaining"], volume)
            elif abs(price - order["price"]) < 1e-9:
                ahead = min(order["queue_ahead"], volume)
                order["queue_ahead"] -= ahead
                used = min(order["remaining"], volume - ahead)
            else:
                continue
            if used > 1e-9:
                self.fill(order_id, used, through)

    def observe(self, now, message):
        self.advance_clock(now)
        kind, body = message.get("type"), message.get("msg", {})
        if body.get("market_ticker") == self.ticker:
            if kind == "orderbook_snapshot":
                if self.book is not None:
                    raise ValueError("Unexpected replacement snapshot; queue continuity is unknown")
                self.book = Book(body)
            elif kind == "orderbook_delta":
                if self.book is None:
                    raise ValueError("Delta before snapshot")
                self.book.update(body)
            elif kind == "trade":
                self.trade(body)
        self.decide()

    def summary(self):
        inventory_value, missing_depth = 0., abs(self.position)
        if self.book and abs(self.position) > 1e-9:
            inventory_value, missing_depth = self.book.liquidation_value(
                "yes" if self.position > 0 else "no", abs(self.position))
        equity = self.cash + self.reserved + inventory_value
        return {"ticker": self.ticker, "config": asdict(self.config), **self.counters,
            "shadow_fill_events": len(self.fills), "filled_contracts": sum(f["quantity"] for f in self.fills),
            "entry_contracts": sum(f["quantity"] - f["closing_quantity"] for f in self.fills),
            "closing_contracts": sum(f["closing_quantity"] for f in self.fills),
            "realized_pnl": self.realized, "inventory": self.position, "inventory_cost_per_contract": self.basis,
            "inventory_liquidation_mark": inventory_value, "inventory_without_bid_depth": missing_depth,
            "cash": self.cash, "pending_reserved_cash": self.reserved, "fees": self.total_fees,
            "liquidation_marked_pnl": equity - self.config.starting_cash,
            "peak_inventory": self.peak_inventory, "outstanding_orders": len(self.orders),
            "deployment_ready": False, "status": "prospective_queue_shadow_not_verified_fills"}


def replay(path, ticker, config=MakerConfig()):
    engine, sequence = PassiveReplay(ticker, config), BookSequence()
    first = previous = first_wall = None
    gap, ended = None, False
    with gzip.open(path, "rt") as stream:
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
            if row["type"] == "connection_gap" or (ended and row["type"] == "connection_start"):
                gap = "Connection gap: outstanding order fills/queue are indeterminate after this point"
                break
            if row["type"] == "connection_end":
                engine.advance_clock(when)
                ended = True
            if row["type"] == "kalshi_message":
                try:
                    if row["message"].get("type") == "error":
                        raise ValueError("Recorded websocket subscription error")
                    sequence.observe(row["message"])
                    if row["message"].get("type") == "trade":
                        body = row["message"]["msg"]
                        if first_wall is None or body.get("ts_ms") is None:
                            raise ValueError("Trade lacks an exchange/receipt timestamp")
                        body["exchange_elapsed_seconds"] = float(body["ts_ms"]) / 1000 - first_wall
                    engine.observe(when, row["message"])
                except ValueError as error:
                    gap = str(error)
                    break
    if engine.book is None and gap is None:
        gap = "No snapshot for the requested market"
    result = engine.summary()
    result.update({"input": str(path), "observed_seconds": (previous - first) / 1e9 if first is not None else 0,
        "continuity_error": gap, "full_replay_valid": gap is None,
        "limitations": ["Local receipt time substitutes for exchange processing time",
            "Exchange trade timestamps must follow modeled arrival; clocks are aligned using the initial local UTC timestamp",
            "Displayed queue plus public trades cannot verify actual priority or hidden execution semantics",
            "Depth cancellations never reduce queue ahead; adverse trade-throughs still fill",
            "Partial inventory is marked against visible bid depth after taker fees, never omitted",
            "Open orders remain exposed at capture end; ending marks are not completed liquidation",
            "No live orders, measured fills, or profitability claim"]})
    return result, engine.fills


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--ticker", required=True, help="One market; YES/NO quotes share its inventory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latency", type=float, default=.68)
    parser.add_argument("--maker-fee-rate", type=float, default=MakerConfig().maker_fee_rate)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; use a new path")
    result, fills = replay(args.capture, args.ticker, MakerConfig(submission_seconds=args.latency,
                           cancellation_seconds=args.latency, maker_fee_rate=args.maker_fee_rate))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({**result, "fills": fills}, indent=2, allow_nan=False))
    print(json.dumps({k: result[k] for k in ("shadow_fill_events", "inventory", "liquidation_marked_pnl", "full_replay_valid")}, indent=2))


if __name__ == "__main__":
    main()
