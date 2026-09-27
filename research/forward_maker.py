"""Passive hold-to-settlement entries using the frozen probability correction."""
from __future__ import annotations

import argparse
import heapq
from pathlib import Path

from research.forward_value import ForwardValueGame, run
from research.passive import fee


class ForwardMakerGame(ForwardValueGame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counters.update(maker_orders=0, maker_cancels=0, maker_post_only_rejections=0)

    def cancel_quote(self, order_id, order):
        if not order["cancelling"]:
            order["cancelling"] = True
            self.schedule(self.now + self.config.cancellation_seconds, "cancel", order_id)
            self.counters["maker_cancels"] += 1

    def decide(self):
        if self.book is None or self.now < self.last_decision + self.config.decision_seconds:
            return
        self.last_decision = self.now
        estimate = self.estimate() if not self.entry_filled else None
        choices = []
        if estimate is not None:
            p, features = estimate
            for side, book, fair in (("yes", self.book, p), ("no", self.peer_book, 1-p)):
                spread = book.ask("yes") - book.bid("yes")
                price = round(book.bid("yes") + (.01 if spread >= .03 - 1e-9 else 0.), 4)
                edge = fair - price - fee(1, price, .0175)
                if .05 <= book.ask("yes") <= .95 and 0 < price < book.ask("yes") - 1e-9 and edge >= .01 - 1e-9:
                    choices.append((edge, side, price))
        desired = max(choices) if choices else None
        for order_id, order in self.orders.items():
            if (desired is None or order["side"] != desired[1] or abs(order["price"] - desired[2]) > 1e-9
                    or self.now - order["submitted_at"] >= 60.):
                self.cancel_quote(order_id, order)
        if self.orders or desired is None:
            return
        _, side, price = desired
        reserve = price + fee(1, price, .0175) + .01
        if reserve > self.cash + 1e-9:
            self.counters["cash_rejections"] += 1
            return
        self.cash -= reserve
        self.reserved += reserve
        order_id = self.serial + 1
        self.orders[order_id] = dict(side=side, price=price, remaining=1., reserved=reserve,
            ticker=self.ticker if side == "yes" else self.peer_ticker, submitted_at=self.now,
            active=False, cancelling=False, queue_ahead=None, features=features)
        self.schedule(self.now + self.config.submission_seconds, "submit", order_id)
        self.counters["maker_orders"] += 1

    def advance(self, now):
        while self.events and self.events[0][0] <= now:
            stamp, _, kind, order_id = heapq.heappop(self.events)
            self.now = stamp
            order = self.orders.get(order_id)
            if order is None:
                continue
            if kind == "cancel":
                self.remove(order_id)
            elif kind == "submit":
                book = self.book if order["side"] == "yes" else self.peer_book
                if order["price"] >= book.ask("yes") - 1e-9:
                    self.counters["maker_post_only_rejections"] += 1
                    self.remove(order_id)
                else:
                    order.update(active=True, arrived_at=stamp,
                        queue_ahead=book.levels["yes"].get(order["price"], 0.),
                        initial_queue_ahead=book.levels["yes"].get(order["price"], 0.))
            else:
                raise ValueError("Unexpected event in passive value policy")
        self.now = now

    def trade(self, body):
        identity = body.get("trade_id")
        if not identity or identity in self.seen_trades or body.get("is_block_trade"):
            return
        self.seen_trades.add(identity)
        taker = body.get("taker_outcome_side", body.get("taker_side"))
        if taker not in ("yes", "no"):
            raise ValueError("Unknown public trade side")
        if taker != "no":
            return
        price, volume = float(body["yes_price_dollars"]), float(body["count_fp"])
        if not 0 < price < 1 or not 0 < volume < float("inf"):
            raise ValueError("Invalid passive fill evidence")
        for order_id, order in list(self.orders.items()):
            if (not order["active"] or body.get("market_ticker") != order["ticker"]
                    or min(self.now, body.get("exchange_elapsed_seconds", self.now)) <= order["arrived_at"]):
                continue
            through = price < order["price"] - 1e-9
            if through:
                used = min(volume, order["remaining"])
            elif abs(price - order["price"]) < 1e-9:
                ahead = min(order["queue_ahead"], volume)
                order["queue_ahead"] -= ahead
                used = min(order["remaining"], volume - ahead)
            else:
                continue
            if used > 1e-9:
                self.fill_quote(order_id, used, through)

    def fill_quote(self, order_id, quantity, through):
        order = self.orders[order_id]
        price = order["price"]
        charge = fee(quantity, price, .0175)
        cost = quantity * price + charge
        if cost > order["reserved"] + 1e-9:
            raise ValueError("Passive fill exceeds reserved cash")
        previous = abs(self.position)
        self.basis = (previous * self.basis + cost) / (previous + quantity)
        self.position += quantity * (1 if order["side"] == "yes" else -1)
        self.position_since = self.now if self.position_since is None else self.position_since
        order["remaining"] -= quantity
        order["reserved"] -= cost
        self.reserved -= cost
        self.total_fees += charge
        self.peak_inventory = max(self.peak_inventory, abs(self.position))
        self.entry_filled = True
        self.fills.append(dict(order_id=order_id, received_seconds=self.now, side=order["side"],
            ticker=order["ticker"], action="buy", liquidity_role="maker", price=price, quantity=quantity,
            fee=charge, opening_quantity=quantity, closing_quantity=0., cancel_pending=order["cancelling"],
            traded_through=through, position_after=self.position, realized_pnl_after=self.realized,
            submitted_at=order["submitted_at"], arrived_at=order["arrived_at"], features=order["features"],
            initial_queue_ahead=order["initial_queue_ahead"], queue_wait_seconds=self.now-order["arrived_at"]))
        self.counters["traded_through_fills" if through else "queue_fills"] += 1
        if order["remaining"] <= 1e-9:
            self.remove(order_id)

    def observe(self, now, message):
        self.advance_clock(now)
        if message.get("type") == "trade" and (message.get("msg") or {}).get("market_ticker") == self.peer_ticker:
            self.trade(message["msg"])
        super().observe(now, message)

    def summary(self):
        result = super().summary()
        result["policy"] = "frozen_corrected_01c_passive_hold"
        result["actual_entry_fee_rate"] = .0175
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--slate", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--frozen-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--latency", type=float, choices=[.68, 1.5, 3.], default=.68)
    parser.set_defaults(penalty=0.)
    run(parser.parse_args(), engine_class=ForwardMakerGame)


if __name__ == "__main__":
    main()
