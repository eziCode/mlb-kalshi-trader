"""A shared cash account and chronological clock for passive game replays."""
from __future__ import annotations

from dataclasses import dataclass
import math

from research.maker_study import MakerStudyReplay


@dataclass
class CashAccount:
    starting_cash: float
    cash: float

    @classmethod
    def funded(cls, amount):
        if not math.isfinite(amount) or amount <= 0:
            raise ValueError("Starting cash must be finite and positive")
        return cls(amount, amount)


class PortfolioGame(MakerStudyReplay):
    def __init__(self, ticker, peer_ticker, game_pk, config, account):
        # Parent initialization must not reset an already shared bankroll.
        self.account = None
        super().__init__(ticker, peer_ticker, game_pk, config)
        self.account = account

    @property
    def cash(self):
        return self.account.cash if self.account is not None else self._initial_cash

    @cash.setter
    def cash(self, amount):
        if self.account is None:
            self._initial_cash = amount
        else:
            if amount < -1e-7:
                raise ValueError("Shared cash became negative")
            self.account.cash = amount

    def summary(self):
        result = super().summary()
        # Attribute inventory and realized gains to a game without treating
        # the common cash account as a separate deposit in every game.
        result["cash"] = None
        result["liquidation_marked_pnl"] = (self.realized
            + result["inventory_liquidation_mark"] - abs(self.position) * self.basis)
        result["account_scope"] = "Game PnL attribution within a shared cash account"
        return result


class PortfolioClock:
    """Run every timer in timestamp order before consuming the next message.

    Advancing one game to the message time and then another can let a later
    cash release finance an earlier order. Merge their clocks instead. Equal
    timestamps use game_pk order, frozen independently of game outcomes.
    """
    def __init__(self, engines, account):
        self.engines = sorted(engines, key=lambda e: e.game_pk)
        self.account = account
        self.now = 0.
        self.maximum_committed_cost = 0.

    def advance_before(self, when, inclusive=False):
        if when < self.now - 1e-9:
            raise ValueError("Portfolio clock moved backwards")
        while True:
            due = []
            for engine in self.engines:
                decision = (engine.last_decision + engine.config.decision_seconds
                            if engine.book is not None else math.inf)
                event = engine.events[0][0] if engine.events else math.inf
                due.append((min(decision, event), engine.game_pk, engine))
            stamp, _, engine = min(due, key=lambda item: item[:2])
            if stamp > when or (not inclusive and stamp >= when - 1e-9):
                break
            engine.advance(stamp)
            engine.decide()
            self.measure()
        # Flush execution events at the observation timestamp using the
        # preceding book. Decisions at this timestamp may consume the message.
        for engine in self.engines:
            engine.advance(when)
        self.now = when

    def measure(self):
        committed = sum(e.reserved + abs(e.position) * e.basis for e in self.engines)
        self.maximum_committed_cost = max(self.maximum_committed_cost, committed)

    def summary(self):
        self.measure()
        games = [e.summary() for e in self.engines]
        reserve = sum(e.reserved for e in self.engines)
        inventory = sum(g["inventory_liquidation_mark"] for g in games)
        equity = self.account.cash + reserve + inventory
        pnl = equity - self.account.starting_cash
        if abs(pnl - sum(g["liquidation_marked_pnl"] for g in games)) > 1e-6:
            raise ValueError("Shared equity does not reconcile with game attribution")
        return {"starting_cash": self.account.starting_cash, "free_cash": self.account.cash,
                "reserved_cash": reserve, "inventory_liquidation_mark": inventory,
                "equity": equity, "liquidation_marked_pnl": pnl,
                "maximum_committed_cost": self.maximum_committed_cost,
                "account_scope": "One chronological shared cash account"}
