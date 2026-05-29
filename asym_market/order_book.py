"""A minimal single-unit limit order book with price-time priority.

Design choices (kept deliberately simple and testable):

* Every order is size 1. There is no partial fill.
* Bids rank by (highest price, earliest timestamp); asks by (lowest price,
  earliest timestamp).
* ``post_bid`` / ``post_ask`` are *marketable limits*: if the new order
  crosses the opposite best, it executes immediately at the resting (maker)
  price and does not rest. Otherwise it rests.
* ``take_bid`` / ``take_ask`` are explicit aggressions against the current
  best, used when a player wants to cross without leaving a resting order.
* Self-trading is forbidden: an aggression that would match the player's own
  resting order is rejected by the environment via ``legal_actions`` (the book
  itself also refuses to match an order against its owner).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .types import Order, Side, Trade


class OrderBook:
    def __init__(self) -> None:
        self._bids: List[Order] = []
        self._asks: List[Order] = []
        self._next_id: int = 0

    # ------------------------------------------------------------------ views
    @property
    def bids(self) -> List[Order]:
        """Resting bids, best first."""
        return sorted(self._bids, key=lambda o: (-o.price, o.ts))

    @property
    def asks(self) -> List[Order]:
        """Resting asks, best first."""
        return sorted(self._asks, key=lambda o: (o.price, o.ts))

    def best_bid(self) -> Optional[Order]:
        bids = self.bids
        return bids[0] if bids else None

    def best_ask(self) -> Optional[Order]:
        asks = self.asks
        return asks[0] if asks else None

    def best_bid_price(self) -> Optional[int]:
        bb = self.best_bid()
        return bb.price if bb else None

    def best_ask_price(self) -> Optional[int]:
        ba = self.best_ask()
        return ba.price if ba else None

    def owner_orders(self, owner: int) -> List[Order]:
        return [o for o in (self._bids + self._asks) if o.owner == owner]

    def get_order(self, order_id: int) -> Optional[Order]:
        for o in self._bids + self._asks:
            if o.order_id == order_id:
                return o
        return None

    def snapshot(self) -> dict:
        """Public, JSON-serialisable view of the book."""
        return {
            "bids": [(o.price, o.owner, o.order_id) for o in self.bids],
            "asks": [(o.price, o.owner, o.order_id) for o in self.asks],
            "best_bid": self.best_bid_price(),
            "best_ask": self.best_ask_price(),
        }

    # --------------------------------------------------------------- mutation
    def _new_id(self) -> int:
        oid = self._next_id
        self._next_id += 1
        return oid

    def post_bid(self, owner: int, price: int, ts: int) -> Tuple[Optional[Order], List[Trade]]:
        """Post a buy limit. Returns (resting_order_or_None, trades)."""
        trades: List[Trade] = []
        ba = self.best_ask()
        if ba is not None and ba.owner != owner and price >= ba.price:
            # marketable: lift the best ask at the maker's price
            self._asks.remove(ba)
            trades.append(
                Trade(ts=ts, price=ba.price, buyer=owner, seller=ba.owner,
                      aggressor=owner, maker_order_id=ba.order_id)
            )
            return None, trades
        order = Order(self._new_id(), owner, Side.BID, price, ts)
        self._bids.append(order)
        return order, trades

    def post_ask(self, owner: int, price: int, ts: int) -> Tuple[Optional[Order], List[Trade]]:
        """Post a sell limit. Returns (resting_order_or_None, trades)."""
        trades: List[Trade] = []
        bb = self.best_bid()
        if bb is not None and bb.owner != owner and price <= bb.price:
            # marketable: hit the best bid at the maker's price
            self._bids.remove(bb)
            trades.append(
                Trade(ts=ts, price=bb.price, buyer=bb.owner, seller=owner,
                      aggressor=owner, maker_order_id=bb.order_id)
            )
            return None, trades
        order = Order(self._new_id(), owner, Side.ASK, price, ts)
        self._asks.append(order)
        return order, trades

    def take_bid(self, taker: int, ts: int) -> Optional[Trade]:
        """``taker`` sells 1 unit into the best bid."""
        bb = self.best_bid()
        if bb is None or bb.owner == taker:
            return None
        self._bids.remove(bb)
        return Trade(ts=ts, price=bb.price, buyer=bb.owner, seller=taker,
                     aggressor=taker, maker_order_id=bb.order_id)

    def take_ask(self, taker: int, ts: int) -> Optional[Trade]:
        """``taker`` buys 1 unit from the best ask."""
        ba = self.best_ask()
        if ba is None or ba.owner == taker:
            return None
        self._asks.remove(ba)
        return Trade(ts=ts, price=ba.price, buyer=taker, seller=ba.owner,
                     aggressor=taker, maker_order_id=ba.order_id)

    def cancel(self, order_id: int, owner: int) -> bool:
        """Cancel an order owned by ``owner``. Returns True on success."""
        for bucket in (self._bids, self._asks):
            for o in bucket:
                if o.order_id == order_id and o.owner == owner:
                    bucket.remove(o)
                    return True
        return False
