"""Continuous double-auction market for Figgie.

One limit order book per suit. Orders are for a single card. Matching uses
price-time priority. Buyers must have cash; sellers must own the card; both
are re-validated at match time, and resting orders that become invalid are
cancelled. Players never trade with themselves.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from .cards import ALL_SUITS, Suit


@dataclass
class Order:
    id: int
    player: int
    suit: Suit
    is_buy: bool
    price: int
    time: int


@dataclass
class Trade:
    suit: Suit
    price: int
    buyer: int
    seller: int
    time: int


class OrderBook:
    """A single-suit book holding resting bids and asks (one card each)."""

    def __init__(self, suit: Suit):
        self.suit = suit
        self.bids: list[Order] = []  # resting buy orders
        self.asks: list[Order] = []  # resting sell orders

    def best_bid(self) -> Order | None:
        return self._best(self.bids, want_high=True)

    def best_ask(self) -> Order | None:
        return self._best(self.asks, want_high=False)

    @staticmethod
    def _best(orders: list[Order], want_high: bool) -> Order | None:
        if not orders:
            return None
        # Price priority, then time priority (earlier wins on equal price).
        return (max if want_high else min)(
            orders, key=lambda o: (o.price if want_high else -o.price, -o.time)
        )

    def remove(self, order: Order) -> None:
        (self.bids if order.is_buy else self.asks).remove(order)


class Market:
    """The four order books plus a trade log and validation hooks."""

    def __init__(self, owns, can_afford, on_trade):
        # owns(player, suit) -> int (cards held)
        # can_afford(player, price) -> bool
        # on_trade(Trade) -> None  (settles cash + cards)
        self.books: dict[Suit, OrderBook] = {s: OrderBook(s) for s in ALL_SUITS}
        self.trades: list[Trade] = []
        self._owns = owns
        self._can_afford = can_afford
        self._on_trade = on_trade
        self._ids = itertools.count()
        self._clock = itertools.count()

    def _now(self) -> int:
        return next(self._clock)

    def submit(self, player: int, suit: Suit, is_buy: bool, price: int) -> Trade | None:
        """Submit a limit order for one card. Crosses immediately if possible,
        otherwise rests in the book. Returns the resulting Trade or None."""
        if price <= 0:
            return None
        book = self.books[suit]
        if is_buy:
            if not self._can_afford(player, price):
                return None
            ask = self._best_valid_ask(book)
            if ask is not None and price >= ask.price and ask.player != player:
                book.remove(ask)
                return self._execute(suit, ask.price, player, ask.player)
            self._rest(book, player, suit, is_buy=True, price=price)
            return None
        else:
            if self._owns(player, suit) < 1:
                return None
            bid = self._best_valid_bid(book)
            if bid is not None and price <= bid.price and bid.player != player:
                book.remove(bid)
                return self._execute(suit, bid.price, bid.player, player)
            self._rest(book, player, suit, is_buy=False, price=price)
            return None

    def _best_valid_ask(self, book: OrderBook) -> Order | None:
        """Best ask whose owner still holds the card; cancel stale ones."""
        while True:
            ask = book.best_ask()
            if ask is None:
                return None
            if self._owns(ask.player, book.suit) < 1:
                book.remove(ask)
                continue
            return ask

    def _best_valid_bid(self, book: OrderBook) -> Order | None:
        while True:
            bid = book.best_bid()
            if bid is None:
                return None
            if not self._can_afford(bid.player, bid.price):
                book.remove(bid)
                continue
            return bid

    def _rest(self, book: OrderBook, player: int, suit: Suit, is_buy: bool, price: int) -> None:
        # A new quote replaces the player's existing resting order on this side,
        # so each player holds at most one bid and one ask per suit. This mirrors
        # a market maker updating its quote and keeps the books bounded.
        side = book.bids if is_buy else book.asks
        side[:] = [o for o in side if o.player != player]
        side.append(Order(next(self._ids), player, suit, is_buy, price, self._now()))

    def _execute(self, suit: Suit, price: int, buyer: int, seller: int) -> Trade:
        trade = Trade(suit, price, buyer, seller, self._now())
        self._on_trade(trade)
        self.trades.append(trade)
        return trade

    def cancel_player_orders(self, player: int) -> None:
        for book in self.books.values():
            book.bids = [o for o in book.bids if o.player != player]
            book.asks = [o for o in book.asks if o.player != player]
