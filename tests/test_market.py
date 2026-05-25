from figgie.cards import ALL_SUITS, Suit
from figgie.market import Market


class World:
    """Minimal cash/card bookkeeping to drive the Market in isolation."""

    def __init__(self, players):
        self.cash = {p: 100 for p in players}
        self.cards = {p: {s: 0 for s in ALL_SUITS} for p in players}
        self.market = Market(self.owns, self.afford, self.settle)

    def owns(self, p, suit):
        return self.cards[p][suit]

    def afford(self, p, price):
        return self.cash[p] >= price

    def settle(self, t):
        self.cash[t.buyer] -= t.price
        self.cash[t.seller] += t.price
        self.cards[t.buyer][t.suit] += 1
        self.cards[t.seller][t.suit] -= 1


def test_bid_then_ask_crosses():
    w = World([0, 1])
    w.cards[1][Suit.SPADES] = 1
    assert w.market.submit(0, Suit.SPADES, True, 7) is None  # rests as bid
    trade = w.market.submit(1, Suit.SPADES, False, 5)        # crosses at bid (7)
    assert trade is not None
    assert trade.price == 7  # trades at the resting order's price
    assert w.cash[0] == 93 and w.cash[1] == 107
    assert w.cards[0][Suit.SPADES] == 1 and w.cards[1][Suit.SPADES] == 0


def test_ask_then_bid_crosses_at_ask():
    w = World([0, 1])
    w.cards[0][Suit.HEARTS] = 1
    assert w.market.submit(0, Suit.HEARTS, False, 4) is None  # rests as ask
    trade = w.market.submit(1, Suit.HEARTS, True, 9)          # crosses at ask (4)
    assert trade.price == 4
    assert w.cash[1] == 96 and w.cash[0] == 104


def test_cannot_sell_unowned():
    w = World([0, 1])
    assert w.market.submit(0, Suit.CLUBS, False, 3) is None
    assert not w.market.books[Suit.CLUBS].asks  # rejected, not rested


def test_cannot_buy_without_cash():
    w = World([0, 1])
    w.cash[0] = 2
    assert w.market.submit(0, Suit.CLUBS, True, 5) is None
    assert not w.market.books[Suit.CLUBS].bids


def test_no_self_trade():
    w = World([0])
    w.cards[0][Suit.DIAMONDS] = 1
    w.market.submit(0, Suit.DIAMONDS, False, 3)   # own ask rests
    assert w.market.submit(0, Suit.DIAMONDS, True, 9) is None  # would self-cross
    # no trade happened; the bid rests alongside the ask
    assert len(w.market.trades) == 0


def test_stale_ask_cancelled_when_owner_no_longer_holds():
    w = World([0, 1, 2])
    w.cards[0][Suit.SPADES] = 1
    w.market.submit(0, Suit.SPADES, False, 5)   # player 0 offers
    # player 0 sells the card elsewhere first
    w.cards[1][Suit.SPADES] = 0
    w.cash[1] = 100
    # simulate player 0 losing the card by direct manipulation
    w.cards[0][Suit.SPADES] = 0
    # now player 2 tries to buy; the stale ask should be cancelled, no trade
    assert w.market.submit(2, Suit.SPADES, True, 9) is None
    assert len(w.market.trades) == 0


def test_quote_replaces_prior_resting_order():
    w = World([0, 1])
    w.market.submit(0, Suit.SPADES, True, 3)   # bid 3
    w.market.submit(0, Suit.SPADES, True, 5)   # update to bid 5
    bids = w.market.books[Suit.SPADES].bids
    assert len(bids) == 1 and bids[0].price == 5


def test_conservation():
    w = World([0, 1, 2, 3])
    for p in range(4):
        w.cards[p][Suit.SPADES] = 2
    start_cash = sum(w.cash.values())
    start_cards = sum(w.cards[p][Suit.SPADES] for p in range(4))
    import random
    rng = random.Random(0)
    for _ in range(500):
        p = rng.randint(0, 3)
        is_buy = rng.random() < 0.5
        w.market.submit(p, Suit.SPADES, is_buy, rng.randint(1, 12))
    assert sum(w.cash.values()) == start_cash
    assert sum(w.cards[p][Suit.SPADES] for p in range(4)) == start_cards
