"""Order-matching correctness tests."""

from asym_market.order_book import OrderBook
from asym_market.types import Side


def test_post_and_best_prices():
    ob = OrderBook()
    ob.post_bid(owner=0, price=40, ts=0)
    ob.post_ask(owner=1, price=60, ts=1)
    assert ob.best_bid_price() == 40
    assert ob.best_ask_price() == 60


def test_marketable_bid_crosses_ask_at_maker_price():
    ob = OrderBook()
    ob.post_ask(owner=1, price=50, ts=0)
    resting, trades = ob.post_bid(owner=2, price=55, ts=1)  # crosses
    assert resting is None
    assert len(trades) == 1
    tr = trades[0]
    assert tr.price == 50            # executes at the resting (maker) price
    assert tr.buyer == 2 and tr.seller == 1
    assert tr.aggressor == 2
    assert ob.best_ask_price() is None  # ask consumed


def test_marketable_ask_crosses_bid_at_maker_price():
    ob = OrderBook()
    ob.post_bid(owner=0, price=70, ts=0)
    resting, trades = ob.post_ask(owner=1, price=65, ts=1)
    assert resting is None
    assert trades[0].price == 70
    assert trades[0].buyer == 0 and trades[0].seller == 1


def test_non_marketable_orders_rest():
    ob = OrderBook()
    r1, t1 = ob.post_bid(owner=0, price=40, ts=0)
    r2, t2 = ob.post_ask(owner=1, price=60, ts=1)
    assert r1 is not None and not t1
    assert r2 is not None and not t2


def test_price_time_priority_on_bids():
    ob = OrderBook()
    ob.post_bid(owner=0, price=50, ts=0)   # earlier
    ob.post_bid(owner=1, price=50, ts=1)   # later, same price
    bb = ob.best_bid()
    assert bb.owner == 0 and bb.ts == 0    # earliest wins
    # an aggressor selling hits the earliest order
    tr = ob.take_bid(taker=2, ts=2)
    assert tr.maker_order_id == bb.order_id
    assert ob.best_bid().owner == 1        # later order now on top


def test_better_price_wins_priority():
    ob = OrderBook()
    ob.post_ask(owner=0, price=60, ts=0)
    ob.post_ask(owner=1, price=55, ts=1)   # better (lower) ask
    assert ob.best_ask().owner == 1


def test_take_ask_and_take_bid():
    ob = OrderBook()
    ob.post_ask(owner=1, price=50, ts=0)
    tr = ob.take_ask(taker=2, ts=1)
    assert tr is not None and tr.buyer == 2 and tr.seller == 1 and tr.price == 50
    assert ob.best_ask_price() is None

    ob.post_bid(owner=0, price=45, ts=2)
    tr2 = ob.take_bid(taker=1, ts=3)
    assert tr2.buyer == 0 and tr2.seller == 1 and tr2.price == 45


def test_no_self_trade_on_post():
    ob = OrderBook()
    ob.post_bid(owner=0, price=70, ts=0)
    # owner 0 posts an ask that would cross its OWN bid -> must rest, not match
    resting, trades = ob.post_ask(owner=0, price=65, ts=1)
    assert resting is not None and not trades
    assert ob.best_bid_price() == 70 and ob.best_ask_price() == 65


def test_no_self_take():
    ob = OrderBook()
    ob.post_ask(owner=2, price=50, ts=0)
    assert ob.take_ask(taker=2, ts=1) is None  # cannot take own ask


def test_cancel():
    ob = OrderBook()
    r, _ = ob.post_bid(owner=0, price=40, ts=0)
    assert ob.cancel(r.order_id, owner=0) is True
    assert ob.best_bid_price() is None
    assert ob.cancel(r.order_id, owner=0) is False  # already gone
