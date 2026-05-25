import random

from figgie.bots import CollectorBot
from figgie.bots.base import Observation, Quote
from figgie.cards import ALL_SUITS, Suit


def _obs(hand, market=None, cash=300):
    market = market or {s: Quote(None, None) for s in ALL_SUITS}
    return Observation(player=0, num_players=4, pot=200, hand=hand, cash=cash,
                       market=market, trades=[], tick=0, total_ticks=2000)


def test_picks_longest_and_partner():
    bot = CollectorBot(rng=random.Random(0))
    hand = {Suit.SPADES: 5, Suit.CLUBS: 2, Suit.HEARTS: 2, Suit.DIAMONDS: 1}
    bot._desired(hand)  # triggers target selection
    assert bot._keep == {Suit.SPADES, Suit.CLUBS}  # longest (spades) + partner (clubs)


def test_desired_is_6600():
    bot = CollectorBot(rng=random.Random(0))
    hand = {Suit.HEARTS: 4, Suit.DIAMONDS: 3, Suit.SPADES: 2, Suit.CLUBS: 1}
    desired = bot._desired(hand)
    assert desired[Suit.HEARTS] == 6 and desired[Suit.DIAMONDS] == 6
    assert desired[Suit.SPADES] == 0 and desired[Suit.CLUBS] == 0


def test_sells_dump_suit_into_bid():
    bot = CollectorBot(rng=random.Random(0))
    hand = {Suit.SPADES: 5, Suit.CLUBS: 1, Suit.HEARTS: 3, Suit.DIAMONDS: 1}
    # keep = spades/clubs; hearts is a dump suit it holds. A bid exists for hearts.
    market = {s: Quote(None, None) for s in ALL_SUITS}
    market[Suit.HEARTS] = Quote(bid=5, ask=None)
    action = bot.act(_obs(hand, market))
    from figgie.bots.base import Ask
    assert isinstance(action, Ask) and action.suit is Suit.HEARTS and action.price == 5


def test_buys_target_suit_from_offer():
    bot = CollectorBot(rng=random.Random(0))
    hand = {Suit.SPADES: 5, Suit.CLUBS: 1, Suit.HEARTS: 0, Suit.DIAMONDS: 0}
    # keep = spades/clubs; wants more clubs (1 < 6). Cheap clubs offered.
    market = {s: Quote(None, None) for s in ALL_SUITS}
    market[Suit.CLUBS] = Quote(bid=None, ask=4)
    action = bot.act(_obs(hand, market))
    from figgie.bots.base import Bid
    assert isinstance(action, Bid) and action.suit is Suit.CLUBS and action.price == 4


def test_wont_pay_above_buy_cap():
    bot = CollectorBot(rng=random.Random(0), buy_cap=8)
    hand = {Suit.SPADES: 5, Suit.CLUBS: 1, Suit.HEARTS: 0, Suit.DIAMONDS: 0}
    market = {s: Quote(None, None) for s in ALL_SUITS}
    market[Suit.CLUBS] = Quote(bid=None, ask=12)  # above cap
    # no dump holdings to sell either -> should not lift the expensive offer
    from figgie.bots.base import Bid
    action = bot.act(_obs(hand, market))
    assert not (isinstance(action, Bid) and action.price == 12)
