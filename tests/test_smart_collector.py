import random

from figgie.bots import SmartCollectorBot
from figgie.bots.base import Observation, Quote
from figgie.cards import ALL_SUITS, Suit


def _obs(hand, market=None, cash=300):
    market = market or {s: Quote(None, None) for s in ALL_SUITS}
    return Observation(player=0, num_players=4, pot=200, hand=hand, cash=cash,
                       market=market, trades=[], tick=0, total_ticks=2000)


def test_color_by_total_and_lo_hi_targets():
    bot = SmartCollectorBot(rng=random.Random(0), target_lo=6, target_hi=3)
    # more red cards -> keep hearts/diamonds; hearts(4) is more-held=hi, diamonds(2) fewer=lo
    hand = {Suit.SPADES: 2, Suit.CLUBS: 2, Suit.HEARTS: 4, Suit.DIAMONDS: 2}
    d = bot._desired_counts(hand)
    assert d[Suit.DIAMONDS] == 6  # fewer-held color suit (goal guess) -> target_lo
    assert d[Suit.HEARTS] == 3    # more-held color suit (common guess) -> target_hi
    assert d[Suit.SPADES] == 0 and d[Suit.CLUBS] == 0


def test_concentrates_on_fewer_held_color_suit():
    bot = SmartCollectorBot(rng=random.Random(0), target_lo=6, target_hi=2, buy_cap=8)
    hand = {Suit.SPADES: 5, Suit.CLUBS: 1, Suit.HEARTS: 1, Suit.DIAMONDS: 0}
    # black is the majority color; clubs(1) fewer-held -> goal guess target 6
    market = {s: Quote(None, None) for s in ALL_SUITS}
    market[Suit.CLUBS] = Quote(bid=None, ask=4)
    from figgie.bots.base import Bid
    action = bot.act(_obs(hand, market))
    assert isinstance(action, Bid) and action.suit is Suit.CLUBS


def test_respects_buy_cap():
    bot = SmartCollectorBot(rng=random.Random(0), buy_cap=7)
    hand = {Suit.SPADES: 5, Suit.CLUBS: 1, Suit.HEARTS: 0, Suit.DIAMONDS: 0}
    market = {s: Quote(None, None) for s in ALL_SUITS}
    market[Suit.CLUBS] = Quote(bid=None, ask=11)  # above cap
    from figgie.bots.base import Bid
    action = bot.act(_obs(hand, market))
    assert not (isinstance(action, Bid) and action.price == 11)
