import random

from figgie.bots import AdaptiveCollectorBot
from figgie.cards import Suit


def test_confident_hand_enters_collect_mode():
    bot = AdaptiveCollectorBot(rng=random.Random(0), commit=0.5)
    # very long in spades -> spades likely common, clubs likely goal, black color
    # confidence high -> collect black
    hand = {Suit.SPADES: 7, Suit.CLUBS: 1, Suit.HEARTS: 1, Suit.DIAMONDS: 1}
    bot._decide(hand)
    assert bot._mode == "collect"
    assert bot._desired[Suit.SPADES] == 6 and bot._desired[Suit.CLUBS] == 6
    assert bot._desired[Suit.HEARTS] == 0 and bot._desired[Suit.DIAMONDS] == 0


def test_ambiguous_hand_falls_back_to_value():
    bot = AdaptiveCollectorBot(rng=random.Random(0), commit=0.95)
    # balanced hand -> no confident color -> value mode (commit threshold very high)
    hand = {Suit.SPADES: 3, Suit.CLUBS: 2, Suit.HEARTS: 3, Suit.DIAMONDS: 2}
    bot._decide(hand)
    assert bot._mode == "value"
