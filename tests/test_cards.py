import random

from figgie.cards import ALL_SUITS, Deck, Suit


def test_partner_colors():
    assert Suit.SPADES.partner is Suit.CLUBS
    assert Suit.CLUBS.partner is Suit.SPADES
    assert Suit.HEARTS.partner is Suit.DIAMONDS
    assert Suit.DIAMONDS.partner is Suit.HEARTS
    assert Suit.SPADES.is_black and Suit.CLUBS.is_black
    assert not Suit.HEARTS.is_black and not Suit.DIAMONDS.is_black


def test_random_deck_invariants():
    rng = random.Random(0)
    for _ in range(2000):
        d = Deck.random(rng)
        assert sum(d.counts.values()) == 40
        assert sorted(d.counts.values()) == [8, 10, 10, 12]
        # common suit has 12
        assert d.counts[d.common_suit] == 12
        # goal is the same-color partner of common, and never equals common
        assert d.goal_suit is d.common_suit.partner
        assert d.goal_suit is not d.common_suit
        assert d.goal_suit.is_black == d.common_suit.is_black
        # goal suit always has 8 or 10
        assert d.counts[d.goal_suit] in (8, 10)


def test_goal_size_distribution():
    rng = random.Random(1)
    sizes = [Deck.random(rng).goal_count for _ in range(6000)]
    frac8 = sizes.count(8) / len(sizes)
    # goal is the 8-card suit with prob 1/3 under uniform config sampling
    assert 0.28 < frac8 < 0.39
