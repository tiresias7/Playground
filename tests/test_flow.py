from figgie.cards import ALL_SUITS, Suit
from figgie.flow import combine_goal_size
from figgie.inference import goal_size_posterior


def _flat_goal(joint):
    return {s: sum(joint[s].values()) for s in ALL_SUITS}


def test_no_flow_is_identity():
    hand = {Suit.SPADES: 3, Suit.CLUBS: 2, Suit.HEARTS: 3, Suit.DIAMONDS: 2}
    j = goal_size_posterior(hand)
    net = {s: 0 for s in ALL_SUITS}
    out = combine_goal_size(j, net, lam=0.5, cap=6)
    for s in ALL_SUITS:
        assert abs(sum(out[s].values()) - sum(j[s].values())) < 1e-9


def test_buying_pressure_raises_goal_belief():
    hand = {Suit.SPADES: 3, Suit.CLUBS: 2, Suit.HEARTS: 3, Suit.DIAMONDS: 2}
    j = goal_size_posterior(hand)
    base = _flat_goal(j)
    # heavy aggressive buying of clubs, selling of its partner (spades)
    net = {Suit.CLUBS: 5, Suit.SPADES: -5, Suit.HEARTS: 0, Suit.DIAMONDS: 0}
    out = _flat_goal(combine_goal_size(j, net, lam=0.4, cap=6))
    assert out[Suit.CLUBS] > base[Suit.CLUBS]
    assert out[Suit.SPADES] < base[Suit.SPADES]


def test_posterior_still_normalized():
    hand = {Suit.SPADES: 4, Suit.CLUBS: 2, Suit.HEARTS: 1, Suit.DIAMONDS: 3}
    j = goal_size_posterior(hand)
    net = {Suit.HEARTS: 3, Suit.DIAMONDS: -2, Suit.SPADES: 1, Suit.CLUBS: 0}
    out = combine_goal_size(j, net, lam=0.4, cap=6)
    total = sum(sum(out[s].values()) for s in ALL_SUITS)
    assert abs(total - 1.0) < 1e-9
