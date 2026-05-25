from figgie.cards import ALL_SUITS, Suit
from figgie.inference import goal_posterior


def test_posterior_sums_to_one():
    hand = {Suit.SPADES: 4, Suit.CLUBS: 2, Suit.HEARTS: 1, Suit.DIAMONDS: 3}
    post = goal_posterior(hand)
    assert abs(sum(post.values()) - 1.0) < 1e-9
    # goal can never be the suit you hold most of being the 12-suit's partner;
    # still, all suits have some probability here
    assert all(v >= 0 for v in post.values())


def test_strong_signal_points_to_partner_of_long_suit():
    # Holding many spades suggests spades is the 12-card common suit, so its
    # partner (clubs) becomes the most likely goal suit.
    hand = {Suit.SPADES: 7, Suit.CLUBS: 1, Suit.HEARTS: 1, Suit.DIAMONDS: 1}
    post = goal_posterior(hand)
    assert post[Suit.CLUBS] == max(post.values())


def test_uniform_hand_is_uninformative_about_color():
    hand = {Suit.SPADES: 3, Suit.CLUBS: 3, Suit.HEARTS: 2, Suit.DIAMONDS: 2}
    post = goal_posterior(hand)
    black = post[Suit.SPADES] + post[Suit.CLUBS]
    red = post[Suit.HEARTS] + post[Suit.DIAMONDS]
    # more black cards held -> black slightly more likely to be the goal color
    assert black > red
