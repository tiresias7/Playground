from figgie.cards import ALL_SUITS, Suit
from figgie.inference import goal_size_posterior
from figgie.valuation import card_values, share_table


def test_share_table_basics():
    for G in (8, 10):
        S = share_table(4, 10, G)
        assert len(S) == G + 1
        assert S[0] == 0.0                    # holding none -> never the majority
        assert abs(S[G] - 1.0) < 1e-9         # holding all goal cards -> sure win
        # share is non-decreasing in your own holdings
        diffs = [S[k + 1] - S[k] for k in range(G)]
        assert all(d >= -1e-12 for d in diffs)


def test_marginal_peaks_near_majority_threshold():
    # The most valuable card is the pivotal one near the majority threshold,
    # not the first or the last -> the marginal-share curve is hump-shaped.
    for G in (8, 10):
        S = share_table(4, 10, G)
        diffs = [S[k + 1] - S[k] for k in range(G)]
        peak = diffs.index(max(diffs))
        assert 2 <= peak <= G // 2 + 1


def test_holding_majority_share_reasonable():
    # With 10 goal cards over 4 players (~2.5 each), holding 5 should very
    # likely be the majority.
    S = share_table(4, 10, 10)
    assert S[5] > 0.8


def test_goal_belief_drives_value():
    # Long in spades -> spades likely the 12-card common suit -> clubs the goal.
    hand = {Suit.SPADES: 6, Suit.CLUBS: 2, Suit.HEARTS: 1, Suit.DIAMONDS: 1}
    post = goal_size_posterior(hand)
    buy, _ = card_values(hand, post, 4, 10, 200)
    assert buy[Suit.CLUBS] == max(buy.values())
    # a confidently-valued goal card should be worth well above the $10 bonus
    assert buy[Suit.CLUBS] > 10


def test_joint_posterior_marginalizes_to_goal_posterior():
    from figgie.inference import goal_posterior
    hand = {Suit.SPADES: 4, Suit.CLUBS: 2, Suit.HEARTS: 1, Suit.DIAMONDS: 3}
    joint = goal_size_posterior(hand)
    flat = goal_posterior(hand)
    for s in ALL_SUITS:
        assert abs(sum(joint[s].values()) - flat[s]) < 1e-9
