import random

from figgie.bots import HeuristicBot, RandomBot
from figgie.cards import ALL_SUITS
from figgie.game import FiggieGame


def make_game(seed=0, total_ticks=300):
    rng = random.Random(seed)
    bots = [
        HeuristicBot("h0", random.Random(rng.random())),
        HeuristicBot("h1", random.Random(rng.random())),
        HeuristicBot("h2", random.Random(rng.random())),
        RandomBot("r0", rng=random.Random(rng.random())),
    ]
    return FiggieGame(bots, total_ticks=total_ticks, rng=rng)


def test_deal_counts():
    g = make_game()
    g._deal()
    # each player gets 10 cards; suit totals match the deck
    for p in range(4):
        assert sum(g.hands[p].values()) == 10
    for s in ALL_SUITS:
        assert sum(g.hands[p][s] for p in range(4)) == g.deck.counts[s]


def test_money_conserved_across_round():
    g = make_game(seed=3)
    result = g.play()
    # total final cash == total starting cash (antes are returned via the pot,
    # which is fully paid out)
    total_start = sum(result.starting_cash.values())
    total_final = sum(result.final_cash.values())
    assert total_final == total_start


def test_cards_conserved_across_round():
    g = make_game(seed=5)
    result = g.play()
    for s in ALL_SUITS:
        assert sum(result.final_hands[p][s] for p in range(4)) == g.deck.counts[s]


def test_pot_fully_paid_out():
    g = make_game(seed=7)
    result = g.play()
    assert sum(result.payouts.values()) == g.pot


def test_majority_winner_gets_remainder():
    g = make_game(seed=11)
    result = g.play()
    goal = g.deck.goal_suit
    holdings = {p: result.final_hands[p][goal] for p in range(4)}
    most = max(holdings.values())
    winners = [p for p in range(4) if holdings[p] == most]
    remainder = g.pot - g.deck.goal_count * 10
    share = remainder // len(winners)
    for p in winners:
        # winner payout = own goal cards * 10 + (at least) the floor share
        assert result.payouts[p] >= holdings[p] * 10 + share


def test_trading_happens():
    # over a full-length round, value traders should transact
    rng = random.Random(0)
    bots = [HeuristicBot(f"h{i}", random.Random(rng.random())) for i in range(4)]
    g = FiggieGame(bots, total_ticks=2000, rng=rng)
    result = g.play()
    assert result.num_trades > 0
