"""A value-trading market maker.

Each round it infers the posterior probability that each suit is the goal suit
from its own hand (see figgie.inference), turns that into a fair value per
suit, then trades for edge: it lifts offers priced below fair, hits bids priced
above fair, and otherwise rests passive quotes to provide liquidity. It leans
toward accumulating the suit it believes is the goal.

This is the "simple baseline" of the strategy roadmap: inference uses the hand
only, not order flow, and pot-majority equity is approximated by a constant
premium on top of the $10 goal-card bonus.
"""

from __future__ import annotations

import random

from ..cards import ALL_SUITS, Suit
from ..inference import goal_posterior
from .base import Action, Ask, Bid, Bot, Observation

GOAL_CARD_BONUS = 10
MAJORITY_PREMIUM = 4   # extra value/card to reflect majority-pot equity
EDGE = 1               # minimum profit (in $) required to cross the spread


class HeuristicBot(Bot):
    def __init__(self, name: str | None = None, rng: random.Random | None = None):
        super().__init__(name)
        self.rng = rng or random.Random()
        self._cache: dict[tuple, dict[Suit, float]] = {}

    def _fair_values(self, hand: dict[Suit, int]) -> dict[Suit, float]:
        # The hand only changes when this bot trades, so cache by hand contents.
        key = tuple(hand[s] for s in ALL_SUITS)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        post = goal_posterior(hand)
        per_goal_card = GOAL_CARD_BONUS + MAJORITY_PREMIUM
        fair = {s: post[s] * per_goal_card for s in ALL_SUITS}
        self._cache[key] = fair
        return fair

    def act(self, obs: Observation) -> Action:
        fair = self._fair_values(obs.hand)
        target = max(ALL_SUITS, key=lambda s: fair[s])  # suit to accumulate

        # 1) Best buy: lift an offer priced below our fair value.
        best_buy, best_buy_edge = None, EDGE - 1
        for s in ALL_SUITS:
            ask = obs.market[s].ask
            if ask is None or ask > obs.cash:
                continue
            edge = fair[s] - ask
            if edge > best_buy_edge:
                best_buy, best_buy_edge = s, edge

        # 2) Best sell: hit a bid priced above our fair value, for a suit we hold.
        best_sell, best_sell_edge = None, EDGE - 1
        for s in ALL_SUITS:
            bid = obs.market[s].bid
            if bid is None or obs.hand[s] < 1:
                continue
            edge = bid - fair[s]
            if edge > best_sell_edge:
                best_sell, best_sell_edge = s, edge

        if best_buy is not None and best_buy_edge >= best_sell_edge:
            return Bid(best_buy, obs.market[best_buy].ask)
        if best_sell is not None:
            return Ask(best_sell, obs.market[best_sell].bid)

        # 3) Provide liquidity: rest a passive quote, randomly bidding for the
        # target suit or offering an over-valued holding.
        if self.rng.random() < 0.5:
            price = max(1, round(fair[target]) - 1)
            if price <= obs.cash and (obs.market[target].ask is None
                                      or price < obs.market[target].ask):
                return Bid(target, price)
        else:
            sellable = [s for s in ALL_SUITS if obs.hand[s] >= 1 and s != target]
            if sellable:
                s = max(sellable, key=lambda x: obs.hand[x])
                price = round(fair[s]) + 2
                if obs.market[s].bid is None or price > obs.market[s].bid:
                    return Ask(s, price)
        return None
