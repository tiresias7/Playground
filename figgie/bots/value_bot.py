"""A market maker that prices cards by marginal majority-pot equity.

It infers the joint posterior over (goal suit, goal size) from its hand, then
values the marginal card on each side via figgie.valuation: the next card to
acquire (buy value) and the card it already holds (sell value). Because those
marginal values bracket a natural spread, it lifts offers below its buy value,
hits bids above its sell value, and otherwise rests passive quotes around them.

Still hand-only inference (no order flow): opponents' goal holdings are modeled
by the deal-based prior. The improvement over HeuristicBot is purely in how a
card is valued -- inventory-aware marginal equity instead of a flat constant.
"""

from __future__ import annotations

import random

from ..cards import ALL_SUITS, Suit
from ..inference import goal_size_posterior
from ..valuation import card_values
from .base import Action, Ask, Bid, Bot, Observation


class ValueBot(Bot):
    def __init__(self, name: str | None = None, rng: random.Random | None = None):
        super().__init__(name)
        self.rng = rng or random.Random()
        self._cache: dict[tuple, tuple[dict, dict]] = {}

    def _values(self, obs: Observation):
        key = (obs.pot, obs.num_players) + tuple(obs.hand[s] for s in ALL_SUITS)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        per = 40 // obs.num_players
        post = goal_size_posterior(obs.hand)
        vals = card_values(obs.hand, post, obs.num_players, per, obs.pot)
        self._cache[key] = vals
        return vals

    def act(self, obs: Observation) -> Action:
        buy, sell = self._values(obs)

        # 1) Take the most profitable crossing trade, if any.
        best_edge, best_action = 0.0, None
        for s in ALL_SUITS:
            ask = obs.market[s].ask
            if ask is not None and ask <= obs.cash:
                edge = buy[s] - ask
                if edge > best_edge:
                    best_edge, best_action = edge, Bid(s, ask)
        for s in ALL_SUITS:
            bid = obs.market[s].bid
            if bid is not None and obs.hand[s] >= 1:
                edge = bid - sell[s]
                if edge > best_edge:
                    best_edge, best_action = edge, Ask(s, bid)
        if best_action is not None:
            return best_action

        # 2) Provide liquidity: bid the suit we most want, or offer a holding.
        if self.rng.random() < 0.5:
            s = max(ALL_SUITS, key=lambda x: buy[x])
            price = int(buy[s])  # floor: never bid above marginal value
            cur_ask = obs.market[s].ask
            if price >= 1 and price <= obs.cash and (cur_ask is None or price < cur_ask):
                return Bid(s, price)
        else:
            held = [s for s in ALL_SUITS if obs.hand[s] >= 1]
            if held:
                s = max(held, key=lambda x: obs.hand[x])
                price = int(sell[s]) + 1  # ceil-ish: never ask below marginal value
                cur_bid = obs.market[s].bid
                if cur_bid is None or price > cur_bid:
                    return Ask(s, price)
        return None
