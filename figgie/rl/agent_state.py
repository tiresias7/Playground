"""Per-episode feature/state tracker shared by the gradient-RL agent: maintains
order-flow, last-trade, goal-posterior and marginal-value state, and turns an
Observation into (features, action mask, resolved actions)."""

from __future__ import annotations

from ..cards import ALL_SUITS
from ..inference import goal_size_posterior
from ..valuation import card_values
from . import features as F


class AgentState:
    def __init__(self):
        self.reset()

    def reset(self):
        self._net = {s: 0 for s in ALL_SUITS}
        self._last = {s: None for s in ALL_SUITS}
        self._cursor = 0
        self._cache = {}

    def observe(self, obs):
        me = obs.player
        tr = obs.trades
        for i in range(self._cursor, len(tr)):
            t = tr[i]
            self._last[t.suit] = t.price
            if t.aggressor != me:
                self._net[t.suit] += 1 if t.buyer == t.aggressor else -1
        self._cursor = len(tr)

        key = tuple(obs.hand[s] for s in ALL_SUITS)
        st = self._cache.get(key)
        if st is None:
            per = 40 // obs.num_players
            joint = goal_size_posterior(obs.hand)
            post = {s: sum(joint[s].values()) for s in ALL_SUITS}
            buy, sell = card_values(obs.hand, joint, obs.num_players, per, obs.pot)
            st = (post, buy, sell)
            self._cache[key] = st
        post, buy, sell = st
        return F.encode(obs, self._net, self._last, post, buy, sell)
