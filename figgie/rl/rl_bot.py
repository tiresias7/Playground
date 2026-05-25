"""A bot driven by a learned numpy policy (figgie.rl.policy). It maintains the
same order-flow / last-trade / goal-posterior state the hand-coded bots use, so
the policy's job is purely *what to do*, not re-deriving inference."""

from __future__ import annotations

import numpy as np

from ..bots.base import Action, Bot, Observation
from ..cards import ALL_SUITS
from ..inference import goal_size_posterior
from ..valuation import card_values
from . import features as F
from .policy import Policy, select


class RLBot(Bot):
    def __init__(self, policy: Policy, theta, name: str | None = None,
                 greedy: bool = True, seed: int = 0):
        super().__init__(name)
        self.policy = policy
        self.theta = np.asarray(theta, dtype=np.float32)
        self.greedy = greedy
        self.rng = np.random.default_rng(seed)
        self.reset(None)

    def reset(self, observation: Observation | None) -> None:
        self._net = {s: 0 for s in ALL_SUITS}
        self._last = {s: None for s in ALL_SUITS}
        self._post_cache: dict[tuple, dict] = {}
        self._cursor = 0

    def _update(self, obs: Observation) -> None:
        me = obs.player
        tr = obs.trades
        for i in range(self._cursor, len(tr)):
            t = tr[i]
            self._last[t.suit] = t.price
            if t.aggressor != me:
                self._net[t.suit] += 1 if t.buyer == t.aggressor else -1
        self._cursor = len(tr)

    def _state(self, obs):
        # Cache (goal posterior, marginal buy/sell values) by hand contents.
        hand = obs.hand
        key = tuple(hand[s] for s in ALL_SUITS)
        st = self._post_cache.get(key)
        if st is None:
            per = 40 // obs.num_players
            joint = goal_size_posterior(hand)
            post = {s: sum(joint[s].values()) for s in ALL_SUITS}
            buy, sell = card_values(hand, joint, obs.num_players, per, obs.pot)
            st = (post, buy, sell)
            self._post_cache[key] = st
        return st

    def act(self, obs: Observation) -> Action:
        self._update(obs)
        post, buy, sell = self._state(obs)
        f, mask, actions = F.encode(obs, self._net, self._last, post, buy, sell)
        logits = self.policy.logits(self.theta, f)
        a = select(logits, mask, rng=self.rng, greedy=self.greedy)
        return actions[a]
