"""A bot driven by a learned numpy policy (figgie.rl.policy). It maintains the
same order-flow / last-trade / goal-posterior state the hand-coded bots use, so
the policy's job is purely *what to do*, not re-deriving inference."""

from __future__ import annotations

import numpy as np

from ..bots.base import Action, Bot, Observation
from ..cards import ALL_SUITS
from ..inference import goal_posterior
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

    def _post(self, hand):
        key = tuple(hand[s] for s in ALL_SUITS)
        p = self._post_cache.get(key)
        if p is None:
            p = goal_posterior(hand)
            self._post_cache[key] = p
        return p

    def act(self, obs: Observation) -> Action:
        self._update(obs)
        post = self._post(obs.hand)
        f, mask, actions = F.encode(obs, self._net, self._last, post)
        logits = self.policy.logits(self.theta, f)
        a = select(logits, mask, rng=self.rng, greedy=self.greedy)
        return actions[a]
