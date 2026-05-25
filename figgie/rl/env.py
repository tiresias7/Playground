"""A single-agent, steppable Figgie environment for gradient RL (PPO).

The learner controls seat 0; the other three seats are bots from a league,
fixed for the episode. One env step = one market tick (seat 0 plays the given
order, the opponents play their own, in randomized order).

Reward is dense via potential-based shaping: the per-step reward is the change
in the learner's mark-to-market value (cash + holdings marked at expected goal
bonus). The terminal step adds the correction so the episode return equals the
true round net -- giving PPO a dense signal while preserving the true objective.
"""

from __future__ import annotations

import random

from ..bots.base import Ask, Bid, Bot
from ..cards import ALL_SUITS
from ..game import FiggieGame
from ..inference import goal_posterior
from ..market import Market


class _Dummy(Bot):
    def act(self, obs):
        return None


class FiggieEnv:
    def __init__(self, opp_factories, total_ticks=250, pot=200, starting_cash=350):
        self.opp_factories = opp_factories  # list of 3 factory(rng) -> Bot
        self.total_ticks = total_ticks
        self.pot = pot
        self.starting_cash = starting_cash

    def reset(self, seed):
        self.rng = random.Random(seed)
        bots = [_Dummy()] + [f(self.rng) for f in self.opp_factories]
        self.game = FiggieGame(bots, pot=self.pot, starting_cash=self.starting_cash,
                               total_ticks=self.total_ticks,
                               rng=random.Random(self.rng.random()))
        g = self.game
        g._deal()
        g._dealt_hands = {p: dict(g.hands[p]) for p in range(g.n)}
        self.start = dict(g.cash)
        self.market = Market(g._owns, g._can_afford, g._settle)
        self.tick = 0
        for p in range(1, g.n):
            g.bots[p].reset(g._observation(p, self.market, 0))
        self._post_cache = {}
        self.V0 = self._phi()
        self.Vprev = self.V0
        return g._observation(0, self.market, 0)

    def _phi(self) -> float:
        g = self.game
        h = g.hands[0]
        key = tuple(h[s] for s in ALL_SUITS)
        post = self._post_cache.get(key)
        if post is None:
            post = goal_posterior(h)
            self._post_cache[key] = post
        mark = sum(h[s] * post[s] * 10 for s in ALL_SUITS)
        return g.cash[0] + mark

    def _apply(self, p, action):
        if isinstance(action, Bid):
            self.market.submit(p, action.suit, True, action.price)
        elif isinstance(action, Ask):
            self.market.submit(p, action.suit, False, action.price)

    def step(self, action):
        """action: the seat-0 resolved order (Bid/Ask/None)."""
        g = self.game
        order = list(range(g.n))
        self.rng.shuffle(order)
        for p in order:
            if p == 0:
                self._apply(0, action)
            else:
                self._apply(p, g.bots[p].act(g._observation(p, self.market, self.tick)))
        self.tick += 1
        done = self.tick >= self.total_ticks
        if not done:
            v = self._phi()
            r = v - self.Vprev
            self.Vprev = v
            return g._observation(0, self.market, self.tick), r, False
        res = g._settle_round(self.start, self.market)
        r = res.net[0] - (self.Vprev - self.V0)
        return g._observation(0, self.market, self.tick), r, True
