"""A noise trader: posts random bids/asks. Useful as a liquidity source and as
a baseline opponent for testing the engine."""

from __future__ import annotations

import random

from ..cards import ALL_SUITS
from .base import Action, Ask, Bid, Bot, Observation


class RandomBot(Bot):
    def __init__(self, name: str | None = None, activity: float = 0.3,
                 rng: random.Random | None = None):
        super().__init__(name)
        self.activity = activity
        self.rng = rng or random.Random()

    def act(self, obs: Observation) -> Action:
        if self.rng.random() > self.activity:
            return None
        suit = self.rng.choice(ALL_SUITS)
        if self.rng.random() < 0.5:
            price = self.rng.randint(1, 9)
            if obs.cash >= price:
                return Bid(suit, price)
            return None
        if obs.hand[suit] >= 1:
            return Ask(suit, self.rng.randint(2, 12))
        return None
