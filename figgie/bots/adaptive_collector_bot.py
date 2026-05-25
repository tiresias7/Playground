"""Adaptive collector: collect only when the hand is 'ideal', else play value.

The collector's big losses come from cornering the WRONG color on ambiguous
hands (it buys ~12 worthless cards). So this bot first asks whether the starting
hand gives a confident color read:

    ideal  <=>  max(P(black is goal-color), P(red is goal-color)) >= commit

where those colour probabilities come from the hand's goal posterior. (A value
model and colour-collection conflict -- valuing cards refuses to buy the
worthless common half, which is exactly what makes cornering robust -- so the
two are kept as separate *modes* rather than blended.)

  * ideal  -> COLLECT the confident colour: corner both its suits to `target`,
    sell the other colour, paying up to `buy_cap`. `take_p` controls how often it
    crosses the spread (lifts offers) vs. rests cheaper passive bids.
  * not ideal -> delegate every action to an internal MMBot (the value model).

`commit`, `take_p`, `buy_cap` are the knobs worth tuning.
"""

from __future__ import annotations

import random

from ..cards import ALL_SUITS, Suit
from ..inference import goal_posterior
from .base import Action, Ask, Bid, Bot, Observation
from .mm_bot import MMBot


class AdaptiveCollectorBot(Bot):
    def __init__(self, name: str | None = None, rng: random.Random | None = None,
                 commit: float = 0.62, target: int = 6, buy_cap: int = 8,
                 sell_floor: int = 1, take_p: float = 1.0):
        super().__init__(name)
        self.rng = rng or random.Random()
        self.commit = commit
        self.target = target
        self.buy_cap = buy_cap
        self.sell_floor = sell_floor
        self.take_p = take_p
        self._fallback = MMBot(rng=self.rng)
        self._mode: str | None = None
        self._desired: dict | None = None

    def reset(self, observation: Observation | None) -> None:
        self._mode = None
        self._desired = None
        self._fallback.reset(observation)

    def _decide(self, hand) -> None:
        post = goal_posterior(hand)
        black = post[Suit.SPADES] + post[Suit.CLUBS]
        red = post[Suit.HEARTS] + post[Suit.DIAMONDS]
        if max(black, red) >= self.commit:
            self._mode = "collect"
            color = (Suit.SPADES, Suit.CLUBS) if black >= red else (Suit.HEARTS, Suit.DIAMONDS)
            self._desired = {s: (self.target if s in color else 0) for s in ALL_SUITS}
        else:
            self._mode = "value"

    def act(self, obs: Observation) -> Action:
        if self._mode is None:
            self._decide(obs.hand)
        if self._mode == "value":
            return self._fallback.act(obs)

        desired = self._desired
        # Candidate crossing trades (shed over-target, acquire under-target).
        best_sell = best_buy = None
        for s in ALL_SUITS:
            bid = obs.market[s].bid
            if obs.hand[s] > desired[s] and bid is not None and bid >= self.sell_floor:
                if best_sell is None or bid > best_sell[0]:
                    best_sell = (bid, s)
            ask = obs.market[s].ask
            if (obs.hand[s] < desired[s] and ask is not None
                    and ask <= self.buy_cap and ask <= obs.cash):
                if best_buy is None or ask < best_buy[0]:
                    best_buy = (ask, s)

        # Take aggression: cross only with probability take_p; else rest a quote.
        if (best_sell or best_buy) and self.rng.random() < self.take_p:
            if best_sell is not None and (best_buy is None or
                                          best_sell[0] - self.sell_floor >= self.buy_cap - best_buy[0]):
                return Ask(best_sell[1], best_sell[0])
            if best_buy is not None:
                return Bid(best_buy[1], best_buy[0])

        # Passive: improving quote toward the target.
        if self.rng.random() < 0.5:
            wants = [s for s in ALL_SUITS if obs.hand[s] < desired[s]]
            if wants:
                s = self.rng.choice(wants)
                bb, ba = obs.market[s].bid, obs.market[s].ask
                price = min(self.buy_cap, (bb + 1) if bb is not None else self.buy_cap)
                if 1 <= price <= obs.cash and (ba is None or price < ba) and (bb is None or price > bb):
                    return Bid(s, price)
        else:
            sheds = [s for s in ALL_SUITS if obs.hand[s] > desired[s]]
            if sheds:
                s = self.rng.choice(sheds)
                bb, ba = obs.market[s].bid, obs.market[s].ask
                price = max(self.sell_floor, (ba - 1) if ba is not None else self.sell_floor)
                if bb is None or price > bb:
                    return Ask(s, price)
        return None
