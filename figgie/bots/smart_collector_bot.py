"""Improved collector baseline.

CollectorBot wins the most games (it corners the goal's color) but bleeds net by
overpaying. The robust core is collecting a whole color (hand-only goal-color ID
~60% beats single-suit ID ~39%), so this keeps that. Experiments showed the only
changes that helped (and only marginally -- the collector is near its archetype
ceiling) were:

  * picking the color from the hand's goal posterior (`color_by="bayes"`), which
    edges out the raw longest-suit rule (and beats raw total counts);
  * mild price shading via a lower `buy_cap` (8 vs 10) so it stops chasing
    pivotal cards at the ~10-13 peak.

Reducing the more-held suit's target below 6 (the `target_lo`/`target_hi` split)
*hurt*: that suit is often the goal, so under-collecting it loses the majority.
Defaults therefore keep full 6/6 cornering; the split knobs remain available.
"""

from __future__ import annotations

import random

from ..cards import ALL_SUITS, Suit
from ..inference import goal_posterior
from .base import Action, Ask, Bid, Bot, Observation


class SmartCollectorBot(Bot):
    def __init__(self, name: str | None = None, rng: random.Random | None = None,
                 target_lo: int = 6, target_hi: int = 6, buy_cap: int = 8,
                 sell_floor: int = 1, color_by: str = "bayes"):
        super().__init__(name)
        self.rng = rng or random.Random()
        self.target_lo = target_lo
        self.target_hi = target_hi
        self.buy_cap = buy_cap
        self.sell_floor = sell_floor
        self.color_by = color_by
        self._desired: dict | None = None

    def reset(self, observation: Observation | None) -> None:
        self._desired = None

    def _desired_counts(self, hand) -> dict:
        if self._desired is not None:
            return self._desired
        if self.color_by == "longest":
            longest = max(ALL_SUITS, key=lambda s: hand[s])
            color = (longest, longest.partner)
        elif self.color_by == "bayes":
            # Pick the color the hand posterior says is most likely the goal color.
            post = goal_posterior(hand)
            black = post[Suit.SPADES] + post[Suit.CLUBS]
            red = post[Suit.HEARTS] + post[Suit.DIAMONDS]
            color = (Suit.SPADES, Suit.CLUBS) if black >= red else (Suit.HEARTS, Suit.DIAMONDS)
        else:  # by total cards held per color (more robust than longest)
            black = hand[Suit.SPADES] + hand[Suit.CLUBS]
            red = hand[Suit.HEARTS] + hand[Suit.DIAMONDS]
            color = (Suit.SPADES, Suit.CLUBS) if black >= red else (Suit.HEARTS, Suit.DIAMONDS)
        # lo = fewer-held color suit (more likely the goal); hi = more-held (likely common)
        lo, hi = sorted(color, key=lambda s: hand[s])
        d = {s: 0 for s in ALL_SUITS}
        d[lo] = self.target_lo
        d[hi] = self.target_hi
        self._desired = d
        return d

    def act(self, obs: Observation) -> Action:
        desired = self._desired_counts(obs.hand)

        best_sell = None
        for s in ALL_SUITS:
            bid = obs.market[s].bid
            if obs.hand[s] > desired[s] and bid is not None and bid >= self.sell_floor:
                if best_sell is None or bid > best_sell[0]:
                    best_sell = (bid, s)
        best_buy = None
        for s in ALL_SUITS:
            ask = obs.market[s].ask
            if (obs.hand[s] < desired[s] and ask is not None
                    and ask <= self.buy_cap and ask <= obs.cash):
                if best_buy is None or ask < best_buy[0]:
                    best_buy = (ask, s)

        if best_sell is not None:
            return Ask(best_sell[1], best_sell[0])
        if best_buy is not None:
            return Bid(best_buy[1], best_buy[0])

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
