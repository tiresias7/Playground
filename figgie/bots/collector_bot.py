"""Baseline: collect one color to a 6/6/0/0 target hand.

A simple, common human heuristic. From the starting hand it picks two suits to
keep -- the longest suit (most likely the 12-card common suit) and that suit's
same-color partner (the goal suit) -- and tries to reach `target` (6) of each
while selling the other two suits down to zero. It corners the goal's color
without needing to know which of the pair is actually the goal.

Trading is deliberately crude (it has no valuation): it lifts offers up to
`buy_cap` for suits it still wants and hits any bid at/above `sell_floor` for
suits it wants to shed, and otherwise rests improving quotes toward its target.
"""

from __future__ import annotations

import random

from ..cards import ALL_SUITS
from .base import Action, Ask, Bid, Bot, Observation


class CollectorBot(Bot):
    def __init__(self, name: str | None = None, rng: random.Random | None = None,
                 target: int = 6, buy_cap: int = 10, sell_floor: int = 1):
        super().__init__(name)
        self.rng = rng or random.Random()
        self.target = target
        self.buy_cap = buy_cap
        self.sell_floor = sell_floor
        self._keep: set | None = None

    def reset(self, observation: Observation | None) -> None:
        self._keep = None

    def _desired(self, hand) -> dict:
        if self._keep is None:
            # Longest suit (ties broken by suit order) plus its same-color partner.
            longest = max(ALL_SUITS, key=lambda s: hand[s])
            self._keep = {longest, longest.partner}
        return {s: (self.target if s in self._keep else 0) for s in ALL_SUITS}

    def act(self, obs: Observation) -> Action:
        desired = self._desired(obs.hand)

        # 1) Shed an over-target suit into the highest acceptable bid.
        best_sell = None
        for s in ALL_SUITS:
            bid = obs.market[s].bid
            if obs.hand[s] > desired[s] and bid is not None and bid >= self.sell_floor:
                if best_sell is None or bid > best_sell[0]:
                    best_sell = (bid, s)

        # 2) Acquire an under-target suit from the cheapest acceptable offer.
        best_buy = None
        for s in ALL_SUITS:
            ask = obs.market[s].ask
            if (obs.hand[s] < desired[s] and ask is not None
                    and ask <= self.buy_cap and ask <= obs.cash):
                if best_buy is None or ask < best_buy[0]:
                    best_buy = (ask, s)

        # Prefer whichever take is more urgent: shed first (worthless inventory),
        # then acquire.
        if best_sell is not None:
            return Ask(best_sell[1], best_sell[0])
        if best_buy is not None:
            return Bid(best_buy[1], best_buy[0])

        # 3) Rest an improving quote toward the target.
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
