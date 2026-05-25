"""Inventory-target market-making bot: the flagship.

Diagnostics across many fields showed the dominant profit source, whenever weak
traders are present, is *scalping volume* -- buying below and selling above the
going price -- not buying the pot. But naive scalpers (HeuristicBot) sell their
whole goal position and forfeit the pot, while value-holders (ValueBot) hold the
goal and miss the scalp. This bot does both via an inventory target:

  * It wants to be long a `target` of each suit, proportional to that suit's
    probability of being the goal. The likely goal suit gets a positive target
    (kept for the pot); everything else targets ~0.
  * It quotes around a market reference (`ref`, blending fundamental value and
    the observed book/last price), skewed by how far its holdings sit from
    target: under target -> quote up (accumulate); over target -> quote down
    (sell the excess). So it scalps the goal's excess against noise while
    keeping its core position, and offloads junk into high bids.
  * Bids are capped by `buy_frac * fundamental value` and asks floored at
    `sell_floor * value`, so it never overpays or gives away deep value.

Goal belief = hand posterior + order flow. Parameters are tuned by self-play in
figgie.tune.
"""

from __future__ import annotations

import random

from ..cards import ALL_SUITS
from ..flow import combine_goal_size
from ..inference import goal_size_posterior
from ..valuation import card_values
from .base import Action, Ask, Bid, Bot, Observation


class MMBot(Bot):
    def __init__(self, name: str | None = None, rng: random.Random | None = None,
                 buy_frac: float = 0.9, edge: float = 1.5, inv_coef: float = 3.0,
                 target_goal: float = 3.5, ref_alpha: float = 0.4,
                 sell_floor: float = 0.5, flow_lam: float = 0.45, flow_cap: float = 6.0):
        super().__init__(name)
        self.rng = rng or random.Random()
        self.buy_frac = buy_frac
        self.edge = edge
        self.inv_coef = inv_coef
        self.target_goal = target_goal
        self.ref_alpha = ref_alpha
        self.sell_floor = sell_floor
        self.flow_lam = flow_lam
        self.flow_cap = flow_cap
        self.reset(None)

    def reset(self, observation: Observation | None) -> None:
        self._net = {s: 0 for s in ALL_SUITS}
        self._last = {s: None for s in ALL_SUITS}
        self._cursor = 0
        self._hand_cache: dict[tuple, dict] = {}

    def _hand_joint(self, hand):
        key = tuple(hand[s] for s in ALL_SUITS)
        j = self._hand_cache.get(key)
        if j is None:
            j = goal_size_posterior(hand)
            self._hand_cache[key] = j
        return j

    def _update(self, obs: Observation) -> None:
        me = obs.player
        tr = obs.trades
        for i in range(self._cursor, len(tr)):
            t = tr[i]
            self._last[t.suit] = t.price
            if t.aggressor != me:
                self._net[t.suit] += 1 if t.buyer == t.aggressor else -1
        self._cursor = len(tr)

    def _ref(self, s, obs, midval):
        bb, ba = obs.market[s].bid, obs.market[s].ask
        if bb is not None and ba is not None:
            mkt = (bb + ba) / 2
        elif self._last[s] is not None:
            mkt = self._last[s]
        elif bb is not None:
            mkt = bb + self.edge
        elif ba is not None:
            mkt = ba - self.edge
        else:
            return midval
        return self.ref_alpha * midval + (1 - self.ref_alpha) * mkt

    def act(self, obs: Observation) -> Action:
        self._update(obs)
        joint = combine_goal_size(self._hand_joint(obs.hand), self._net,
                                  self.flow_lam, self.flow_cap)
        per = 40 // obs.num_players
        buy, sell = card_values(obs.hand, joint, obs.num_players, per, obs.pot)

        bid_px, ask_px = {}, {}
        for s in ALL_SUITS:
            held = obs.hand[s]
            midval = (buy[s] + sell[s]) / 2 if held >= 1 else buy[s]
            ref = self._ref(s, obs, midval)
            goal_prob = sum(joint[s].values())
            skew = self.inv_coef * (self.target_goal * goal_prob - held)
            cap = buy[s] * self.buy_frac
            floor = sell[s] * self.sell_floor
            bid_px[s] = min(cap, ref - self.edge + skew)
            ask_px[s] = max(floor, ref + self.edge + skew)

        # 1) Take: lift offers below our bid price, hit bids above our ask price.
        best_edge, best_action = 0.0, None
        for s in ALL_SUITS:
            ask = obs.market[s].ask
            if ask is not None and ask <= obs.cash and ask <= bid_px[s]:
                e = bid_px[s] - ask
                if e > best_edge:
                    best_edge, best_action = e, Bid(s, ask)
        for s in ALL_SUITS:
            bid = obs.market[s].bid
            if bid is not None and obs.hand[s] >= 1 and bid >= ask_px[s]:
                e = bid - ask_px[s]
                if e > best_edge:
                    best_edge, best_action = e, Ask(s, bid)
        if best_action is not None:
            return best_action

        # 2) Passive: rest the most attractive quote, penny-improving the book.
        if self.rng.random() < 0.5:
            s = max(ALL_SUITS, key=lambda x: bid_px[x])
            price = int(bid_px[s])
            if price >= 1:
                bb, ba = obs.market[s].bid, obs.market[s].ask
                if bb is not None:
                    price = min(price, bb + 1)
                if 1 <= price <= obs.cash and (ba is None or price < ba) and (bb is None or price > bb):
                    return Bid(s, price)
        else:
            held = [s for s in ALL_SUITS if obs.hand[s] >= 1]
            if held:
                s = min(held, key=lambda x: ask_px[x])
                price = int(ask_px[s]) + 1
                bb, ba = obs.market[s].bid, obs.market[s].ask
                if ba is not None:
                    price = max(price, ba - 1)
                if bb is None or price > bb:
                    return Ask(s, price)
        return None
