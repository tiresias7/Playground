"""The flagship bot: marginal-equity valuation + order-flow goal inference +
surplus-capturing market making.

Over ValueBot it adds two things the head-to-head diagnostics showed were
missing:

  * Order flow. It tracks opponents' aggressive buys/sells (excluding its own)
    and reweights its goal-suit belief, so it locks onto the goal faster than
    hand-only inference allows.
  * Fractional surplus capture. Marginal value is the *most* it would pay, not
    its quote. Its willingness to pay is a fraction `buy_frac` of buy value and
    its reservation to sell is a multiple `sell_frac` of sell value. Because
    goal-card values span ~1 to ~65, a fractional margin scales with the stake
    -- a fixed dollar margin captured almost nothing on the pivotal card. It
    lifts only offers below buy_frac*value, hits only bids above sell_frac*value,
    and rests passive quotes at those same shaded prices, penny-improving the
    book rather than jumping to full value.

The free parameters (buy_frac, sell_frac, flow_lam, flow_cap) are tuned by
self-play in figgie.tune.
"""

from __future__ import annotations

import random

from ..cards import ALL_SUITS
from ..flow import combine_goal_size
from ..inference import goal_size_posterior
from ..valuation import card_values
from .base import Action, Ask, Bid, Bot, Observation


class SharpBot(Bot):
    def __init__(self, name: str | None = None, rng: random.Random | None = None,
                 buy_frac: float = 0.65, sell_frac: float = 1.35,
                 flow_lam: float = 0.45, flow_cap: float = 6.0):
        super().__init__(name)
        self.rng = rng or random.Random()
        self.buy_frac = buy_frac
        self.sell_frac = sell_frac
        self.flow_lam = flow_lam
        self.flow_cap = flow_cap
        self.reset(None)

    def reset(self, observation: Observation | None) -> None:
        self._net = {s: 0 for s in ALL_SUITS}
        self._cursor = 0
        self._hand_cache: dict[tuple, dict] = {}

    def _hand_joint(self, hand):
        key = tuple(hand[s] for s in ALL_SUITS)
        j = self._hand_cache.get(key)
        if j is None:
            j = goal_size_posterior(hand)
            self._hand_cache[key] = j
        return j

    def _update_flow(self, obs: Observation) -> None:
        me = obs.player
        tr = obs.trades
        for i in range(self._cursor, len(tr)):
            t = tr[i]
            if t.aggressor != me:
                self._net[t.suit] += 1 if t.buyer == t.aggressor else -1
        self._cursor = len(tr)

    def act(self, obs: Observation) -> Action:
        self._update_flow(obs)
        joint = combine_goal_size(self._hand_joint(obs.hand), self._net,
                                  self.flow_lam, self.flow_cap)
        per = 40 // obs.num_players
        buy, sell = card_values(obs.hand, joint, obs.num_players, per, obs.pot)

        # Shaded reservation prices: pay at most buy_frac*value, sell for at
        # least sell_frac*value.
        pay = {s: buy[s] * self.buy_frac for s in ALL_SUITS}
        want = {s: sell[s] * self.sell_frac for s in ALL_SUITS}

        # 1) Take any offer/bid that beats our shaded price (most edge first).
        best_edge, best_action = 0.0, None
        for s in ALL_SUITS:
            ask = obs.market[s].ask
            if ask is not None and ask <= obs.cash:
                edge = pay[s] - ask
                if edge > best_edge:
                    best_edge, best_action = edge, Bid(s, ask)
        for s in ALL_SUITS:
            bid = obs.market[s].bid
            if bid is not None and obs.hand[s] >= 1:
                edge = bid - want[s]
                if edge > best_edge:
                    best_edge, best_action = edge, Ask(s, bid)
        if best_action is not None:
            return best_action

        # 2) Rest a passive quote at the shaded price, penny-improving the book.
        if self.rng.random() < 0.5:
            s = max(ALL_SUITS, key=lambda x: pay[x])
            cap = int(pay[s])
            if cap >= 1:
                cur_bid = obs.market[s].bid
                cur_ask = obs.market[s].ask
                price = cap if cur_bid is None else min(cap, cur_bid + 1)
                if (1 <= price <= obs.cash
                        and (cur_ask is None or price < cur_ask)
                        and (cur_bid is None or price > cur_bid)):
                    return Bid(s, price)
        else:
            held = [s for s in ALL_SUITS if obs.hand[s] >= 1]
            if held:
                s = min(held, key=lambda x: want[x])  # offer least-valued holding
                floor_price = int(want[s]) + 1
                cur_bid = obs.market[s].bid
                cur_ask = obs.market[s].ask
                price = floor_price if cur_ask is None else max(floor_price, cur_ask - 1)
                if cur_bid is None or price > cur_bid:
                    return Ask(s, price)
        return None
