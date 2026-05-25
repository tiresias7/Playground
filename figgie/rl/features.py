"""Observation -> feature vector, and action id -> market order, for the RL bot.

Features (26): per-suit hand counts, cash, per-suit best bid/ask, per-suit last
trade price, per-suit net aggressive order flow, per-suit goal posterior, and
time fraction. Prices are normalized; missing quotes are encoded as -1.

Actions (33): pass, plus for each of the 4 suits {take-bid, take-ask} and
{post-bid, post-ask} at three price levels (passive / mid / aggressive) relative
to a per-suit market reference. Invalid actions (no quote to take, can't afford,
don't own the card) are masked out.
"""

from __future__ import annotations

import numpy as np

from ..bots.base import Ask, Bid
from ..cards import ALL_SUITS

SUITS = list(ALL_SUITS)
# hand(4) cash(1) bid/ask(8) last(4) flow(4) goal-post(4) buy/sell value(8) time(1)
N_FEATURES = 4 + 1 + 8 + 4 + 4 + 4 + 8 + 1
VALUE_NORM = 20.0
PER_SUIT_ACTIONS = 8  # take_buy, take_sell, 3 bid levels, 3 ask levels
N_ACTIONS = 1 + PER_SUIT_ACTIONS * 4
BID_DELTAS = (-3, -1, 1)   # passive, mid, aggressive
ASK_DELTAS = (3, 1, -1)    # passive, mid, aggressive
PRICE_NORM = 15.0
CASH_NORM = 200.0


def _ref(obs, last, s) -> float:
    bb, ba = obs.market[s].bid, obs.market[s].ask
    if bb is not None and ba is not None:
        return (bb + ba) / 2
    if last[s] is not None:
        return float(last[s])
    if bb is not None:
        return bb + 2
    if ba is not None:
        return ba - 2
    return 6.0


def encode(obs, net, last, post, buy, sell):
    """Return (features: np.float32[N_FEATURES], mask: bool[N_ACTIONS],
    actions: list[N_ACTIONS] of Bid/Ask/None). `buy`/`sell` are the marginal
    card values per suit (figgie.valuation)."""
    per = 40 // obs.num_players
    f = np.empty(N_FEATURES, dtype=np.float32)
    k = 0
    for s in SUITS:
        f[k] = obs.hand[s] / per; k += 1
    f[k] = obs.cash / CASH_NORM; k += 1
    for s in SUITS:
        bb, ba = obs.market[s].bid, obs.market[s].ask
        f[k] = (bb / PRICE_NORM) if bb is not None else -1.0; k += 1
        f[k] = (ba / PRICE_NORM) if ba is not None else -1.0; k += 1
    for s in SUITS:
        f[k] = (last[s] / PRICE_NORM) if last[s] is not None else -1.0; k += 1
    for s in SUITS:
        f[k] = max(-8, min(8, net[s])) / 8.0; k += 1
    for s in SUITS:
        f[k] = post[s]; k += 1
    for s in SUITS:
        f[k] = buy[s] / VALUE_NORM; k += 1
        f[k] = sell[s] / VALUE_NORM; k += 1
    f[k] = obs.tick / max(1, obs.total_ticks); k += 1

    actions = [None] * N_ACTIONS
    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[0] = True  # pass is always valid
    for i, s in enumerate(SUITS):
        base = 1 + i * PER_SUIT_ACTIONS
        bb, ba = obs.market[s].bid, obs.market[s].ask
        ref = _ref(obs, last, s)
        if ba is not None and ba <= obs.cash:
            actions[base + 0] = Bid(s, ba); mask[base + 0] = True
        if bb is not None and obs.hand[s] >= 1:
            actions[base + 1] = Ask(s, bb); mask[base + 1] = True
        for j, d in enumerate(BID_DELTAS):
            p = int(round(ref + d))
            if 1 <= p <= obs.cash:
                actions[base + 2 + j] = Bid(s, p); mask[base + 2 + j] = True
        for j, d in enumerate(ASK_DELTAS):
            p = int(round(ref + d))
            if p >= 1 and obs.hand[s] >= 1:
                actions[base + 5 + j] = Ask(s, p); mask[base + 5 + j] = True
    return f, mask, actions
