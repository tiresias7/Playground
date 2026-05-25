"""Marginal, inventory-aware valuation of Figgie cards.

A goal-suit card pays a fixed $10 bonus to whoever holds it, plus a share of the
remainder pot -- but the remainder goes entirely to the majority holder, so its
value to *you* depends on how close your holdings are to the majority threshold.
We therefore value the *marginal* card at your current inventory.

In a world where suit s is the goal with G total goal cards and you hold k, the
remaining G-k are split among the other n-1 players. Modeling that split as a
multivariate-hypergeometric draw (the deal-based prior, before any order-flow
information), S(k) is your expected share of the remainder pot. The marginal
value of acquiring the (k+1)-th card is

    10 + R(G) * (S(k+1) - S(k)),     R(G) = pot - 10*G

and the value of the k-th card you already hold (your reservation to sell it) is

    10 + R(G) * (S(k) - S(k-1)).

S is concave, so the buy value sits below the sell value: a natural bid-ask
spread that stops the bot churning or dumping goal cards.
"""

from __future__ import annotations

from functools import lru_cache
from math import comb

from .cards import ALL_SUITS

GOAL_CARD_BONUS = 10


def _compositions(total: int, parts: int, cap: int):
    """Yield length-`parts` tuples of ints in [0, cap] summing to `total`."""
    if parts == 1:
        if 0 <= total <= cap:
            yield (total,)
        return
    for first in range(min(total, cap) + 1):
        for rest in _compositions(total - first, parts - 1, cap):
            yield (first, *rest)


@lru_cache(maxsize=None)
def share_table(n_players: int, per: int, G: int) -> tuple[float, ...]:
    """S[k] = expected share of the remainder pot when holding k of G goal
    cards, with the other G-k spread among the n_players-1 opponents."""
    m = n_players - 1
    N = m * per
    table = []
    for k in range(G + 1):
        K = G - k                       # goal cards held by opponents
        denom = comb(N, K)
        acc = 0.0
        for g in _compositions(K, m, per):
            prob = 1
            for gi in g:
                prob *= comb(per, gi)
            prob /= denom
            top = max(g)
            if k > top:
                share = 1.0
            elif k == top:
                share = 1.0 / (g.count(top) + 1)
            else:
                share = 0.0
            acc += prob * share
        table.append(acc)
    return tuple(table)


def card_values(hand, goal_size_post, n_players: int, per: int, pot: int):
    """Return (buy_value, sell_value) dicts per suit.

    buy_value[s]  = EV of acquiring one more card of suit s (k -> k+1)
    sell_value[s] = EV of a card of suit s you already hold (k -> k-1)
    `goal_size_post[s]` is {G: P(goal == s and size == G)}.
    """
    buy, sell = {}, {}
    for s in ALL_SUITS:
        k = hand[s]
        b = sv = 0.0
        for G, p in goal_size_post[s].items():
            if p <= 0:
                continue
            R = pot - GOAL_CARD_BONUS * G
            S = share_table(n_players, per, G)
            kk = min(k, G)
            b += p * (GOAL_CARD_BONUS + R * (S[min(kk + 1, G)] - S[kk]))
            if k >= 1:
                sv += p * (GOAL_CARD_BONUS + R * (S[kk] - S[min(k - 1, G)]))
        buy[s] = b
        sell[s] = sv if k >= 1 else b
    return buy, sell
