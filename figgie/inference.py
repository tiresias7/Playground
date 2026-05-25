"""Bayesian inference of the goal suit from a player's own hand.

Every legal Figgie deck is one of 12 equally likely configurations: 4 choices
of common (12-card) suit times 3 choices of which of the remaining suits holds
8 cards (the other two hold 10). The goal suit is fixed once the common suit is
chosen (its same-color partner).

Given the cards a player was dealt, the posterior over configurations is
proportional to the multivariate-hypergeometric likelihood of that hand, which
reduces to the product of C(count_suit, hand_suit) across suits (the C(40, k)
denominator is constant across configs and cancels).
"""

from __future__ import annotations

from math import comb

from .cards import ALL_SUITS, Suit


def _legal_configs() -> list[dict[Suit, int]]:
    configs = []
    for common in ALL_SUITS:
        others = [s for s in ALL_SUITS if s != common]
        for eight in others:  # which non-common suit holds 8 cards
            counts = {common: 12}
            for s in others:
                counts[s] = 8 if s == eight else 10
            configs.append(counts)
    return configs


_CONFIGS = _legal_configs()


def goal_posterior(hand: dict[Suit, int]) -> dict[Suit, float]:
    """Posterior probability that each suit is the goal suit, given `hand`."""
    weights = []
    for counts in _CONFIGS:
        w = 1.0
        ok = True
        for s in ALL_SUITS:
            if hand[s] > counts[s]:
                ok = False  # impossible: holding more than the deck contains
                break
            w *= comb(counts[s], hand[s])
        weights.append(w if ok else 0.0)

    total = sum(weights)
    post = {s: 0.0 for s in ALL_SUITS}
    if total == 0:
        return post
    for counts, w in zip(_CONFIGS, weights):
        common = max(counts, key=lambda s: counts[s])
        post[common.partner] += w / total
    return post
