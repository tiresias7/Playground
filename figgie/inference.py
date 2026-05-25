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


def _posterior_weights(hand: dict[Suit, int]) -> list[tuple[dict[Suit, int], float]]:
    """Normalized posterior weight for each legal config given the hand."""
    weights = []
    for counts in _CONFIGS:
        w = 1.0
        for s in ALL_SUITS:
            if hand[s] > counts[s]:
                w = 0.0  # impossible: holding more than the deck contains
                break
            w *= comb(counts[s], hand[s])
        weights.append(w)
    total = sum(weights)
    if total == 0:
        return [(c, 0.0) for c in _CONFIGS]
    return [(c, w / total) for c, w in zip(_CONFIGS, weights)]


def goal_posterior(hand: dict[Suit, int]) -> dict[Suit, float]:
    """Posterior probability that each suit is the goal suit, given `hand`."""
    post = {s: 0.0 for s in ALL_SUITS}
    for counts, w in _posterior_weights(hand):
        common = max(counts, key=lambda s: counts[s])
        post[common.partner] += w
    return post


def goal_size_posterior(hand: dict[Suit, int]) -> dict[Suit, dict[int, float]]:
    """Joint posterior P(goal == suit and goal size == G), as {suit: {G: prob}}."""
    out: dict[Suit, dict[int, float]] = {s: {} for s in ALL_SUITS}
    for counts, w in _posterior_weights(hand):
        if w == 0.0:
            continue
        common = max(counts, key=lambda s: counts[s])
        goal = common.partner
        G = counts[goal]
        out[goal][G] = out[goal].get(G, 0.0) + w
    return out
