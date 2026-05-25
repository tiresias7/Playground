"""Card, suit, and deck logic for Figgie.

The Figgie deck has 40 cards across 4 suits, with counts that are always a
permutation of {8, 10, 10, 12}. The 12-card suit is the "common" suit; the
"goal" suit is the *other* suit of the same color (spades<->clubs are black,
hearts<->diamonds are red). The goal suit therefore always has 8 or 10 cards.
"""

from __future__ import annotations

import random
from enum import Enum


class Suit(Enum):
    SPADES = "spades"
    CLUBS = "clubs"
    HEARTS = "hearts"
    DIAMONDS = "diamonds"

    @property
    def symbol(self) -> str:
        return {"spades": "♠", "clubs": "♣", "hearts": "♥", "diamonds": "♦"}[self.value]

    @property
    def is_black(self) -> bool:
        return self in (Suit.SPADES, Suit.CLUBS)

    @property
    def partner(self) -> "Suit":
        """The other suit of the same color."""
        return {
            Suit.SPADES: Suit.CLUBS,
            Suit.CLUBS: Suit.SPADES,
            Suit.HEARTS: Suit.DIAMONDS,
            Suit.DIAMONDS: Suit.HEARTS,
        }[self]


ALL_SUITS: tuple[Suit, ...] = tuple(Suit)


class Deck:
    """A dealt Figgie deck: a concrete suit->count assignment plus the hidden
    common/goal suits that determine scoring."""

    def __init__(self, counts: dict[Suit, int], common_suit: Suit):
        if sorted(counts.values()) != [8, 10, 10, 12]:
            raise ValueError(f"Invalid Figgie counts: {counts}")
        if counts[common_suit] != 12:
            raise ValueError("Common suit must be the 12-card suit")
        goal = common_suit.partner
        if counts[goal] not in (8, 10):
            raise ValueError("Goal suit must have 8 or 10 cards")
        self.counts = dict(counts)
        self.common_suit = common_suit
        self.goal_suit = goal

    @property
    def goal_count(self) -> int:
        return self.counts[self.goal_suit]

    @classmethod
    def random(cls, rng: random.Random | None = None) -> "Deck":
        rng = rng or random.Random()
        common = rng.choice(ALL_SUITS)
        others = [s for s in ALL_SUITS if s != common]
        # The three non-common suits get a random permutation of {8, 10, 10}.
        small_counts = [8, 10, 10]
        rng.shuffle(small_counts)
        counts = {common: 12}
        for suit, c in zip(others, small_counts):
            counts[suit] = c
        return cls(counts, common)

    def __repr__(self) -> str:
        c = ", ".join(f"{s.symbol}{self.counts[s]}" for s in ALL_SUITS)
        return f"Deck({c}; common={self.common_suit.symbol}, goal={self.goal_suit.symbol})"
