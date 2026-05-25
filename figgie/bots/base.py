"""Bot interface plus the observation/action types passed across it."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..cards import Suit


@dataclass(frozen=True)
class Quote:
    """Top-of-book for one suit, as seen by a bot."""

    bid: int | None
    ask: int | None


@dataclass(frozen=True)
class Observation:
    player: int
    num_players: int
    pot: int                   # total pot paid out at settlement
    hand: dict[Suit, int]      # cards this bot currently holds, per suit
    cash: int                  # this bot's current cash
    market: dict[Suit, Quote]  # top of book per suit
    trades: list               # live full trade log (read-only); Trade has an aggressor
    tick: int
    total_ticks: int


# An action is a 4-tuple (kind, suit, price) variants below; we use small
# frozen dataclasses for clarity.
@dataclass(frozen=True)
class Bid:
    suit: Suit
    price: int


@dataclass(frozen=True)
class Ask:
    suit: Suit
    price: int


# A bot returns one of: Bid, Ask, or None (pass).
Action = Bid | Ask | None


class Bot(ABC):
    def __init__(self, name: str | None = None):
        self.name = name or type(self).__name__

    def reset(self, observation: Observation) -> None:
        """Called once at the start of a round with the initial observation."""

    @abstractmethod
    def act(self, observation: Observation) -> Action:
        """Return a single market action, or None to pass this tick."""
