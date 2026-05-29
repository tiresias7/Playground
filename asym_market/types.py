"""Core value types for the three-player asymmetric-information trading game.

Player indices are fixed and meaningful throughout the codebase:

    0 = S (State player)  -- privately observes ``state in {FIX, RANDOM}``
    1 = X (Number player) -- privately observes ``x in {0..100}``
    2 = B (Buyer)         -- privately observes ``k in {1,2,3}`` (buy obligation)

Nothing in this module encodes a behavioural prior about *who should act*.
These are pure data containers used by the environment, the order book and
the solver.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional


class Player(enum.IntEnum):
    """Fixed player roles. Values double as list indices."""

    S = 0  # State player
    X = 1  # Number player
    B = 2  # Buyer


class State(enum.Enum):
    """The hidden market regime that S observes."""

    FIX = "FIX"
    RANDOM = "RANDOM"


class Side(enum.Enum):
    """Side of a resting limit order."""

    BID = "BID"  # a resting buy order
    ASK = "ASK"  # a resting sell order


class ActionType(enum.Enum):
    """The atomic actions a player may take on their turn."""

    WAIT = "WAIT"
    POST_BID = "POST_BID"  # rest a buy limit at ``price`` (marketable if it crosses)
    POST_ASK = "POST_ASK"  # rest a sell limit at ``price``
    TAKE_BID = "TAKE_BID"  # sell 1 unit into the current best bid
    TAKE_ASK = "TAKE_ASK"  # buy 1 unit from the current best ask
    CANCEL = "CANCEL"      # cancel own outstanding order


@dataclass(frozen=True)
class Action:
    """An action chosen by a player on their turn.

    ``price`` is required for POST_BID / POST_ASK and ignored otherwise.
    ``order_id`` is required for CANCEL and ignored otherwise.
    """

    type: ActionType
    price: Optional[int] = None
    order_id: Optional[int] = None

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        if self.type in (ActionType.POST_BID, ActionType.POST_ASK):
            return f"{self.type.value}({self.price})"
        if self.type is ActionType.CANCEL:
            return f"CANCEL(#{self.order_id})"
        return self.type.value


@dataclass
class Order:
    """A single-unit resting limit order."""

    order_id: int
    owner: int          # Player index
    side: Side
    price: int
    ts: int             # global decision counter when the order was posted


@dataclass
class Trade:
    """A single executed trade (size is always 1)."""

    ts: int
    price: int
    buyer: int          # Player index that gained +1 position
    seller: int         # Player index that gained -1 position
    aggressor: int      # Player index that initiated the execution
    maker_order_id: Optional[int]  # resting order that was hit (None if N/A)
