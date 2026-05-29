"""Game configuration.

Everything that defines a concrete instance of the game lives here so that
experiments (Phase 4 ablations) are just different ``GameConfig`` objects.
No strategic assumptions are baked in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


def _grid(step: int, lo: int = 0, hi: int = 100) -> Tuple[int, ...]:
    return tuple(range(lo, hi + 1, step))


@dataclass(frozen=True)
class GameConfig:
    """Defines a single playable game instance.

    Attributes
    ----------
    horizon:
        Number of *rounds* ``T``. In each round all three players act exactly
        once, in a freshly drawn random order.
    price_grid:
        Tick grid the order book snaps to (also the default posting grid).
    post_price_grid:
        Prices a player may *post* at. Defaults to ``price_grid`` but can be
        made coarser to keep best-response search tractable.
    x_values:
        Support of the number player's private value ``x``.
    fix_value:
        Payoff ``V`` when the regime is FIX (here 30).
    k_values:
        Support of the buyer's private obligation ``k``.
    penalty_per_unit:
        Penalty charged to B for each unit short of ``k`` at the end.
    p_fix:
        Prior probability the regime is FIX.
    max_outstanding_per_player:
        Cap on simultaneously resting orders per player (Phase 1 uses 1).
    allow_short:
        If False, a player cannot sell/POST_ASK below zero inventory and
        cannot have net-negative position. If True (default) positions are
        unconstrained -- the informed players are genuine two-sided traders.
    buyer_obligation:
        If False, B has no minimum-buy requirement and no penalty (Ablation A).
    """

    horizon: int = 5
    price_grid: Tuple[int, ...] = field(default_factory=lambda: _grid(5))
    post_price_grid: Tuple[int, ...] = field(default_factory=lambda: _grid(5))
    x_values: Tuple[int, ...] = field(default_factory=lambda: _grid(5))
    fix_value: int = 30
    k_values: Tuple[int, ...] = (1, 2, 3)
    penalty_per_unit: float = 50.0
    p_fix: float = 0.5
    max_outstanding_per_player: int = 1
    allow_short: bool = True
    buyer_obligation: bool = True

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if not self.x_values:
            raise ValueError("x_values must be non-empty")
        if not (0.0 <= self.p_fix <= 1.0):
            raise ValueError("p_fix must be in [0, 1]")
        # post grid must be a subset of the tick grid
        if not set(self.post_price_grid).issubset(set(self.price_grid)):
            raise ValueError("post_price_grid must be a subset of price_grid")

    @property
    def ev_random(self) -> float:
        """Prior expected value of ``x`` under RANDOM."""
        return sum(self.x_values) / len(self.x_values)

    @property
    def ev_value(self) -> float:
        """Unconditional prior E[V] before any private info is revealed."""
        return self.p_fix * self.fix_value + (1.0 - self.p_fix) * self.ev_random
