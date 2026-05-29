"""Monte-Carlo evaluation of a strategy profile.

``evaluate_policy`` returns far more than a mean payoff: it keeps the episode
logs so that Phase-2 behavioural diagnostics (who posts first, aggression vs
time, posting probability conditional on private info, ...) can be computed
from exactly the same runs.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import GameConfig
from .env import EpisodeLog, GameEnv
from .policies import Policy, make_act_fn


@dataclass
class EvalResult:
    n: int
    mean_pnl: Dict[int, float]
    stderr_pnl: Dict[int, float]
    logs: List[EpisodeLog] = field(default_factory=list)

    def summary(self) -> str:
        names = {0: "S", 1: "X", 2: "B"}
        parts = [
            f"{names[p]}={self.mean_pnl[p]:+.2f}±{self.stderr_pnl[p]:.2f}"
            for p in (0, 1, 2)
        ]
        return f"[n={self.n}] " + "  ".join(parts)


def evaluate_policy(
    profile: Dict[int, Policy],
    config: GameConfig,
    num_simulations: int = 2000,
    seed: int = 0,
    keep_logs: bool = True,
) -> EvalResult:
    """Estimate mean PnL per player under ``profile`` over ``num_simulations``.

    Uses a single seeded RNG so results are fully reproducible.
    """
    env = GameEnv(config)
    rng = random.Random(seed)
    act_fn = make_act_fn(profile, rng)

    sums = {p: 0.0 for p in (0, 1, 2)}
    sumsq = {p: 0.0 for p in (0, 1, 2)}
    logs: List[EpisodeLog] = []

    for _ in range(num_simulations):
        log = env.play_episode(act_fn, rng)
        for p in (0, 1, 2):
            v = log.pnl[p]
            sums[p] += v
            sumsq[p] += v * v
        if keep_logs:
            logs.append(log)

    n = num_simulations
    mean = {p: sums[p] / n for p in (0, 1, 2)}
    stderr = {}
    for p in (0, 1, 2):
        var = max(0.0, sumsq[p] / n - mean[p] ** 2)
        stderr[p] = math.sqrt(var / n)
    return EvalResult(n=n, mean_pnl=mean, stderr_pnl=stderr, logs=logs)
