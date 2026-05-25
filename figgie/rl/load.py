"""Load a saved policy (.npy of flat weights) into an RLBot, inferring the
hidden-layer size from the parameter count."""

from __future__ import annotations

import random

import numpy as np

from .features import N_ACTIONS, N_FEATURES
from .policy import Policy
from .rl_bot import RLBot


def load(path):
    theta = np.load(path).astype(np.float32)
    d, ni, no = len(theta), N_FEATURES, N_ACTIONS
    hidden = 0 if d == no * (ni + 1) else (d - no) // (ni + 1 + no)
    pol = Policy(ni, no, hidden)
    if pol.dim != d:
        raise ValueError(f"param count {d} does not match any policy shape")
    return pol, theta


def rl_factory(path, greedy: bool = True):
    """Return a factory(rng) -> RLBot for use in tournaments."""
    pol, theta = load(path)
    return lambda r: RLBot(pol, theta, greedy=greedy,
                           seed=random.Random(r.random()).randrange(1 << 30))
