"""A tiny numpy policy network (no torch). Parameters live as one flat vector so
evolution strategies can perturb them directly."""

from __future__ import annotations

import numpy as np


class Policy:
    def __init__(self, n_in: int, n_out: int, hidden: int = 0):
        self.n_in, self.n_out, self.hidden = n_in, n_out, hidden
        if hidden:
            self.shapes = [(hidden, n_in), (hidden,), (n_out, hidden), (n_out,)]
        else:
            self.shapes = [(n_out, n_in), (n_out,)]
        self.sizes = [int(np.prod(s)) for s in self.shapes]
        self.dim = sum(self.sizes)

    def init_theta(self, rng: np.random.Generator, scale: float = 0.1) -> np.ndarray:
        # Small weights so the initial policy is near-uniform over valid actions.
        return rng.standard_normal(self.dim).astype(np.float32) * scale

    def _unpack(self, theta):
        out, i = [], 0
        for shp, sz in zip(self.shapes, self.sizes):
            out.append(theta[i:i + sz].reshape(shp)); i += sz
        return out

    def logits(self, theta, f):
        p = self._unpack(theta)
        if self.hidden:
            W1, b1, W2, b2 = p
            h = np.tanh(W1 @ f + b1)
            return W2 @ h + b2
        W, b = p
        return W @ f + b


def select(logits, mask, rng=None, greedy=True):
    """Pick an action id among valid (masked) actions."""
    z = np.where(mask, logits, -1e9)
    if greedy:
        return int(np.argmax(z))
    z = z - z.max()
    e = np.exp(z) * mask
    p = e / e.sum()
    return int(rng.choice(len(p), p=p))
