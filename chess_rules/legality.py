"""Recover the legal-move *support* (the rules) from observations.

We only ever see moves the bots actually played -- all legal, mostly preferred.
To recover the rule "which moves are even possible", we use negative sampling
(a form of PU learning): observed moves are positives, random ``(from, to)``
corruptions are treated as negatives. Some corruptions are in fact legal
(label noise), so the classifier learns the *shape* of the legal support rather
than memorizing. We then score it against python-chess ground truth, which the
classifier never sees during training.

Features are built only from the opaque state, the move's cells, and a grid
scaffold (established as recoverable in Experiment 1). "Empty" is not given --
we infer the background symbol as the most common occupant and define path
occupancy relative to it.
"""
from __future__ import annotations

import chess
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from .datagen import Dataset
from .encoding import N_CELLS


class FeatureBuilder:
    """Turns (state, from, to) triples into legality features."""

    def __init__(self, grid_coords: np.ndarray, background_symbol: int, n_symbols: int = 13):
        self.grid = np.rint(grid_coords).astype(int)
        self.background = int(background_symbol)
        self.n_symbols = n_symbols
        self.coord2idx = {(int(x), int(y)): i for i, (x, y) in enumerate(self.grid)}

    def _path_block(self, states, from_idx, to_idx):
        """Count non-background cells strictly between from and to (grid line)."""
        n = len(from_idx)
        out = np.zeros(n, dtype=np.float64)
        g = self.grid
        for i in range(n):
            fx, fy = g[from_idx[i]]
            tx, ty = g[to_idx[i]]
            dx, dy = tx - fx, ty - fy
            adx, ady = abs(dx), abs(dy)
            if not ((dx == 0 or dy == 0 or adx == ady) and max(adx, ady) > 1):
                continue  # not a sliding line; no path squares
            steps = max(adx, ady)
            ux, uy = np.sign(dx), np.sign(dy)
            blocked = 0
            for k in range(1, steps):
                idx = self.coord2idx.get((int(fx + ux * k), int(fy + uy * k)))
                if idx is not None and states[i, idx] != self.background:
                    blocked += 1
            out[i] = blocked
        return out

    def build(self, states, from_idx, to_idx):
        g = self.grid
        d = g[to_idx] - g[from_idx]
        df, dr = d[:, 0], d[:, 1]
        adf, adr = np.abs(df), np.abs(dr)
        cheb = np.maximum(adf, adr)
        nz = (cheb > 0).astype(float)

        is_knight = (((adf == 1) & (adr == 2)) | ((adf == 2) & (adr == 1))).astype(float)
        is_diag = ((adf == adr) & (cheb > 0)).astype(float)
        is_orth = (((df == 0) | (dr == 0)) & (cheb > 0)).astype(float)
        is_step = (cheb == 1).astype(float)

        mover = states[np.arange(len(from_idx)), from_idx]
        target = states[np.arange(len(to_idx)), to_idx]
        mover_oh = np.eye(self.n_symbols)[mover]
        target_oh = np.eye(self.n_symbols)[target]

        from_bg = (mover == self.background).astype(float)
        to_bg = (target == self.background).astype(float)
        path = self._path_block(states, from_idx, to_idx)

        geo = np.column_stack(
            [df, dr, adf, adr, cheb, nz, is_knight, is_diag, is_orth, is_step,
             from_bg, to_bg, path]
        )
        return np.hstack([geo, mover_oh, target_oh])


def _mode_symbol(states):
    return int(np.bincount(states.reshape(-1)).argmax())


def _sample_negatives(ds, rng, k=1):
    """Random (from,to) corruptions per positive row (treated as negatives)."""
    n = len(ds)
    states = np.repeat(ds.states, k, axis=0)
    f = rng.integers(0, N_CELLS, size=n * k)
    t = rng.integers(0, N_CELLS, size=n * k)
    bad = f == t
    t[bad] = (t[bad] + 1) % N_CELLS
    return states, f, t


def train_legality(ds: Dataset, grid_coords, neg_per_pos=2, seed=0):
    """Train the legality classifier on a dataset via negative sampling."""
    rng = np.random.default_rng(seed)
    bg = _mode_symbol(ds.states)
    fb = FeatureBuilder(grid_coords, bg, n_symbols=int(ds.states.max()) + 1)

    Xpos = fb.build(ds.states, ds.from_idx, ds.to_idx)
    ns, nf, nt = _sample_negatives(ds, rng, k=neg_per_pos)
    Xneg = fb.build(ns, nf, nt)

    X = np.vstack([Xpos, Xneg])
    y = np.concatenate([np.ones(len(Xpos)), np.zeros(len(Xneg))])
    clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.15,
                                         random_state=seed)
    clf.fit(X, y)
    return clf, fb


def _ground_truth_candidates(fen, rng, n_illegal=12):
    """Build (from,to,legal?) candidates for one position via python-chess."""
    board = chess.Board(fen)
    legal = {(m.from_square, m.to_square) for m in board.legal_moves}
    cand = [(f, t, 1) for (f, t) in legal]
    # Sample illegal corruptions for balance.
    tries = 0
    illegal = []
    while len(illegal) < n_illegal and tries < n_illegal * 6:
        tries += 1
        f = int(rng.integers(64)); t = int(rng.integers(64))
        if f != t and (f, t) not in legal:
            illegal.append((f, t, 0))
    return cand + illegal, board


def evaluate_legality(clf, fb, eval_ds: Dataset, codec, seed=0, max_positions=400):
    """Score recovered legality against python-chess ground truth.

    Candidates and labels come from real boards (eval only); features are built
    from the opaque state. Cells are mapped through the codec so the
    classifier's grid scaffold lines up with python-chess squares.
    """
    rng = np.random.default_rng(seed)
    fens = eval_ds.fens
    n = min(max_positions, len(fens))
    pick = rng.choice(len(fens), size=n, replace=False)

    all_states, all_f, all_t, all_y = [], [], [], []
    for i in pick:
        cands, board = _ground_truth_candidates(fens[i], rng)
        state = eval_ds.states[i]
        for sq_f, sq_t, lab in cands:
            # python-chess square -> opaque data index
            all_states.append(state)
            all_f.append(int(codec.cell_perm_inv[sq_f]))
            all_t.append(int(codec.cell_perm_inv[sq_t]))
            all_y.append(lab)

    states = np.array(all_states)
    f = np.array(all_f); t = np.array(all_t); y = np.array(all_y)
    X = fb.build(states, f, t)
    p = clf.predict_proba(X)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "avg_precision": float(average_precision_score(y, p)),
        "base_rate_legal": float(y.mean()),
        "n_candidates": int(len(y)),
    }
