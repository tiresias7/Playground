"""Recover strategy (move preference) and contrast it with rules.

Conditioned on the legal set (shown recoverable in Experiment 3), a bot's
choice among legal moves is its *strategy*. We fit a ranker that scores legal
candidate moves and predicts which one the bot played, then measure how well it
transfers between bots.

The headline contrast:
  * Legality transfers across bots (rules are invariant) -- see ``legality``.
  * Move preference does NOT transfer (strategy is bot-specific) -- shown here
    by a train-bot x eval-bot top-1 accuracy matrix that is strongly diagonal.

We also decompose predictability into bits explained by rules vs. by strategy.
"""
from __future__ import annotations

import chess
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .datagen import Dataset
from .legality import FeatureBuilder, _mode_symbol


def _legal_candidates(fen, codec):
    """Legal moves of a position as data-index (from,to) pairs."""
    board = chess.Board(fen)
    return [
        (int(codec.cell_perm_inv[m.from_square]), int(codec.cell_perm_inv[m.to_square]))
        for m in board.legal_moves
    ]


def _build_ranker_data(ds: Dataset, fb: FeatureBuilder, codec, max_positions=4000, seed=0):
    rng = np.random.default_rng(seed)
    n = min(max_positions, len(ds))
    pick = rng.choice(len(ds), size=n, replace=False)
    states, f, t, y = [], [], [], []
    for i in pick:
        cands = _legal_candidates(ds.fens[i], codec)
        chosen = (int(ds.from_idx[i]), int(ds.to_idx[i]))
        for (cf, ct) in cands:
            states.append(ds.states[i]); f.append(cf); t.append(ct)
            y.append(1 if (cf, ct) == chosen else 0)
    states = np.array(states); f = np.array(f); t = np.array(t); y = np.array(y)
    X = fb.build(states, f, t)
    return X, y


def train_strategy(ds: Dataset, grid_coords, codec, seed=0, max_positions=4000):
    bg = _mode_symbol(ds.states)
    fb = FeatureBuilder(grid_coords, bg, n_symbols=int(ds.states.max()) + 1)
    X, y = _build_ranker_data(ds, fb, codec, max_positions=max_positions, seed=seed)
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.15,
                                         random_state=seed)
    clf.fit(X, y)
    return clf, fb


def top1_accuracy(clf, fb, ds: Dataset, codec, max_positions=1500, seed=0):
    """Fraction of positions where the top-scored legal move is the bot's move."""
    rng = np.random.default_rng(seed)
    n = min(max_positions, len(ds))
    pick = rng.choice(len(ds), size=n, replace=False)
    correct = 0
    for i in pick:
        cands = _legal_candidates(ds.fens[i], codec)
        if not cands:
            continue
        states = np.tile(ds.states[i], (len(cands), 1))
        f = np.array([c[0] for c in cands]); t = np.array([c[1] for c in cands])
        X = fb.build(states, f, t)
        scores = clf.predict_proba(X)[:, 1]
        best = cands[int(np.argmax(scores))]
        if best == (int(ds.from_idx[i]), int(ds.to_idx[i])):
            correct += 1
    return correct / n


def decompose_bits(ds: Dataset, codec, max_positions=1500, seed=0):
    """Split move predictability into a rules term and a strategy ceiling.

    * random guess: log2(64*64) bits to name a move blind.
    * rules: log2(#legal moves) -- bits left once the legal set is known.
    * the gap (random - rules) is what knowing the *rules* alone buys you;
      strategy can only shave the residual log2(#legal).
    """
    rng = np.random.default_rng(seed)
    n = min(max_positions, len(ds))
    pick = rng.choice(len(ds), size=n, replace=False)
    rules_bits = []
    for i in pick:
        nlegal = max(len(_legal_candidates(ds.fens[i], codec)), 1)
        rules_bits.append(np.log2(nlegal))
    blind = np.log2(64 * 64)
    rb = float(np.mean(rules_bits))
    return {
        "blind_bits": float(blind),
        "rules_residual_bits": rb,
        "bits_explained_by_rules": float(blind - rb),
        "frac_explained_by_rules": float((blind - rb) / blind),
    }
