"""Can we predict the look-ahead bots better with richer features?

The ceiling analysis (SANITY [I]) showed center/mobility/minimax are highly
predictable in principle, yet our static geometry+symbol model reaches only
36-52% of ceiling. Hypothesis: that gap is a *feature* limitation -- those
strategies score moves by their *consequences* (destination centrality, the
mobility/material that results), which a single-frame move descriptor cannot
see.

We test three feature sets for the move ranker:
  base     : the legality FeatureBuilder features (current model).
  +geo     : + destination centrality from the RECOVERED grid (honest).
  +oracle  : + consequences that require simulating the move (material gained,
             replies left to the opponent). Labelled ORACLE because it uses the
             engine -- it is an upper bound on what consequence-features buy,
             not a firewall-respecting model.

If +oracle closes the gap to ceiling, the shortfall is feature-poverty, not a
fundamental limit.

Run: python -m chess_rules.enrich
"""
from __future__ import annotations

import chess
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from . import datagen
from .bots import ALL_BOTS, _PIECE_VALUE
from .encoding import make_codec
from .legality import FeatureBuilder, _mode_symbol

BOTS = ["center", "mobility", "minimax"]


def _extra_features(board, f_sq, t_sq, grid_centroid, codec, kinds):
    """Per-candidate extra features for a single (board, move)."""
    feats = []
    if "geo" in kinds or "oracle" in kinds:
        to_idx = codec.cell_perm_inv[t_sq]
        c = codec.true_coords[to_idx]
        feats.append(-np.hypot(*(c - grid_centroid)))  # destination centrality
    if "oracle" in kinds:
        victim = board.piece_at(t_sq)
        feats.append(_PIECE_VALUE[victim.piece_type] if victim else 0.0)  # material
        mv = board.find_move(f_sq, t_sq) if _can(board, f_sq, t_sq) else None
        if mv is not None:
            board.push(mv)
            feats.append(-board.legal_moves.count())  # fewer opponent replies = better
            board.pop()
        else:
            feats.append(0.0)
    return feats


def _can(board, f, t):
    return any(m.from_square == f and m.to_square == t for m in board.legal_moves)


def _ranker_xy(ds, fb, codec, kinds, grid_centroid, max_pos, seed):
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(ds), size=min(max_pos, len(ds)), replace=False)
    base_s, f, t, y, extra = [], [], [], [], []
    for i in pick:
        board = chess.Board(ds.fens[i])
        chosen = (int(ds.from_idx[i]), int(ds.to_idx[i]))
        for m in board.legal_moves:
            fi, ti = codec.encode_move(m)
            base_s.append(ds.states[i]); f.append(fi); t.append(ti)
            y.append(1 if (fi, ti) == chosen else 0)
            if kinds:
                extra.append(_extra_features(board, m.from_square, m.to_square,
                                             grid_centroid, codec, kinds))
    X = fb.build(np.array(base_s), np.array(f), np.array(t))
    if kinds:
        X = np.hstack([X, np.array(extra)])
    return X, np.array(y)


def _top1_and_ceiling(clf, fb, ds, codec, kinds, grid_centroid, bot, max_pos, seed):
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(ds), size=min(max_pos, len(ds)), replace=False)
    hit, ceil = [], []
    for i in pick:
        board = chess.Board(ds.fens[i])
        moves, p = bot.policy(board)
        if len(moves) < 2:
            continue
        ceil.append(p.max())
        base_s, f, t, extra = [], [], [], []
        for m in moves:
            fi, ti = codec.encode_move(m)
            base_s.append(ds.states[i]); f.append(fi); t.append(ti)
            if kinds:
                extra.append(_extra_features(board, m.from_square, m.to_square,
                                             grid_centroid, codec, kinds))
        X = fb.build(np.array(base_s), np.array(f), np.array(t))
        if kinds:
            X = np.hstack([X, np.array(extra)])
        sc = clf.predict_proba(X)[:, 1]
        best = (int(f[int(np.argmax(sc))]), int(t[int(np.argmax(sc))]))
        hit.append(best == (int(ds.from_idx[i]), int(ds.to_idx[i])))
    return float(np.mean(hit)), float(np.mean(ceil))


def run():
    codec = make_codec(seed=0)
    grid = codec.true_coords
    centroid = grid.mean(0)
    _log = lambda m: print(m, flush=True)
    _log("per-bot move top-1 as % of ceiling, by feature set\n")
    _log(f"{'bot':9s} {'ceiling':>8} {'base':>14} {'+geo':>14} {'+oracle':>14}")

    for b in BOTS:
        train = datagen.generate(codec, [b], games_per_bot=40, seed=11, record_fens=True)
        ev = datagen.generate(codec, [b], games_per_bot=18, seed=22, record_fens=True)
        bg = _mode_symbol(train.states)
        fb = FeatureBuilder(grid, bg, n_symbols=int(train.states.max()) + 1)
        bot = ALL_BOTS[b](seed=0)

        row = []
        ceiling = None
        for kinds in ([], ["geo"], ["geo", "oracle"]):
            X, y = _ranker_xy(train, fb, codec, kinds, centroid, 2500, seed=0)
            clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.15,
                                                 random_state=0).fit(X, y)
            top1, ceil = _top1_and_ceiling(clf, fb, ev, codec, kinds, centroid,
                                           bot, 800, seed=0)
            ceiling = ceil
            row.append((top1, top1 / ceil * 100))
        _log(f"{b:9s} {ceiling:>8.3f} "
             + " ".join(f"{t:.3f} ({pc:>3.0f}%)" for t, pc in row))

    _log("\n+geo = destination centrality (recovered grid, honest).")
    _log("+oracle = + material gained + opponent replies after the move (uses the")
    _log("engine; an upper bound showing whether the gap is feature-poverty).")


if __name__ == "__main__":
    run()
