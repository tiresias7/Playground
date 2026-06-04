"""Sanity checks for the rule-recovery experiment.

Addresses the obvious skeptic's questions:

  A. Position overlap -- do train and eval share positions (memorization risk)?
  B. Unseen-move generalization -- the classifier is scored on legal moves it
     never saw *played*; does it still call them legal?
  C. Memorization baselines -- can a pure frequency lookup match the model? If
     so, the "rule" is just memorized common move-shapes.
  D. Hard negatives -- legal vs. *plausible-but-illegal* moves (self-capture,
     pin/check violations), not just random garbage. Reveals which layers of
     the rules were actually recovered.
  E. Rule extrapolation -- hide all long slides from training, then test whether
     the model still labels held-out long slides legal (rule, not lookup).
  F. Complexity -- branching factor, game phase, per-piece move coverage.

Run: python -m chess_rules.sanity
"""
from __future__ import annotations

import chess
import numpy as np
from sklearn.metrics import roc_auc_score

from . import datagen
from .encoding import PIECE_NAME, make_codec
from .legality import FeatureBuilder, _mode_symbol, train_legality

BOTS = ["random", "capture", "center", "mobility", "minimax"]


def _log(m):
    print(m, flush=True)


# ---------------------------------------------------------------------------
def _piece_offset_pred(piece_type, color):
    def pred(df, dr):
        adf, adr = abs(df), abs(dr)
        cheb = max(adf, adr)
        if cheb == 0:
            return False
        if piece_type == chess.KNIGHT:
            return (adf, adr) in {(1, 2), (2, 1)}
        if piece_type == chess.BISHOP:
            return adf == adr
        if piece_type == chess.ROOK:
            return df == 0 or dr == 0
        if piece_type == chess.QUEEN:
            return adf == adr or df == 0 or dr == 0
        if piece_type == chess.KING:
            return cheb == 1 or (adr == 0 and adf == 2)
        if piece_type == chess.PAWN:
            fwd = 1 if color else -1
            return (df == 0 and dr in (fwd, 2 * fwd)) or (adf == 1 and dr == fwd)
        return False
    return pred


def _candidates(fen, rng, kind, n=14):
    """Return list of (from_sq, to_sq, legal?) candidates of a given negative kind."""
    board = chess.Board(fen)
    legal = {(m.from_square, m.to_square) for m in board.legal_moves}
    pos = [(f, t, 1) for (f, t) in legal]

    neg = []
    if kind == "random":
        tries = 0
        while len(neg) < n and tries < n * 8:
            tries += 1
            f, t = int(rng.integers(64)), int(rng.integers(64))
            if f != t and (f, t) not in legal:
                neg.append((f, t, 0))
    elif kind == "self_capture":
        # Geometrically valid, path-clear moves that land on a friendly piece.
        for sq in chess.SquareSet(board.occupied_co[board.turn]):
            for t in board.attacks(sq):
                tp = board.piece_at(t)
                if tp is not None and tp.color == board.turn and (sq, t) not in legal:
                    neg.append((sq, t, 0))
    elif kind == "pin_check":
        # Pseudo-legal but not legal: illegal only because of king safety.
        legalset = set(legal)
        for m in board.pseudo_legal_moves:
            if (m.from_square, m.to_square) not in legalset:
                neg.append((m.from_square, m.to_square, 0))
    if len(neg) > n:
        idx = rng.choice(len(neg), size=n, replace=False)
        neg = [neg[i] for i in idx]
    return pos + neg, board


def _score_candidates(clf, fb, codec, fens, kind, rng, max_positions=400):
    states_idx, f, t, y = [], [], [], []
    fb_states = []
    pick = rng.choice(len(fens), size=min(max_positions, len(fens)), replace=False)
    for i in pick:
        cands, _ = _candidates(fens[i], rng, kind)
        if not any(lab == 0 for _, _, lab in cands):
            continue
        for sf, st, lab in cands:
            fb_states.append(i)
            f.append(int(codec.cell_perm_inv[sf]))
            t.append(int(codec.cell_perm_inv[st]))
            y.append(lab)
    return np.array(fb_states), np.array(f), np.array(t), np.array(y)


# ---------------------------------------------------------------------------
def run():
    rng = np.random.default_rng(0)
    codec = make_codec(seed=0)
    grid = codec.true_coords

    _log("generating data (train + eval, with FENs) ...")
    train = datagen.generate(codec, BOTS, games_per_bot=45, seed=0, record_fens=True)
    ev = datagen.generate(codec, BOTS, games_per_bot=18, seed=99, record_fens=True)

    # ---- A. position overlap -------------------------------------------
    # Measure on the OPAQUE STATE the model actually sees (occupancy only),
    # not the FEN (whose move counters make every position look unique).
    ev_fens = ev.fens
    train_states = {s.tobytes() for s in train.states}
    ev_state_overlap = np.mean([s.tobytes() in train_states for s in ev.states])
    # also on (state, played-move) pairs
    train_sm = {(s.tobytes(), int(a), int(b))
                for s, a, b in zip(train.states, train.from_idx, train.to_idx)}
    ev_sm_overlap = np.mean([(s.tobytes(), int(a), int(b)) in train_sm
                             for s, a, b in zip(ev.states, ev.from_idx, ev.to_idx)])
    _log("\n[A] Overlap on what the model sees (occupancy state, not FEN)")
    _log(f"    eval states also seen in train:        {ev_state_overlap:.1%}")
    _log(f"    eval (state,move) also seen in train:  {ev_sm_overlap:.1%}")
    _log(f"    distinct train states: {len(train_states):,} / {len(train):,} rows")

    clf, fb = train_legality(train, grid, seed=0)

    # ---- D. legal vs different negative kinds --------------------------
    _log("\n[D] Legality AUC by negative type (the 0.995 stress test)")
    for kind in ("random", "self_capture", "pin_check"):
        si, f, t, y = _score_candidates(clf, fb, codec, ev_fens, kind, rng, 400)
        if len(np.unique(y)) < 2:
            _log(f"    {kind:12s}: (insufficient negatives)")
            continue
        X = fb.build(ev.states[si], f, t)
        p = clf.predict_proba(X)[:, 1]
        _log(f"    legal vs {kind:12s}: AUC={roc_auc_score(y, p):.3f}  "
             f"(n={len(y)}, legal frac={y.mean():.2f})")

    # ---- B. unseen-move generalization ---------------------------------
    played = set(zip(train.from_idx.tolist(), train.to_idx.tolist()))
    si, f, t, y = _score_candidates(clf, fb, codec, ev_fens, "random", rng, 400)
    seen = np.array([(int(a), int(b)) in played for a, b in zip(f, t)])
    legal_unseen = (y == 1) & (~seen)
    _log("\n[B] Generalization to legal moves never observed as played")
    _log(f"    legal eval candidates never played in train: {legal_unseen.sum()}/"
         f"{(y==1).sum()} ({legal_unseen.mean()*100:.0f}% of legal)")
    Xb = fb.build(ev.states[si], f, t)
    pb = clf.predict_proba(Xb)[:, 1]
    # AUC on a balanced set of unseen-legal vs illegal.
    keep = legal_unseen | (y == 0)
    _log(f"    AUC(unseen-legal vs illegal): {roc_auc_score(y[keep], pb[keep]):.3f}")

    # ---- C. memorization baselines -------------------------------------
    _log("\n[C] Memorization baselines vs the feature model")
    _log("    A pure lookup that beats the model => result is mostly memorized.")
    # (i) exact (from,to) index-pair frequency
    pair_count = {}
    for a, b in zip(train.from_idx.tolist(), train.to_idx.tolist()):
        pair_count[(a, b)] = pair_count.get((a, b), 0) + 1
    # (ii) (mover_symbol, displacement) frequency -- generalizes by shape
    mover_tr = train.states[np.arange(len(train)), train.from_idx]
    disp_tr = (grid[train.to_idx] - grid[train.from_idx])
    shape_count = {}
    for ms, (dx, dy) in zip(mover_tr.tolist(), disp_tr.astype(int).tolist()):
        shape_count[(ms, dx, dy)] = shape_count.get((ms, dx, dy), 0) + 1

    def _baselines(kind):
        si_, f_, t_, y_ = _score_candidates(clf, fb, codec, ev_fens, kind, rng, 400)
        if len(np.unique(y_)) < 2:
            return
        mover_ = ev.states[si_, f_]
        dxy_ = (grid[t_] - grid[f_]).astype(int)
        pair_s = np.array([pair_count.get((int(a), int(b)), 0) for a, b in zip(f_, t_)], float)
        shape_s = np.array([shape_count.get((int(m), int(dx), int(dy)), 0)
                            for m, (dx, dy) in zip(mover_, dxy_)], float)
        model_s = clf.predict_proba(fb.build(ev.states[si_], f_, t_))[:, 1]
        _log(f"    vs {kind:12s}: exact-pair={roc_auc_score(y_, pair_s):.3f}  "
             f"(sym,disp)={roc_auc_score(y_, shape_s):.3f}  "
             f"model={roc_auc_score(y_, model_s):.3f}")

    _baselines("random")
    _baselines("self_capture")

    # ---- E. rule extrapolation (hold out long slides) ------------------
    _log("\n[E] Rule extrapolation: train with NO move of reach>=5, test on them")
    reach_tr = np.abs(grid[train.to_idx] - grid[train.from_idx]).max(axis=1)
    short_mask = reach_tr < 5
    train_short = train.subset(short_mask)
    _log(f"    dropped {(~short_mask).sum():,} long moves; trained on {short_mask.sum():,}")
    clf2, fb2 = train_legality(train_short, grid, seed=0)
    # Build candidates that are specifically long slides: legal long slides vs
    # illegal long (from,to) of the same reach.
    si2, f2, t2, y2 = _score_candidates(clf2, fb2, codec, ev_fens, "random", rng, 400)
    reach_ev = np.abs(grid[t2] - grid[f2]).max(axis=1)
    longmask = reach_ev >= 5
    if longmask.sum() > 20 and len(np.unique(y2[longmask])) == 2:
        X2 = fb2.build(ev.states[si2], f2, t2)
        p2 = clf2.predict_proba(X2)[:, 1]
        _log(f"    AUC on held-out long moves: {roc_auc_score(y2[longmask], p2[longmask]):.3f}"
             f"  (n={longmask.sum()})")
        # Of legal long slides, fraction the model calls legal (>0.5).
        ll = longmask & (y2 == 1)
        _log(f"    legal long slides recognised (p>0.5): "
             f"{(p2[ll] > 0.5).mean()*100:.0f}% of {ll.sum()}")

    # ---- F. complexity stats -------------------------------------------
    _log("\n[F] Complexity of the observed distribution")
    branch, npieces = [], []
    for fen in rng.choice(ev_fens, size=min(800, len(ev_fens)), replace=False):
        b = chess.Board(fen)
        branch.append(b.legal_moves.count())
        npieces.append(len(b.piece_map()))
    branch = np.array(branch); npieces = np.array(npieces)
    _log(f"    branching factor: mean={branch.mean():.1f}  "
         f"p5={np.percentile(branch,5):.0f}  p95={np.percentile(branch,95):.0f}")
    _log(f"    pieces on board : mean={npieces.mean():.1f}  "
         f"min={npieces.min()}  max={npieces.max()}  (32=opening)")

    # per-piece displacement coverage (recall of distinct legal offsets seen)
    _log("    per-piece move-shape coverage (distinct offsets seen / true):")
    _coverage(train, grid, codec)


def _coverage(train, grid, codec):
    """How many of each piece's distinct legal offsets were ever observed."""
    mover = train.states[np.arange(len(train)), train.from_idx]
    disp = (grid[train.to_idx] - grid[train.from_idx]).astype(int)
    # group distinct offsets by true piece type (color-folded)
    seen = {}
    for ms, (dx, dy) in zip(mover.tolist(), disp.tolist()):
        pt = codec.true_piece_type.get(int(ms))
        if pt is None:
            continue
        seen.setdefault(pt, set()).add((abs(dx), abs(dy)))
    # crude "true distinct |offset|" counts (color-folded, ignoring pawn dir)
    truth = {chess.KNIGHT: 2, chess.KING: 4, chess.PAWN: 3,
             chess.BISHOP: 7, chess.ROOK: 14, chess.QUEEN: 21}
    for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
        _log(f"      {PIECE_NAME[pt]:7s}: {len(seen.get(pt, set())):>2} distinct "
             f"|offset| classes observed (true ~{truth[pt]})")


if __name__ == "__main__":
    run()
