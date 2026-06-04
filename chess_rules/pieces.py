"""Recover per-symbol movement rules without knowing what the symbols mean.

Given the board geometry (from Experiment 1) and the opaque occupant symbol of
the moving cell, we ask: does each symbol have a characteristic *set of
displacement vectors*? If chess structure is present, the symbols should sort
themselves into a handful of movement archetypes -- the piece kinds -- and the
recovered displacement set for each should match a real piece's move pattern.

This module never uses piece names or types. It is handed the recovered (or
true) grid coordinates and the opaque dataset, and returns, per symbol, a
movement fingerprint plus an unsupervised clustering of symbols. Scoring
against the true piece types happens in ``evaluate_pieces`` using an answer key
supplied by the caller.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from .datagen import Dataset


def _round_disp(coords, from_idx, to_idx):
    """Integer displacement vectors on the (snapped) grid."""
    d = coords[to_idx] - coords[from_idx]
    return np.rint(d).astype(int)


def _fingerprint(disp: np.ndarray) -> dict:
    """Summarize a bag of displacement vectors into archetype frequencies.

    Directions (knight / diagonal / orthogonal) describe the *piece kind* and
    are color-invariant; ``long`` separates sliders from steppers; ``reach`` is
    the longest move seen. ``forward_bias`` encodes the moving side's direction
    (i.e. color), so it is reported but kept out of the type clustering.
    """
    df = np.abs(disp[:, 0])
    dr = disp[:, 1]
    adr = np.abs(dr)
    cheb = np.maximum(df, adr)
    nz = cheb > 0
    n = max(int(nz.sum()), 1)

    is_knight = (((df == 1) & (adr == 2)) | ((df == 2) & (adr == 1)))
    is_diag = (df == adr) & nz & ~is_knight
    is_orth = ((df == 0) | (adr == 0)) & nz & ~is_knight
    is_long = (cheb >= 2) & ~is_knight
    return {
        "knight": float(is_knight.sum()) / n,
        "diag": float(is_diag.sum()) / n,
        "orth": float(is_orth.sum()) / n,
        "long": float((is_long & nz).sum()) / n,
        "max_reach": float(cheb[nz].max()) if nz.any() else 0.0,
        "forward_bias": float(np.sign(dr[nz]).mean()) if nz.any() else 0.0,
        "n_obs": int(nz.sum()),
    }


def symbol_fingerprints(ds: Dataset, coords: np.ndarray) -> dict:
    """Map each moving-symbol -> movement fingerprint."""
    moving_symbol = ds.states[np.arange(len(ds)), ds.from_idx]
    disp = _round_disp(coords, ds.from_idx, ds.to_idx)
    out = {}
    for s in np.unique(moving_symbol):
        m = moving_symbol == s
        out[int(s)] = _fingerprint(disp[m])
    return out


# Color-invariant, type-discriminative features (forward_bias deliberately omitted).
_FEATURES = ["knight", "diag", "orth", "long", "max_reach"]


def cluster_symbols(fingerprints: dict, n_clusters: int = 6, seed: int = 0):
    """Cluster symbols by movement fingerprint (unsupervised)."""
    syms = sorted(fingerprints)
    X = np.array([[fingerprints[s][f] for f in _FEATURES] for s in syms], dtype=float)
    Xn = X.copy()
    Xn[:, -1] /= 7.0  # normalize max_reach to ~[0,1]
    k = min(n_clusters, len(syms))
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Xn)
    return syms, dict(zip(syms, labels.tolist()))


# --------------------------------------------------------------------------
# Evaluation against the answer key.
# --------------------------------------------------------------------------
def evaluate_pieces(ds, coords, codec) -> dict:
    """Score recovered movement structure against true piece types."""
    fps = symbol_fingerprints(ds, coords)
    moving_symbols = [s for s in fps if codec.true_piece_type.get(s) is not None]

    # 1) Unsupervised clustering of moving symbols vs. true piece type.
    syms, clab = cluster_symbols({s: fps[s] for s in moving_symbols}, n_clusters=6)
    true_type = [codec.true_piece_type[s] for s in syms]
    pred_clust = [clab[s] for s in syms]
    ari = adjusted_rand_score(true_type, pred_clust)

    # 2) Per-symbol displacement fidelity: do observed displacements fall inside
    #    the symbol's *true* legal move pattern? (precision) and how much of the
    #    pattern was seen (coverage of distinct directions).
    fidelity = _displacement_fidelity(ds, coords, codec)

    return {
        "movement_cluster_ARI": float(ari),
        "displacement_precision": fidelity["precision"],
        "n_moving_symbols": len(moving_symbols),
        "fingerprints": fps,
        "clusters": clab,
    }


def _legal_offsets(piece_type, color):
    """Canonical move-offset predicate for a true piece type (df, dr)."""
    import chess

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
            return cheb == 1 or (adr == 0 and adf == 2)  # incl. castling slide
        if piece_type == chess.PAWN:
            fwd = 1 if color else -1
            return (df == 0 and dr in (fwd, 2 * fwd)) or (adf == 1 and dr == fwd)
        return False

    return pred


def _displacement_fidelity(ds, coords, codec) -> dict:
    """Fraction of observed displacements that are legal for the true piece."""
    moving_symbol = ds.states[np.arange(len(ds)), ds.from_idx]
    disp = _round_disp(coords, ds.from_idx, ds.to_idx)

    total, ok = 0, 0
    for s in np.unique(moving_symbol):
        pt = codec.true_piece_type.get(int(s))
        if pt is None:
            continue
        color = codec.true_color[int(s)]
        pred = _legal_offsets(pt, color)
        m = moving_symbol == s
        for df, dr in disp[m]:
            total += 1
            if pred(int(df), int(dr)):
                ok += 1
    return {"precision": ok / max(total, 1)}
