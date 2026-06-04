"""Recover the board's spatial layout from move data alone.

Input: only ``from_idx`` / ``to_idx`` arrays. No notion of rows, columns, or
distance is provided. We build a cell-to-cell affinity from how often moves
connect two cells, then embed the 64 cells into 2-D with Laplacian eigenmaps.

The intuition: short-range moves (king steps, pawn pushes) connect cells that
are physically adjacent and are very common, so the dominant connectivity of
the affinity graph is the grid's local structure. The low-frequency Laplacian
eigenvectors then lay the cells out on (a rotated, gently warped) 8x8 grid.

Nothing here imports the ground-truth coordinates; evaluation lives in
``evaluate_geometry`` which is handed the answer key from the caller.
"""
from __future__ import annotations

import numpy as np

N_CELLS = 64


def cooccurrence(from_idx: np.ndarray, to_idx: np.ndarray) -> np.ndarray:
    """Symmetric move co-occurrence counts between cells."""
    A = np.zeros((N_CELLS, N_CELLS), dtype=np.float64)
    np.add.at(A, (from_idx, to_idx), 1.0)
    A = A + A.T
    np.fill_diagonal(A, 0.0)
    return A


def embed(from_idx: np.ndarray, to_idx: np.ndarray, dim: int = 2) -> np.ndarray:
    """Return an ``(64, dim)`` layout via normalized Laplacian eigenmaps."""
    A = cooccurrence(from_idx, to_idx)

    # PMI-style reweighting damps "hub" long-range edges from sliding pieces so
    # that local (short-move) structure dominates the embedding.
    deg = A.sum(axis=1, keepdims=True)
    total = A.sum()
    expected = (deg @ deg.T) / max(total, 1.0)
    W = A / np.maximum(expected, 1e-9)
    W = np.where(A > 0, np.log1p(W), 0.0)
    W = 0.5 * (W + W.T)

    d = W.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    L = np.eye(N_CELLS) - (d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :])

    vals, vecs = np.linalg.eigh(L)
    order = np.argsort(vals)
    # Skip the trivial near-zero eigenvector; take the next ``dim``.
    cols = order[1 : 1 + dim]
    coords = vecs[:, cols] * d_inv_sqrt[:, None]
    # Standardize for stable downstream distances.
    coords = (coords - coords.mean(0)) / (coords.std(0) + 1e-12)
    return coords


# --------------------------------------------------------------------------
# Evaluation (uses the ground-truth answer key, never seen by ``embed``).
# --------------------------------------------------------------------------
def _procrustes(X: np.ndarray, Y: np.ndarray) -> float:
    """Disparity in [0,1] after best similarity transform mapping X->Y."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    X = X / np.linalg.norm(X)
    Y = Y / np.linalg.norm(Y)
    U, s, Vt = np.linalg.svd(X.T @ Y)
    R = U @ Vt
    return float(1.0 - (s.sum() ** 2))  # 0 = perfect alignment


def neighbor_preservation(coords: np.ndarray, true_coords: np.ndarray) -> float:
    """Fraction of each cell's true grid neighbours recovered as nearest cells.

    True neighbours = cells at king-step distance 1 (8-neighbourhood). For each
    cell we look at as many recovered nearest cells as it has true neighbours
    and measure overlap.
    """
    td = np.abs(true_coords[:, None, :] - true_coords[None, :, :])
    true_adj = (np.maximum(td[..., 0], td[..., 1]) == 1)  # Chebyshev distance 1

    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    order = np.argsort(D, axis=1)

    scores = []
    for i in range(N_CELLS):
        k = int(true_adj[i].sum())
        if k == 0:
            continue
        recovered = set(order[i, :k].tolist())
        truth = set(np.nonzero(true_adj[i])[0].tolist())
        scores.append(len(recovered & truth) / k)
    return float(np.mean(scores))


def evaluate_geometry(coords: np.ndarray, true_coords: np.ndarray) -> dict:
    return {
        "procrustes_disparity": _procrustes(coords, true_coords),
        "neighbor_preservation": neighbor_preservation(coords, true_coords),
    }


def aligned(coords: np.ndarray, true_coords: np.ndarray) -> np.ndarray:
    """Similarity-align recovered coords onto the true grid (for plotting)."""
    X = coords - coords.mean(0)
    Y = true_coords - true_coords.mean(0)
    sx = np.linalg.norm(X)
    U, s, Vt = np.linalg.svd((X / sx).T @ (Y / np.linalg.norm(Y)))
    R = U @ Vt
    scale = s.sum() * np.linalg.norm(Y) / sx
    return (X @ R) * scale + true_coords.mean(0)
