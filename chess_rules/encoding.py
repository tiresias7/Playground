"""Opaque encoding of chess positions and moves.

This module is the *only* place that knows the dataset comes from chess. It
turns a ground-truth board into a generic numeric record that hides every bit
of domain knowledge:

  * A position becomes a length-64 vector of integer **symbols**. Each symbol
    is the occupant of one board cell, but the integer codes are an arbitrary
    (seeded) permutation of {empty, the twelve piece kinds}. The learner is
    never told which code means "knight" or even which codes are pieces.

  * A move becomes a pair ``(from_index, to_index)`` of cell indices in
    ``range(64)``.

  * The 64 cells are shuffled by a fixed random permutation, so cell index
    order carries **no** 8x8 geometry. Recovering that geometry is one of the
    experiments.

The ground-truth tables below (true 2-D coordinates, true piece kind of each
symbol) are exported for the *evaluation* layer only. The learner code in
``geometry.py`` / ``pieces.py`` / ``legality.py`` / ``strategy.py`` must never
import them.
"""
from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np

N_CELLS = 64

# 13 occupant kinds: empty + 6 piece types x 2 colors.
_EMPTY = 0
_PIECE_KINDS = [
    (pt, color)
    for color in (chess.WHITE, chess.BLACK)
    for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)
]  # 12 entries


@dataclass(frozen=True)
class Codec:
    """A fixed, seeded scheme for turning boards/moves into opaque arrays.

    Attributes are split into two groups. ``cell_perm`` and ``symbol_of`` are
    needed to *produce* the opaque data. The ``true_*`` attributes are
    ground-truth answer keys used only to score how much structure a learner
    recovered.
    """

    seed: int
    cell_perm: np.ndarray          # data index -> chess square (0..63)
    cell_perm_inv: np.ndarray      # chess square -> data index
    symbol_of: dict                # (piece_type, color) | None -> symbol int
    n_symbols: int

    # --- ground-truth answer keys (evaluation only) ---
    true_coords: np.ndarray        # (64, 2) true (file, rank) of each data index
    true_piece_type: dict          # symbol -> chess piece_type (1..6) or None
    true_color: dict               # symbol -> chess.WHITE/BLACK or None
    empty_symbol: int

    def encode_board(self, board: chess.Board) -> np.ndarray:
        """Return the length-64 opaque occupant vector for ``board``."""
        out = np.empty(N_CELLS, dtype=np.int64)
        for data_idx in range(N_CELLS):
            sq = int(self.cell_perm[data_idx])
            piece = board.piece_at(sq)
            key = None if piece is None else (piece.piece_type, piece.color)
            out[data_idx] = self.symbol_of[key]
        return out

    def encode_move(self, move: chess.Move) -> tuple[int, int]:
        """Return ``(from_index, to_index)`` in data-index space."""
        return int(self.cell_perm_inv[move.from_square]), int(self.cell_perm_inv[move.to_square])


def make_codec(seed: int = 0, permute_cells: bool = True) -> Codec:
    """Build a reproducible :class:`Codec`.

    With ``permute_cells=True`` the 64 cell indices are shuffled so that no 8x8
    structure is implied by index order. With ``False`` the data index equals
    the native chess square (useful for ablations / sanity checks).
    """
    rng = np.random.default_rng(seed)

    if permute_cells:
        cell_perm = rng.permutation(N_CELLS)
    else:
        cell_perm = np.arange(N_CELLS)
    cell_perm_inv = np.empty(N_CELLS, dtype=np.int64)
    cell_perm_inv[cell_perm] = np.arange(N_CELLS)

    # Arbitrary, meaningless integer codes for the 13 occupant kinds.
    codes = rng.permutation(13).tolist()
    symbol_of: dict = {None: codes[0]}
    for i, key in enumerate(_PIECE_KINDS, start=1):
        symbol_of[key] = codes[i]

    # Ground-truth answer keys.
    true_coords = np.empty((N_CELLS, 2), dtype=np.float64)
    for data_idx in range(N_CELLS):
        sq = int(cell_perm[data_idx])
        true_coords[data_idx] = (chess.square_file(sq), chess.square_rank(sq))

    true_piece_type = {symbol_of[None]: None}
    true_color = {symbol_of[None]: None}
    for (pt, color), _ in zip(_PIECE_KINDS, range(12)):
        s = symbol_of[(pt, color)]
        true_piece_type[s] = pt
        true_color[s] = color

    return Codec(
        seed=seed,
        cell_perm=cell_perm,
        cell_perm_inv=cell_perm_inv,
        symbol_of=symbol_of,
        n_symbols=13,
        true_coords=true_coords,
        true_piece_type=true_piece_type,
        true_color=true_color,
        empty_symbol=symbol_of[None],
    )


PIECE_NAME = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
    None: "empty",
}
