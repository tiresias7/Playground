"""Roll out self-play games and collect opaque (position, move) observations.

The output :class:`Dataset` carries *only* opaque arrays plus a few bookkeeping
fields. Anything ground-truth (used for scoring) lives on the :class:`Codec`,
not here.
"""
from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np

from .bots import ALL_BOTS
from .encoding import Codec, N_CELLS


@dataclass
class Dataset:
    states: np.ndarray      # (n, 64) opaque occupant symbols
    from_idx: np.ndarray    # (n,) source cell index
    to_idx: np.ndarray      # (n,) target cell index
    bot_id: np.ndarray      # (n,) which bot generated the pair
    bot_names: list         # bot_id -> name
    fens: list | None = None  # ground-truth FEN per row (eval only), or None

    def __len__(self):
        return len(self.from_idx)

    def subset(self, mask) -> "Dataset":
        idx = np.nonzero(mask)[0] if mask.dtype == bool else mask
        return Dataset(
            states=self.states[idx],
            from_idx=self.from_idx[idx],
            to_idx=self.to_idx[idx],
            bot_id=self.bot_id[idx],
            bot_names=self.bot_names,
            fens=[self.fens[i] for i in idx] if self.fens is not None else None,
        )


def _play_game(bot, codec, rng, max_plies, opening_plies, record_fens):
    """Return a list of (state_vec, from_idx, to_idx, fen|None) for one game."""
    board = chess.Board()
    # Random opening to diversify the visited positions.
    for _ in range(opening_plies):
        moves = list(board.legal_moves)
        if not moves:
            return []
        board.push(moves[int(rng.integers(len(moves)))])

    out = []
    for _ in range(max_plies):
        if board.is_game_over():
            break
        state = codec.encode_board(board)
        move = bot.choose(board)
        if move is None:
            break
        f, t = codec.encode_move(move)
        out.append((state, f, t, board.fen() if record_fens else None))
        board.push(move)
    return out


def generate(
    codec: Codec,
    bot_names: list[str],
    games_per_bot: int = 60,
    max_plies: int = 50,
    opening_plies: int = 4,
    seed: int = 0,
    record_fens: bool = False,
) -> Dataset:
    """Generate a pooled dataset from several bots."""
    rng = np.random.default_rng(seed)
    states, fro, to, bid, fens = [], [], [], [], []

    for b_index, name in enumerate(bot_names):
        bot = ALL_BOTS[name](seed=int(rng.integers(1 << 30)))
        for _ in range(games_per_bot):
            game = _play_game(
                bot, codec, rng, max_plies, opening_plies, record_fens
            )
            for state, f, t, fen in game:
                states.append(state)
                fro.append(f)
                to.append(t)
                bid.append(b_index)
                fens.append(fen)

    return Dataset(
        states=np.asarray(states, dtype=np.int64).reshape(-1, N_CELLS),
        from_idx=np.asarray(fro, dtype=np.int64),
        to_idx=np.asarray(to, dtype=np.int64),
        bot_id=np.asarray(bid, dtype=np.int64),
        bot_names=list(bot_names),
        fens=fens if record_fens else None,
    )
