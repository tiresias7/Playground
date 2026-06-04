"""Bots that *generate* the observations.

Each bot picks a move from ``board.legal_moves`` according to its own strategy.
Crucially, every bot is constrained to the *same* legal set (the chess rules),
but each weights that set differently (its strategy). The learner never sees
these bots or their internals -- only the (position, move) pairs they emit.

The rules-vs-strategy experiment hinges on this contrast: legality is the part
of the data-generating process that is invariant across all the bots, while the
move *preference* is the part that varies from bot to bot.
"""
from __future__ import annotations

import chess
import numpy as np

_PIECE_VALUE = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.2,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}

# Distance of each square from the board centre; smaller is more central.
_CENTER = np.array([3.5, 3.5])


def _material(board: chess.Board, color: bool) -> float:
    total = 0.0
    for pt, val in _PIECE_VALUE.items():
        total += val * len(board.pieces(pt, color))
    return total


class Bot:
    name = "bot"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def scores(self, board: chess.Board, moves: list[chess.Move]) -> np.ndarray:
        raise NotImplementedError

    def choose(self, board: chess.Board) -> chess.Move | None:
        moves = list(board.legal_moves)
        if not moves:
            return None
        s = self.scores(board, moves)
        # Softmax-ish stochastic choice so the support is explored, not just argmax.
        s = s - s.max()
        p = np.exp(s / self.temperature)
        p /= p.sum()
        return moves[int(self.rng.choice(len(moves), p=p))]

    temperature = 1.0


class RandomBot(Bot):
    """Uniform over legal moves -- explores the legal support most broadly."""

    name = "random"
    temperature = 1.0

    def scores(self, board, moves):
        return np.zeros(len(moves))


class CaptureBot(Bot):
    """Greedy for the most valuable capture; otherwise indifferent."""

    name = "capture"
    temperature = 0.5

    def scores(self, board, moves):
        out = np.zeros(len(moves))
        for i, m in enumerate(moves):
            victim = board.piece_at(m.to_square)
            if victim is not None:
                out[i] = _PIECE_VALUE[victim.piece_type]
        return out


class CenterBot(Bot):
    """Prefers moving pieces toward the centre of the board."""

    name = "center"
    temperature = 0.5

    def scores(self, board, moves):
        out = np.zeros(len(moves))
        for i, m in enumerate(moves):
            f, r = chess.square_file(m.to_square), chess.square_rank(m.to_square)
            out[i] = -np.hypot(f - _CENTER[0], r - _CENTER[1])
        return out


class MobilityBot(Bot):
    """Prefers moves that leave it with many replies (maximise own mobility)."""

    name = "mobility"
    temperature = 0.5

    def scores(self, board, moves):
        out = np.zeros(len(moves))
        for i, m in enumerate(moves):
            board.push(m)
            # After our move it is the opponent's turn; count *our* future replies
            # by peeking one ply further is expensive, so use opponent reply count
            # as a cheap proxy and negate (fewer opponent replies = better).
            out[i] = -board.legal_moves.count()
            board.pop()
        return out


class MaterialMinimaxBot(Bot):
    """Depth-2 negamax on material -- the closest thing to a 'sharp' strategist."""

    name = "minimax"
    temperature = 0.3

    def __init__(self, seed: int = 0, depth: int = 2):
        super().__init__(seed)
        self.depth = depth

    def _eval(self, board: chess.Board, root_color: bool) -> float:
        return _material(board, root_color) - _material(board, not root_color)

    def scores(self, board, moves):
        root_color = board.turn
        out = np.zeros(len(moves))
        for i, m in enumerate(moves):
            board.push(m)
            out[i] = self._negamax(board, self.depth - 1, root_color)
            board.pop()
        return out

    def _negamax(self, board, depth, root_color):
        if depth == 0 or board.is_game_over():
            return self._eval(board, root_color)
        # Minimise the opponent's best material outcome.
        best = None
        for m in board.legal_moves:
            board.push(m)
            val = self._negamax(board, depth - 1, root_color)
            board.pop()
            if board.turn == root_color:
                best = val if best is None else max(best, val)
            else:
                best = val if best is None else min(best, val)
        return best if best is not None else self._eval(board, root_color)


ALL_BOTS = {
    "random": RandomBot,
    "capture": CaptureBot,
    "center": CenterBot,
    "mobility": MobilityBot,
    "minimax": MaterialMinimaxBot,
}
