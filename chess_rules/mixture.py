"""Mixing a strategist into noise: when does the signal survive?

A MixtureBot plays  policy = alpha * capture_policy + (1 - alpha) * uniform.
At alpha=0 it is pure random; at alpha=1 pure capture. We sweep alpha and ask:

  * Rules (legality): does recovery survive at every alpha? (It should -- the
    random component only broadens the legal support.)
  * Strategy signal: can we still detect the capture preference buried in noise,
    and at what alpha does it vanish into the sampling noise?
  * Overfitting: does the strategy model hallucinate a capture signal on training
    data that does not hold out? (Compare train vs held-out detection.)

Run: python -m chess_rules.mixture
"""
from __future__ import annotations

import chess
import numpy as np
from sklearn.metrics import roc_auc_score

from .bots import CaptureBot, RandomBot
from .encoding import N_CELLS, make_codec
from .datagen import Dataset
from .legality import train_legality, FeatureBuilder, _mode_symbol
from .sanity import _candidates
from .strategy import train_strategy


class MixtureBot:
    """alpha * capture + (1-alpha) * uniform, as a true policy mixture."""

    def __init__(self, alpha, seed=0):
        self.alpha = alpha
        self.cap = CaptureBot(seed=seed)
        self.rng = np.random.default_rng(seed)

    def policy(self, board):
        moves, p_cap = self.cap.policy(board)
        if not moves:
            return [], np.array([])
        p = self.alpha * p_cap + (1 - self.alpha) * (1.0 / len(moves))
        return moves, p / p.sum()

    def choose(self, board):
        moves, p = self.policy(board)
        if not moves:
            return None
        return moves[int(self.rng.choice(len(moves), p=p))]


def _rollout(bot, codec, rng, games, max_plies=50, opening=4):
    states, fro, to, fens = [], [], [], []
    for _ in range(games):
        board = chess.Board()
        for _ in range(opening):
            ms = list(board.legal_moves)
            if not ms:
                break
            board.push(ms[int(rng.integers(len(ms)))])
        for _ in range(max_plies):
            if board.is_game_over():
                break
            st = codec.encode_board(board)
            mv = bot.choose(board)
            if mv is None:
                break
            f, t = codec.encode_move(mv)
            states.append(st); fro.append(f); to.append(t); fens.append(board.fen())
            board.push(mv)
    return Dataset(np.array(states).reshape(-1, N_CELLS), np.array(fro),
                   np.array(to), np.zeros(len(fro), int), ["mix"], fens)


def _legality_auc(clf, fb, ev, codec, rng, n=300):
    si, f, t, y = [], [], [], []
    for i in rng.choice(len(ev), size=min(n, len(ev)), replace=False):
        cands, _ = _candidates(ev.fens[i], rng, "self_capture", n=8)
        if not any(l == 0 for _, _, l in cands):
            continue
        for sf, st, l in cands:
            si.append(i); f.append(int(codec.cell_perm_inv[sf]))
            t.append(int(codec.cell_perm_inv[st])); y.append(l)
    if len(set(y)) < 2:
        return float("nan")
    p = clf.predict_proba(fb.build(ev.states[np.array(si)], np.array(f), np.array(t)))[:, 1]
    return roc_auc_score(y, p)


def _capture_signal(sclf, sfb, ds, codec, rng, n=300):
    """AUC of the strategy model's score at picking out capture moves among legal
    moves. 0.5 => no capture preference detected."""
    si, f, t, iscap = [], [], [], []
    for i in rng.choice(len(ds), size=min(n, len(ds)), replace=False):
        board = chess.Board(ds.fens[i])
        for m in board.legal_moves:
            si.append(i)
            f.append(int(codec.cell_perm_inv[m.from_square]))
            t.append(int(codec.cell_perm_inv[m.to_square]))
            iscap.append(1 if board.is_capture(m) else 0)
    iscap = np.array(iscap)
    if iscap.sum() == 0 or iscap.sum() == len(iscap):
        return float("nan")
    p = sclf.predict_proba(sfb.build(ds.states[np.array(si)], np.array(f), np.array(t)))[:, 1]
    return roc_auc_score(iscap, p)


def run():
    rng = np.random.default_rng(0)
    codec = make_codec(seed=0)
    grid = codec.true_coords
    alphas = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]

    _log = lambda m: print(m, flush=True)
    _log("alpha = fraction capture-policy (rest uniform random)\n")
    _log(f"{'alpha':>6} {'data capture%':>13} {'rand baseline%':>15} "
         f"{'legality AUC':>13} {'capture signal AUC':>19} {'(train)':>9}")
    for a in alphas:
        bot = MixtureBot(a, seed=1)
        train = _rollout(bot, codec, rng, games=40)
        ev = _rollout(MixtureBot(a, seed=2), codec, rng, games=18)

        # empirical capture rate in data vs the uniform-random baseline
        cap_rate, base = _empirical_capture(train, codec)

        clf, fb = train_legality(train, grid, seed=0)
        leg = _legality_auc(clf, fb, ev, codec, rng)

        sclf, sfb = train_strategy(train, grid, codec, seed=0, max_positions=2500)
        sig_ev = _capture_signal(sclf, sfb, ev, codec, rng)
        sig_tr = _capture_signal(sclf, sfb, train, codec, rng)

        _log(f"{a:>6.2f} {cap_rate*100:>12.1f}% {base*100:>14.1f}% "
             f"{leg:>13.3f} {sig_ev:>19.3f} {sig_tr:>9.3f}")

    print("\nReading: legality AUC should stay high for all alpha (random part only")
    print("broadens support). capture-signal AUC -> 0.5 as the preference drowns in")
    print("noise; gap between (train) and held-out flags overfitting to noise.")


def _empirical_capture(ds, codec):
    cap, base = 0, []
    for i in range(len(ds)):
        board = chess.Board(ds.fens[i])
        legal = list(board.legal_moves)
        ncap = sum(board.is_capture(m) for m in legal)
        base.append(ncap / max(len(legal), 1))
        mv = chess.Move(int(codec.cell_perm[ds.from_idx[i]]),
                        int(codec.cell_perm[ds.to_idx[i]]))
        if board.is_capture(mv):
            cap += 1
    return cap / len(ds), float(np.mean(base))


if __name__ == "__main__":
    run()
