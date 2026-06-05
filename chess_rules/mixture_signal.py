"""Train on noise-contaminated data; target = the PURE capture strategy.

The faithful version of the question: training data is alpha*capture +
(1-alpha)*uniform, but the TARGET is to recover the *pure* capture bot. Two
things we want to know:

  1. Recovery: how well does a model trained on noisy data predict the pure
     capture bot's moves (as % of the capture ceiling), vs alpha?

  2. Self-awareness: can the MODEL tell when the signal is insignificant,
     WITHOUT being told alpha? We answer this with a bootstrap: train K models
     on independent data samples and look at the spread of the capture
     preference they detect. If the bootstrap confidence interval includes
     chance (0.5), the model itself reports "not significant"; if it excludes
     chance, the signal is real. No ground-truth alpha needed.

Run: python -m chess_rules.mixture_signal
"""
from __future__ import annotations

import chess
import numpy as np
from sklearn.metrics import roc_auc_score

from .bots import CaptureBot
from .encoding import make_codec
from .mixture import MixtureBot, _rollout
from .strategy import train_strategy


def _capture_eval_arrays(ev, codec):
    """Per pure-capture-eval position: legal candidate (state,f,t), is_capture,
    and the move the capture bot actually played."""
    rows = []
    for i in range(len(ev)):
        board = chess.Board(ev.fens[i])
        cand = []
        for m in board.legal_moves:
            fi, ti = codec.encode_move(m)
            cand.append((fi, ti, 1 if board.is_capture(m) else 0))
        rows.append((ev.states[i], cand,
                     (int(ev.from_idx[i]), int(ev.to_idx[i]))))
    return rows


def _detect_and_predict(sclf, sfb, rows):
    """Return (capture-detection AUC, top1 accuracy vs the capture bot's move)."""
    all_states, f, t, iscap = [], [], [], []
    hits = []
    for state, cand, played in rows:
        fa = np.array([c[0] for c in cand]); ta = np.array([c[1] for c in cand])
        st = np.tile(state, (len(cand), 1))
        sc = sclf.predict_proba(sfb.build(st, fa, ta))[:, 1]
        hits.append((int(fa[sc.argmax()]), int(ta[sc.argmax()])) == played)
        all_states.append(sc)  # store scores
        iscap.extend(c[2] for c in cand)
        f.extend(fa.tolist())
    scores = np.concatenate(all_states)
    iscap = np.array(iscap)
    auc = roc_auc_score(iscap, scores) if 0 < iscap.sum() < len(iscap) else float("nan")
    return auc, float(np.mean(hits))


def run():
    rng = np.random.default_rng(0)
    codec = make_codec(seed=0)
    grid = codec.true_coords
    alphas = [0.0, 0.05, 0.1, 0.2, 0.5]
    K = 4  # bootstrap models per alpha

    # Pure capture target + its ceiling.
    pure = _rollout(CaptureBot(seed=99), codec, np.random.default_rng(7), games=20)
    rows = _capture_eval_arrays(pure, codec)
    capbot = CaptureBot(seed=0)
    ceil = np.mean([capbot.policy(chess.Board(f))[1].max()
                    for f in pure.fens if capbot.policy(chess.Board(f))[1].size > 1])

    print(f"target = PURE capture bot (ceiling top1 = {ceil:.3f}); "
          f"{K} bootstrap models per alpha\n")
    print(f"{'alpha':>6} {'capture-detect AUC (mean±2sd)':>30} {'significant?':>13} "
          f"{'predict-capture top1':>21} {'%ceil':>7}")
    for a in alphas:
        aucs, tops = [], []
        for k in range(K):
            train = _rollout(MixtureBot(a, seed=100 + k), codec,
                             np.random.default_rng(100 + k), games=30)
            sclf, sfb = train_strategy(train, grid, codec, seed=0, max_positions=1500)
            auc, top1 = _detect_and_predict(sclf, sfb, rows)
            aucs.append(auc); tops.append(top1)
        aucs = np.array(aucs); tops = np.array(tops)
        lo = aucs.mean() - 2 * aucs.std()
        sig = "YES" if lo > 0.5 else "no"
        print(f"{a:>6.2f} {aucs.mean():>20.3f} ± {2*aucs.std():.3f} {sig:>13} "
              f"{tops.mean():>21.3f} {tops.mean()/ceil*100:>6.0f}%")

    print("\nReading:")
    print("- predict-capture top1 / %ceil: recovery of the pure capture strategy from")
    print("  noisy training. Falls toward chance as alpha shrinks.")
    print("- significant? comes from the bootstrap spread ALONE (no alpha needed):")
    print("  the model itself flags when the capture signal is/ isn't real.")


if __name__ == "__main__":
    run()
