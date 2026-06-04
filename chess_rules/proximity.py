"""Accuracy + train/eval *proximity* analysis.

Two things the skeptic actually wants:

  1. Concrete prediction accuracy (not just AUC) for legality and for the
     move/strategy task.

  2. Exact-match overlap is meaningless -- positions can be *near* a training
     position and thus effectively memorized. So we measure, for each eval
     position, the Hamming distance (number of differing cells) to its nearest
     training position, and ask whether accuracy *degrades with distance*. If
     accuracy is flat far from any training position, the model generalizes; if
     it only works near training data, it interpolates/memorizes.

Run: python -m chess_rules.proximity
"""
from __future__ import annotations

import chess
import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

from . import datagen
from .encoding import make_codec
from .legality import train_legality
from .sanity import _candidates
from .strategy import train_strategy, _legal_candidates

BOTS = ["random", "capture", "center", "mobility", "minimax"]


def _log(m):
    print(m, flush=True)


def _nearest_train_distance(eval_states, train_states, sample_idx):
    """Min Hamming distance (differing cells, 0..64) from each sampled eval
    state to ANY training state."""
    out = np.empty(len(sample_idx), dtype=int)
    for i, idx in enumerate(sample_idx):
        diffs = (train_states != eval_states[idx]).sum(axis=1)
        out[i] = int(diffs.min())
    return out


def run():
    rng = np.random.default_rng(0)
    codec = make_codec(seed=0)
    grid = codec.true_coords

    _log("generating data ...")
    train = datagen.generate(codec, BOTS, games_per_bot=45, seed=0, record_fens=True)
    ev = datagen.generate(codec, BOTS, games_per_bot=18, seed=99, record_fens=True)

    clf, fb = train_legality(train, grid, seed=0)
    sclf, sfb = train_strategy(train, grid, codec, seed=0, max_positions=4000)

    # =====================================================================
    # 1. CONCRETE ACCURACIES
    # =====================================================================
    _log("\n=== PREDICTION ACCURACY ===")
    _log("\n[Legality] accuracy at threshold 0.5 (and balanced acc, AUC):")
    for kind in ("random", "self_capture", "pin_check"):
        si, f, t, y = _candidates_batch(ev, codec, kind, rng, 500)
        if len(np.unique(y)) < 2:
            continue
        p = clf.predict_proba(fb.build(ev.states[si], f, t))[:, 1]
        pred = (p >= 0.5).astype(int)
        acc = (pred == y).mean()
        bacc = balanced_accuracy_score(y, pred)
        _log(f"    legal vs {kind:12s}: acc={acc:.3f}  balanced_acc={bacc:.3f}  "
             f"AUC={roc_auc_score(y, p):.3f}")

    _log("\n[Move / strategy] top-1 accuracy (predict the bot's move among legal):")
    chance, top1_all = [], []
    by_bot = {b: [] for b in BOTS}
    sample = rng.choice(len(ev), size=1500, replace=False)
    for i in sample:
        cands = _legal_candidates(ev.fens[i], codec)
        if not cands:
            continue
        states = np.tile(ev.states[i], (len(cands), 1))
        fa = np.array([c[0] for c in cands]); ta = np.array([c[1] for c in cands])
        scores = sclf.predict_proba(sfb.build(states, fa, ta))[:, 1]
        hit = cands[int(np.argmax(scores))] == (int(ev.from_idx[i]), int(ev.to_idx[i]))
        top1_all.append(hit)
        chance.append(1.0 / len(cands))
        by_bot[BOTS[ev.bot_id[i]]].append(hit)
    _log(f"    pooled top-1        : {np.mean(top1_all):.3f}  "
         f"(random-legal baseline {np.mean(chance):.3f})")
    for b in BOTS:
        if by_bot[b]:
            _log(f"    {b:9s} top-1     : {np.mean(by_bot[b]):.3f}  (n={len(by_bot[b])})")

    # =====================================================================
    # 2. PROXIMITY: distance to nearest training position
    # =====================================================================
    _log("\n=== TRAIN/EVAL PROXIMITY (Hamming distance over 64 cells) ===")
    nn_sample = rng.choice(len(ev), size=700, replace=False)
    nn_dist = _nearest_train_distance(ev.states, train.states, nn_sample)
    _log(f"    distance to NEAREST train position (0=identical, 64=fully different):")
    _log(f"    min={nn_dist.min()}  p5={np.percentile(nn_dist,5):.0f}  "
         f"median={np.median(nn_dist):.0f}  p95={np.percentile(nn_dist,95):.0f}  "
         f"max={nn_dist.max()}")
    _log(f"    eval positions within 2 cells of a train position: "
         f"{(nn_dist<=2).mean()*100:.1f}%")

    # Bin by distance, measure accuracy per bin.
    _log("\n    Does accuracy degrade with distance from training data?")
    _log("    (flat => generalization; falling => interpolation/memorization)")
    bins = [(0, 4), (5, 9), (10, 14), (15, 19), (20, 64)]
    _log(f"    {'NN-dist':>10} {'n':>5} {'legality acc':>13} {'move top-1':>11} "
         f"{'(move chance)':>13}")
    for lo, hi in bins:
        idxs = nn_sample[(nn_dist >= lo) & (nn_dist <= hi)]
        if len(idxs) < 15:
            continue
        leg_acc = _legality_acc_for(clf, fb, ev, codec, idxs, rng)
        mv_acc, mv_chance = _move_acc_for(sclf, sfb, ev, codec, idxs)
        _log(f"    {f'{lo}-{hi}':>10} {len(idxs):>5} {leg_acc:>13.3f} "
             f"{mv_acc:>11.3f} {mv_chance:>13.3f}")


# ---------------------------------------------------------------------------
def _candidates_batch(ev, codec, kind, rng, max_positions):
    si, f, t, y = [], [], [], []
    pick = rng.choice(len(ev), size=min(max_positions, len(ev)), replace=False)
    for i in pick:
        cands, _ = _candidates(ev.fens[i], rng, kind)
        if not any(lab == 0 for _, _, lab in cands):
            continue
        for sf, st, lab in cands:
            si.append(i); f.append(int(codec.cell_perm_inv[sf]))
            t.append(int(codec.cell_perm_inv[st])); y.append(lab)
    return np.array(si), np.array(f), np.array(t), np.array(y)


def _legality_acc_for(clf, fb, ev, codec, idxs, rng):
    """Balanced legality accuracy on legal + (random & self-capture) negatives."""
    si, f, t, y = [], [], [], []
    for i in idxs:
        for kind in ("random", "self_capture"):
            cands, _ = _candidates(ev.fens[i], rng, kind, n=8)
            for sf, st, lab in cands:
                si.append(i); f.append(int(codec.cell_perm_inv[sf]))
                t.append(int(codec.cell_perm_inv[st])); y.append(lab)
    si = np.array(si); f = np.array(f); t = np.array(t); y = np.array(y)
    p = clf.predict_proba(fb.build(ev.states[si], f, t))[:, 1]
    return balanced_accuracy_score(y, (p >= 0.5).astype(int))


def _move_acc_for(sclf, sfb, ev, codec, idxs):
    hits, chance = [], []
    for i in idxs:
        cands = _legal_candidates(ev.fens[i], codec)
        if not cands:
            continue
        states = np.tile(ev.states[i], (len(cands), 1))
        fa = np.array([c[0] for c in cands]); ta = np.array([c[1] for c in cands])
        scores = sclf.predict_proba(sfb.build(states, fa, ta))[:, 1]
        hits.append(cands[int(np.argmax(scores))] == (int(ev.from_idx[i]), int(ev.to_idx[i])))
        chance.append(1.0 / len(cands))
    return float(np.mean(hits)), float(np.mean(chance))


if __name__ == "__main__":
    run()
