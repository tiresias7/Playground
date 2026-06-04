"""Scope of the claims: ceilings, the bot sub-universe, and rule-vs-subset.

Three things the review (rightly) demanded:

  A. The bot move-universe is a *biased subset* of the legal universe. Quantify
     the bias: how much of the legal support each bot actually uses, and how
     peaked / fingerprinted its policy is.

  B. A *ceiling* for move prediction. The bots sample moves stochastically, so
     even a perfect model of bot X cannot beat X's own modal-move probability
     E[max_m P_X(m)]. We get this by calling the bots directly. Our model's
     top-1 only means something relative to this ceiling.

  C. "Trained on bot X, separates X's moves from others" does NOT prove the
     model learned the *rule* — X's moves are a strict subset, so the model
     might learn the subset. The honest test: train legality on ONE narrow bot,
     then measure recall on legal moves that bot *disfavors* (rarely/never
     plays). High recall there = it learned the rule, not the strategy subset.

Run: python -m chess_rules.scope
"""
from __future__ import annotations

import chess
import numpy as np
from sklearn.metrics import roc_auc_score

from . import datagen
from .bots import ALL_BOTS
from .encoding import make_codec
from .legality import train_legality
from .strategy import train_strategy, _legal_candidates

BOTS = ["random", "capture", "center", "mobility", "minimax"]


def _log(m):
    print(m, flush=True)


def _entropy(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def run():
    rng = np.random.default_rng(0)
    codec = make_codec(seed=0)
    grid = codec.true_coords

    _log("generating per-bot eval data (with FENs) ...")
    per_eval = {b: datagen.generate(codec, [b], games_per_bot=20,
                                    seed=300 + i, record_fens=True)
                for i, b in enumerate(BOTS)}

    # =====================================================================
    # A. Sub-universe bias + B. move-prediction ceiling (vs our model)
    # =====================================================================
    _log("\n=== A/B. Bot sub-universe & move-prediction ceiling ===")
    _log("    eff.support = 2^H / branching (1.0=uniform over all legal, small=narrow)")
    _log("    ceiling = E[max_m P_bot(m)] = best ANY predictor can do (labels are")
    _log("    samples from P_bot); model = our per-bot model; %ceil = model/ceiling")
    _log(f"\n    {'bot':9s} {'branch':>7} {'H':>5} {'eff.supp':>9} "
         f"{'ceiling':>8} {'model':>7} {'%ceil':>7}")
    for b in BOTS:
        bot = ALL_BOTS[b](seed=0)
        train_b = datagen.generate(codec, [b], games_per_bot=45, seed=11,
                                   record_fens=True)
        sclf, sfb = train_strategy(train_b, grid, codec, seed=0, max_positions=3500)

        fens = per_eval[b].fens
        states = per_eval[b].states
        pick = rng.choice(len(fens), size=min(250, len(fens)), replace=False)
        branch, ent, eff, ceil, hit = [], [], [], [], []
        for i in pick:
            board = chess.Board(fens[i])
            moves, p = bot.policy(board)
            if len(moves) < 2:
                continue
            branch.append(len(moves)); ent.append(_entropy(p))
            eff.append((2 ** _entropy(p)) / len(moves)); ceil.append(p.max())
            cands = _legal_candidates(fens[i], codec)
            st = np.tile(states[i], (len(cands), 1))
            fa = np.array([c[0] for c in cands]); ta = np.array([c[1] for c in cands])
            sc = sclf.predict_proba(sfb.build(st, fa, ta))[:, 1]
            hit.append(cands[int(np.argmax(sc))] ==
                       (int(per_eval[b].from_idx[i]), int(per_eval[b].to_idx[i])))
        mc = np.mean(ceil); mm = np.mean(hit)
        _log(f"    {b:9s} {np.mean(branch):>7.1f} {np.mean(ent):>5.2f} "
             f"{np.mean(eff):>9.2f} {mc:>8.3f} {mm:>7.3f} {mm/mc*100:>6.0f}%")

    # =====================================================================
    # C. Rule vs. strategy-subset: train legality on ONE narrow bot
    # =====================================================================
    _log("\n=== C. Did legality learn the RULE or just bot X's subset? ===")
    _log("    Train legality on one bot only; test recall on legal moves that")
    _log("    bot DISFAVORS. If it only learned the subset, recall drops there.")
    for train_bot in ("capture", "minimax", "random"):
        _rule_vs_subset(codec, grid, train_bot, per_eval, rng)


def _rule_vs_subset(codec, grid, train_bot, per_eval, rng):
    train = datagen.generate(codec, [train_bot], games_per_bot=45, seed=7,
                             record_fens=True)
    clf, fb = train_legality(train, grid, seed=0)
    bot = ALL_BOTS[train_bot](seed=0)

    # Evaluate on the SAME bot's positions: stratify legal moves by how much the
    # bot favors them, and measure recall (called legal at p>=0.5) per stratum.
    fens = per_eval[train_bot].fens
    states = per_eval[train_bot].states
    pick = rng.choice(len(fens), size=min(400, len(fens)), replace=False)

    # quartiles of bot-probability among legal moves
    q_recall = {0: [], 1: [], 2: [], 3: []}
    never_recall = []  # bot prob below 1/(4*branch): "effectively never plays"
    for i in pick:
        board = chess.Board(fens[i])
        moves, p = bot.policy(board)
        if len(moves) < 4:
            continue
        f = np.array([codec.cell_perm_inv[m.from_square] for m in moves])
        t = np.array([codec.cell_perm_inv[m.to_square] for m in moves])
        st = np.tile(states[i], (len(moves), 1))
        pr = clf.predict_proba(fb.build(st, f, t))[:, 1]
        called_legal = (pr >= 0.5)
        # rank legal moves into quartiles by bot preference
        order = np.argsort(p)  # ascending preference
        quart = np.floor(np.argsort(order) / len(moves) * 4).astype(int).clip(0, 3)
        for q in range(4):
            m = quart == q
            if m.any():
                q_recall[q].extend(called_legal[m].tolist())
        rare = p < (1.0 / (4 * len(moves)))
        if rare.any():
            never_recall.extend(called_legal[rare].tolist())

    _log(f"\n    [{train_bot}] legal-move recall by how much the bot favors the move:")
    _log(f"      least-favored quartile : {np.mean(q_recall[0]):.3f}  "
         f"(n={len(q_recall[0])})")
    _log(f"      2nd quartile           : {np.mean(q_recall[1]):.3f}")
    _log(f"      3rd quartile           : {np.mean(q_recall[2]):.3f}")
    _log(f"      most-favored quartile  : {np.mean(q_recall[3]):.3f}  "
         f"(n={len(q_recall[3])})")
    if never_recall:
        _log(f"      'effectively never played' legal moves recognised as legal: "
             f"{np.mean(never_recall):.3f}  (n={len(never_recall)})")
    gap = np.mean(q_recall[3]) - np.mean(q_recall[0])
    _log(f"      favored-minus-disfavored recall gap: {gap:+.3f}  "
         f"(~0 => learned the RULE, large+ => learned the subset)")


if __name__ == "__main__":
    run()
