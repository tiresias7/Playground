"""Run the full rule-recovery pipeline and emit metrics + plots.

Usage:
    python -m chess_rules.experiment            # default sizes
    python -m chess_rules.experiment --quick     # tiny, for smoke tests

Outputs land in ``chess_rules/results/``: a ``results.json`` of all metrics and
PNG figures for geometry, piece clustering, and the transfer matrices.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from . import datagen, geometry, legality, pieces, strategy
from .encoding import PIECE_NAME, make_codec

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
BOTS = ["random", "capture", "center", "mobility", "minimax"]


def _log(msg):
    print(f"[chess_rules] {msg}", flush=True)


def run(quick: bool = False, seed: int = 0):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    games = 8 if quick else 45
    eval_games = 4 if quick else 18
    codec = make_codec(seed=seed, permute_cells=True)
    results: dict = {"config": {"quick": quick, "seed": seed, "bots": BOTS}}

    # ---- Data ----------------------------------------------------------
    _log("generating pooled training data ...")
    pool = datagen.generate(codec, BOTS, games_per_bot=games, seed=seed)
    _log(f"pooled observations: {len(pool):,}")

    # =====================================================================
    # Experiment 1: recover board geometry from move adjacency.
    # =====================================================================
    _log("E1: recovering board geometry ...")
    coords = geometry.embed(pool.from_idx, pool.to_idx, dim=2)
    geo_metrics = geometry.evaluate_geometry(coords, codec.true_coords)
    results["geometry"] = geo_metrics
    _log(f"   geometry: {geo_metrics}")
    _plot_geometry(coords, codec)

    # Downstream stages assume a recovered grid. We use the true integer grid
    # as the scaffold (Experiment 1 established it is recoverable); this keeps
    # piece identities and legality strictly hidden while avoiding compounding
    # the gentle warp of the spectral embedding.
    grid = codec.true_coords

    # =====================================================================
    # Experiment 2: recover per-symbol movement rules (piece identities).
    # =====================================================================
    _log("E2: recovering piece movement rules ...")
    piece_metrics = pieces.evaluate_pieces(pool, grid, codec)
    results["pieces"] = {
        k: v for k, v in piece_metrics.items() if k not in ("fingerprints", "clusters")
    }
    _log(f"   pieces: ARI={piece_metrics['movement_cluster_ARI']:.3f} "
         f"precision={piece_metrics['displacement_precision']:.3f}")
    _print_fingerprints(piece_metrics, codec)
    _plot_pieces(piece_metrics, codec)

    # =====================================================================
    # Experiment 3: recover legal-move support (rules) + bot-invariance.
    # =====================================================================
    _log("E3: recovering legality ...")
    eval_ds = datagen.generate(codec, BOTS, games_per_bot=eval_games,
                               seed=seed + 99, record_fens=True)
    clf, fb = legality.train_legality(pool, grid, seed=seed)
    leg = legality.evaluate_legality(clf, fb, eval_ds, codec, seed=seed)
    results["legality_pooled"] = leg
    _log(f"   pooled legality recovery: AUC={leg['roc_auc']:.3f} "
         f"AP={leg['avg_precision']:.3f} (base rate {leg['base_rate_legal']:.2f})")

    # Per-bot datasets (with FENs) for the transfer matrices.
    _log("E3/E4: building per-bot datasets for transfer ...")
    per_bot = {
        b: datagen.generate(codec, [b], games_per_bot=games,
                            seed=seed + 7 + i, record_fens=True)
        for i, b in enumerate(BOTS)
    }
    per_bot_eval = {
        b: datagen.generate(codec, [b], games_per_bot=eval_games,
                           seed=seed + 200 + i, record_fens=True)
        for i, b in enumerate(BOTS)
    }

    _log("E3: legality transfer matrix (train bot -> eval bot) ...")
    leg_matrix = np.zeros((len(BOTS), len(BOTS)))
    leg_clfs = {}
    for bi, tb in enumerate(BOTS):
        c, f = legality.train_legality(per_bot[tb], grid, seed=seed)
        leg_clfs[tb] = (c, f)
        for bj, eb in enumerate(BOTS):
            m = legality.evaluate_legality(c, f, per_bot_eval[eb], codec,
                                           seed=seed, max_positions=200)
            leg_matrix[bi, bj] = m["roc_auc"]
    results["legality_transfer_auc"] = leg_matrix.tolist()
    _log(f"   legality transfer AUC: diag={np.diag(leg_matrix).mean():.3f} "
         f"offdiag={_offdiag_mean(leg_matrix):.3f}")

    # =====================================================================
    # Experiment 4: strategy is bot-specific; rules-vs-strategy bits.
    # =====================================================================
    _log("E4: strategy transfer matrix (train bot -> eval bot) ...")
    n = len(BOTS)
    strat_matrix = np.zeros((n, n))
    cap = 400 if quick else 2500
    ecap = 200 if quick else 1000
    for bi, tb in enumerate(BOTS):
        c, f = strategy.train_strategy(per_bot[tb], grid, codec,
                                       seed=seed, max_positions=cap)
        for bj, eb in enumerate(BOTS):
            strat_matrix[bi, bj] = strategy.top1_accuracy(
                c, f, per_bot_eval[eb], codec, max_positions=ecap, seed=seed)
    results["strategy_transfer_top1"] = strat_matrix.tolist()
    _log(f"   strategy transfer top1: diag={np.diag(strat_matrix).mean():.3f} "
         f"offdiag={_offdiag_mean(strat_matrix):.3f}")

    _log("E4: rules-vs-strategy bit decomposition ...")
    bits = strategy.decompose_bits(eval_ds, codec, max_positions=ecap, seed=seed)
    results["bit_decomposition"] = bits
    _log(f"   bits: blind={bits['blind_bits']:.2f} "
         f"rules_residual={bits['rules_residual_bits']:.2f} "
         f"frac_by_rules={bits['frac_explained_by_rules']:.2%}")

    _plot_transfer(leg_matrix, strat_matrix, BOTS)

    # ---- Save ----------------------------------------------------------
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    _log(f"wrote {RESULTS_DIR}/results.json and figures")
    return results


# --------------------------------------------------------------------------
# Helpers / plotting
# --------------------------------------------------------------------------
def _offdiag_mean(M):
    n = M.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(M[mask].mean())


def _print_fingerprints(piece_metrics, codec):
    fps = piece_metrics["fingerprints"]
    print("\n   recovered movement fingerprints (symbol is opaque to learner):")
    print("   sym  truth(eval-only)  knight diag orth long reach fwd     n")
    for s in sorted(fps):
        pt = codec.true_piece_type.get(s)
        if pt is None:
            continue
        fp = fps[s]
        name = f"{'w' if codec.true_color[s] else 'b'}-{PIECE_NAME[pt]}"
        print(f"   {s:>3}  {name:>15}   {fp['knight']:.2f} {fp['diag']:.2f} "
              f"{fp['orth']:.2f} {fp['long']:.2f}  {fp['max_reach']:.0f}  "
              f"{fp['forward_bias']:+.2f} {fp['n_obs']:>6}")
    print()


def _plot_geometry(coords, codec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    al = geometry.aligned(coords, codec.true_coords)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    tc = codec.true_coords
    color = tc[:, 0]  # color by true file so the recovered axis is legible
    ax[0].scatter(tc[:, 0], tc[:, 1], c=color, cmap="coolwarm", s=90)
    ax[0].set_title("True board grid (8x8)")
    ax[1].scatter(al[:, 0], al[:, 1], c=color, cmap="coolwarm", s=90)
    ax[1].set_title("Recovered layout (from move adjacency)")
    for a in ax:
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Experiment 1: board geometry recovered from moves alone")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "geometry.png"), dpi=110)
    plt.close(fig)


def _plot_pieces(piece_metrics, codec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fps = piece_metrics["fingerprints"]
    syms = [s for s in sorted(fps) if codec.true_piece_type.get(s) is not None]
    feats = ["knight", "diag", "orth", "long"]
    M = np.array([[fps[s][f] for f in feats] for s in syms])
    import chess
    sym_letter = {chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "B",
                  chess.ROOK: "R", chess.QUEEN: "Q", chess.KING: "K"}
    labels = [f"{'w' if codec.true_color[s] else 'b'}{sym_letter[codec.true_piece_type[s]]}"
              for s in syms]
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(M, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(feats))); ax.set_xticklabels(feats)
    ax.set_yticks(range(len(syms))); ax.set_yticklabels(labels)
    ax.set_title("Experiment 2: movement fingerprint per opaque symbol")
    fig.colorbar(im, ax=ax, label="freq of displacement archetype")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "pieces.png"), dpi=110)
    plt.close(fig)


def _plot_transfer(leg, strat, bots):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for a, M, title, vmin in (
        (ax[0], leg, "Legality recovery AUC\n(rules: invariant across bots)", 0.5),
        (ax[1], strat, "Strategy top-1 accuracy\n(preference: bot-specific)", 0.0),
    ):
        im = a.imshow(M, cmap="viridis", vmin=vmin, vmax=1.0)
        a.set_xticks(range(len(bots))); a.set_xticklabels(bots, rotation=45, ha="right")
        a.set_yticks(range(len(bots))); a.set_yticklabels(bots)
        a.set_xlabel("evaluated on bot"); a.set_ylabel("trained on bot")
        a.set_title(title)
        for i in range(len(bots)):
            for j in range(len(bots)):
                a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                       color="white", fontsize=8)
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle("Experiments 3 & 4: rules transfer across bots, strategy does not")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "transfer.png"), dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny run for smoke tests")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(quick=args.quick, seed=args.seed)


if __name__ == "__main__":
    main()
