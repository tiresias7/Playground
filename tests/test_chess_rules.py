"""Tests for the chess-rule-recovery experiment.

These run a tiny pipeline and assert the recovery signals are clearly present,
plus they guard the central honesty property: the learner modules never import
ground-truth knowledge (geometry / piece identities / python-chess).
"""
import numpy as np
import pytest

from chess_rules import datagen, geometry, legality, pieces, strategy
from chess_rules.encoding import make_codec


@pytest.fixture(scope="module")
def small_data():
    codec = make_codec(seed=1)
    pool = datagen.generate(
        codec, ["random", "capture", "minimax"], games_per_bot=8, seed=1
    )
    return codec, pool


def test_encoding_is_opaque_and_reversible():
    codec = make_codec(seed=3)
    import chess

    board = chess.Board()
    state = codec.encode_board(board)
    assert state.shape == (64,)
    # Most of a starting board is empty -> background symbol dominates.
    assert (state == codec.empty_symbol).sum() == 32
    # A known move maps to the permuted cell indices.
    move = chess.Move.from_uci("e2e4")
    f, t = codec.encode_move(move)
    assert f == codec.cell_perm_inv[chess.E2]
    assert t == codec.cell_perm_inv[chess.E4]


def test_geometry_recovered(small_data):
    codec, pool = small_data
    coords = geometry.embed(pool.from_idx, pool.to_idx)
    m = geometry.evaluate_geometry(coords, codec.true_coords)
    # Global layout aligns well to the true grid; local adjacency mostly right.
    assert m["procrustes_disparity"] < 0.25
    assert m["neighbor_preservation"] > 0.45


def test_piece_movement_rules_recovered(small_data):
    codec, pool = small_data
    m = pieces.evaluate_pieces(pool, codec.true_coords, codec)
    # Observed moves are always legal for the true piece (bots play legal moves).
    assert m["displacement_precision"] > 0.99
    # Symbols cluster cleanly into piece kinds purely from movement.
    assert m["movement_cluster_ARI"] > 0.6


def test_legality_support_recovered(small_data):
    codec, pool = small_data
    eval_ds = datagen.generate(
        codec, ["random", "capture"], games_per_bot=4, seed=5, record_fens=True
    )
    clf, fb = legality.train_legality(pool, codec.true_coords, seed=0)
    m = legality.evaluate_legality(clf, fb, eval_ds, codec, seed=0, max_positions=60)
    # Legal vs illegal moves are separated far above chance.
    assert m["roc_auc"] > 0.9


def test_rules_invariant_strategy_specific(small_data):
    """The disentanglement claim, in miniature."""
    codec, _ = small_data
    a = datagen.generate(codec, ["capture"], games_per_bot=12, seed=10, record_fens=True)
    b = datagen.generate(codec, ["mobility"], games_per_bot=12, seed=11, record_fens=True)

    # Legality learned on bot A transfers to bot B (rules are shared).
    clf, fb = legality.train_legality(a, codec.true_coords, seed=0)
    leg_b = legality.evaluate_legality(clf, fb, b, codec, seed=0, max_positions=60)
    assert leg_b["roc_auc"] > 0.9

    # Strategy does not transfer as well: A's model predicts A better than B.
    sclf, sfb = strategy.train_strategy(a, codec.true_coords, codec, seed=0, max_positions=300)
    own = strategy.top1_accuracy(sclf, sfb, a, codec, max_positions=120, seed=0)
    cross = strategy.top1_accuracy(sclf, sfb, b, codec, max_positions=120, seed=0)
    assert own >= cross


def test_hard_negative_generator_produces_truly_illegal_moves():
    """sanity.py's hard negatives must actually be illegal (labels are correct)."""
    import chess
    from chess_rules import sanity

    rng = np.random.default_rng(0)
    # A position with pins and self-capture possibilities.
    fen = "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"
    for kind in ("random", "self_capture", "pin_check"):
        cands, board = sanity._candidates(fen, rng, kind)
        legal = {(m.from_square, m.to_square) for m in board.legal_moves}
        for f, t, lab in cands:
            assert lab == (1 if (f, t) in legal else 0)


def test_learner_modules_do_not_import_ground_truth():
    """Firewall: the recovery code must not peek at chess or answer keys."""
    import inspect

    # geometry's recovery routine works on raw move indices only.
    assert "import chess" not in inspect.getsource(geometry)
    embed_src = inspect.getsource(geometry.embed)
    assert "true_coords" not in embed_src
    assert "codec" not in embed_src

    # The piece fingerprints are built without piece identities or python-chess.
    fp_src = inspect.getsource(pieces.symbol_fingerprints) + inspect.getsource(pieces._fingerprint)
    for banned in ("import chess", "true_piece_type", "true_color", "PIECE_NAME"):
        assert banned not in fp_src
