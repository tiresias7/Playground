# Recovering hidden chess rules from observations alone

> Can a model infer the structure and rules of chess purely from observed
> `(position, move)` pairs, without ever being told the data is chess?

This is a thought experiment made runnable. Chess bots emit moves; we strip the
data down to generic numbers and ask how much of the **hidden generative
process** a learner can reconstruct from data alone.

```
observed move = f(chess_rules, board_state, bot_strategy)
the learner sees only:  (board_state, move)
```

The goal is **not** to play strong chess or mimic a bot. It is to measure how
much *latent structure* — geometry, piece movement, legality — can be recovered
when the true rules are hidden, and to separate the **rules** (legal
constraints, shared by every data source) from **strategy** (a bot's
preferences, specific to each source).

## The setup, and the firewall

Each observation is made deliberately generic:

| Concept | What the learner is given |
|---|---|
| Position | a length-64 vector of **integer symbols** (the occupant of each cell) |
| Symbols | an arbitrary, seeded permutation of `{empty, 12 piece kinds}` — no labels |
| Cells | shuffled by a **fixed random permutation**, so index order encodes *no* 8×8 geometry |
| Move | a pair `(from_index, to_index)` in `range(64)` |

No rules, legality, piece identities, board geometry, or domain knowledge are
provided. `python-chess` is used in exactly two places, both walled off from the
learner: (1) generating the data, and (2) *scoring* recovery against ground
truth. The recovery modules (`geometry`, `pieces`, `legality`, `strategy`)
consume only the opaque arrays. A test
(`test_learner_modules_do_not_import_ground_truth`) guards this firewall.

The bots (`bots.py`) are the data sources. Every bot is confined to the *same*
legal move set (the rules) but weights it differently (its strategy):
`random`, `capture`, `center`, `mobility`, and a depth-2 `minimax`.

## What gets recovered — results

Run with `python -m chess_rules.experiment` (seed 0, ~11k pooled observations).

### Experiment 1 — Board geometry from move adjacency
Build a cell-to-cell affinity from how often moves connect two cells, then embed
the 64 cells in 2-D with Laplacian eigenmaps. Short, frequent moves (king steps,
pawn pushes) dominate local connectivity, so the grid emerges.

| metric | value |
|---|---|
| Procrustes disparity to true 8×8 grid | **0.05** (0 = perfect) |
| local neighbor preservation (8-neighborhood) | **0.69** |

The global layout aligns to the true grid with ~5% residual; the file/rank axes
re-emerge from nothing but move counts. See `results/geometry.png`.

### Experiment 2 — Piece movement rules, without knowing the pieces
For each opaque moving-symbol, summarize its displacement vectors into a
movement fingerprint, then cluster symbols (unsupervised).

| metric | value |
|---|---|
| symbol→piece-kind clustering (Adjusted Rand Index) | **1.00** |
| observed displacements legal for the true piece (precision) | **1.00** |

The 12 moving symbols sort **perfectly** into the 6 piece kinds, both colors
together. The recovered fingerprints speak for themselves (symbol is opaque to
the learner; the "truth" column is the eval-only answer key):

```
sym  truth            knight diag orth long reach fwd
  5  w-knight          1.00  0.00 0.00 0.00   2  +0.57   <- only L-shapes
  7  w-bishop          0.00  1.00 0.00 0.47   6  +0.58   <- pure diagonals, long
  1  w-rook            0.00  0.00 1.00 0.50   7  +0.39   <- pure ranks/files, long
  8  w-queen           0.00  0.48 0.52 0.50   7  +0.46   <- diag + orth, long
  3  w-king            0.00  0.48 0.52 0.01   2  +0.29   <- diag + orth, never long
 12  w-pawn            0.00  0.23 0.77 0.26   2  +1.00   <- push + capture, one-way
  0  b-pawn            0.00  0.23 0.77 0.26   2  -1.00   <- mirror direction (color!)
```

`forward_bias` (±1.00 for pawns) even recovers **color** as the symmetry-breaking
direction of pawn movement. See `results/pieces.png`.

### Experiment 3 — Legal-move support (the rules)
A negative-sampling classifier (observed moves as positives, random `(from,to)`
corruptions as negatives — PU learning) recovers the legal support, scored
against python-chess ground truth it never saw:

| metric | value |
|---|---|
| ROC-AUC (legal vs illegal candidates) | **0.995** |
| average precision | **0.995** |

### Experiment 4 — Rules vs. strategy disentanglement
The headline. **Rules = what is invariant across data sources; strategy = what
varies within the legal support.** Train on bot *A*, evaluate on bot *B*:

```
LEGALITY recovery (ROC-AUC)          STRATEGY recovery (top-1 of legal moves)
train\eval  rnd  cap  cen  mob  mmx  train\eval  rnd  cap  cen  mob  mmx
   random  .99  .99  .99 1.00  .99     random  .03  .11  .06  .15  .07
  capture  .99  .99  .99  .99  .99    capture  .05  .30  .12  .24  .12
   center  .99  .99  .99 1.00  .99     center  .04  .20  .17  .20  .11
 mobility  .99  .99  .99 1.00 1.00    mobility  .04  .27  .14  .30  .09
  minimax  .99  .99  .99 1.00  .99     minimax  .04  .26  .14  .23  .12
```

- **Legality is flat at ~0.99 everywhere** — the rules learned from *any* bot
  transfer to *every* bot. They are the invariant.
- **Strategy is diagonal-dominant** — for every bot with a real preference
  (all but `random`), its own model predicts its own moves best: the diagonal
  entry leads its column. The `random` bot is unpredictable by anyone (its whole
  column sits at ~chance, where no training source helps). Preference does not
  transfer; it is bot-specific.

That contrast — one matrix flat, the other diagonal — *is* the disentanglement.

**Bit decomposition** of move predictability:

| quantity | bits |
|---|---|
| naming a move blind (`log2(64×64)`) | 12.00 |
| residual once the legal set is known (`log2 #legal`) | 4.58 |
| **explained by rules alone** | **7.42 (62%)** |

Knowing the *rules* removes ~62% of the uncertainty about the next move; only
the remaining 38% is even available for strategy to explain.

## Answering the research question

**How much hidden structure is recoverable from observations alone?**
Most of it. With zero domain knowledge, the pipeline recovers the board's
geometry (Procrustes 0.05), every piece's movement rule and the 6 piece kinds
(ARI 1.0), color as pawn-direction symmetry, and the legal-move support
(AUC 0.995) — and it cleanly separates rules from strategy by their
**invariance across data sources**.

**Why does this work?** Chess rules are a *hard constraint* on the support of
`P(move | state)`: illegal moves never appear, in any position, under any bot.
That makes them the stable, low-entropy backbone of the data. Strategy only
*reweights* moves within the legal support, so it is softer, source-dependent,
and explains far less of the predictability. The learner recovers the constraint
sharply and the preferences only partially — exactly the asymmetry the framing
predicted.

## Limitations (honest notes)

- The 64-symbol occupancy encoding omits side-to-move, castling rights, and
  en-passant state, so a sliver of true legality is unidentifiable from a single
  frame by construction. Recovery is measured against this information ceiling.
- Geometry recovery is global-accurate but locally soft (~0.69 neighbor
  preservation) due to the cosine warp of spectral embedding and long-range
  sliding-piece edges.
- Experiments 2–4 use a clean integer grid as a scaffold. Experiment 1
  establishes that grid is recoverable; the later stages assume it rather than
  compounding its warp, keeping piece identities and legality strictly hidden.
- Strategy top-1 accuracy is modest in absolute terms because the bots are
  stochastic (softmax move choice); the *relative* transfer structure, not the
  absolute number, carries the claim.

## Files

```
chess_rules/
  encoding.py     opaque codec + ground-truth answer keys (eval only)
  bots.py         five strategies over the shared legal set (data sources)
  datagen.py      self-play rollouts -> opaque (state, move) dataset
  geometry.py     E1: recover the 8x8 grid from move adjacency
  pieces.py       E2: recover per-symbol movement rules / piece kinds
  legality.py     E3: recover the legal-move support (rules)
  strategy.py     E4: recover preferences; rules-vs-strategy bit split
  experiment.py   orchestrates E1-E4, writes results/ (json + figures)
  results/        results.json, geometry.png, pieces.png, transfer.png
tests/test_chess_rules.py
```

## Run it

```bash
pip install -r chess_rules/requirements.txt
python -m chess_rules.experiment          # full run (~2 min)
python -m chess_rules.experiment --quick   # smoke test
pytest tests/test_chess_rules.py
```
