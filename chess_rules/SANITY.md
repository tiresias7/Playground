# Sanity checks — is the recovery real, or memorized?

The headline numbers (legality AUC 0.995, piece-clustering ARI 1.0) are
surprising enough to deserve scrutiny. This document records the checks in
`sanity.py` (`python -m chess_rules.sanity`, seed 0) and what they imply. Short
version: **position memorization is ruled out, but the optimistic 0.995 legality
number is inflated by easy negatives; the rule is recovered in layers, and one
layer (king-safety) is not recovered at all.**

## Setup recap (the skeptic's questions)

- **Where do positions come from?** Self-play. Each bot plays itself from the
  standard start, 4 random opening plies for diversity, then ≤50 plies. So
  positions are *on-policy trajectories*, not uniform samples of legal chess.
- **What bots?** Five shallow heuristics: `random`, `capture` (greedy MVV),
  `center`, `mobility` (1-ply), `minimax` (depth-2 material). All weak. Weakness
  *helps*: `random` explores the legal support broadly.
- **Target complexity?** Branching factor mean **27** (5th–95th pct: 6–42);
  11–32 pieces on board (mean 26 — opening/midgame heavy, endgames thin).

## [A] Position memorization — ruled out

Measured on the **opaque state the model actually sees** (64-cell occupancy),
not the FEN (whose move counters make everything look unique):

| overlap between train and eval | value |
|---|---|
| eval states also seen in train | **0.0%** |
| eval (state, move) also seen in train | **0.0%** |
| distinct train states / rows | 10,933 / 10,955 |

The position space is astronomically large; self-play essentially never repeats.
The model never looks up a seen position — it must generalize.

## [D] Legality AUC by negative type — the 0.995 was the easy case

The eval pits legal moves against *negatives*. The kind of negative is
everything:

| legal vs. … | AUC | what it tests |
|---|---|---|
| random `(from,to)` | **0.995** | trivial: most randoms are geometric nonsense |
| **self-capture** (valid shape, lands on own piece) | **0.960** | occupancy/capture rule |
| **pin/check** (illegal only via king safety) | **0.473** | king-safety rule |

So the rules are recovered **in layers**: movement geometry ✅, capture/occupancy
✅, king-safety dynamics ❌ (chance, even slightly anti-correlated — a pinned
piece's move *looks* perfectly legal, so the model rates it above average). The
64-occupancy encoding carries no check information and we supplied no notion of
"check", so this layer is unidentifiable by construction.

## [C] Memorization baselines — model beats lookup only where it matters

Two pure-lookup baselines: exact `(from,to)` index-pair frequency, and
`(mover_symbol, displacement)` shape frequency. If a lookup matches the model,
the "rule" is just memorized shapes.

| legal vs. | exact-pair | shape lookup | **feature model** |
|---|---|---|---|
| random | 0.917 | 0.984 | 0.995 |
| self-capture | 0.551 | 0.617 | **0.959** |

Against random negatives, memorizing common move-shapes already gets 0.92–0.98 —
so the 0.995 is *not* strong evidence of rule discovery. Against self-capture
negatives the lookups **collapse to ~chance** (the shape is legal; only the
target is wrong), yet the model holds **0.96**. That gap is the genuine,
non-memorized rule-learning: the model uses the *target square's* occupant
symbol, which a shape lookup cannot.

## [B] Generalization to unseen legal moves

Only ~2% of eval legal candidates were never played in training (the bots
collectively cover most common moves), but AUC on that subset is 0.996 — it does
extend to the few legal moves it never saw played.

## [E] Rule extrapolation — fails (anchored to observed magnitudes)

Train with **no move longer than 4 squares**, then test on long slides:

- AUC on held-out long moves: **1.000** (ranking survives — geometry/path
  features still order legal long-slides above random garbage)
- legal long slides the model calls legal (p > 0.5): **0%**

The model did **not** induce "a rook slides arbitrarily far." It learned the
*observed* move-length distribution and calls every unseen-length slide illegal.
The high AUC is misleading here — it reflects relative ranking, not calibrated
legality. This is the clearest memorization-flavored limitation.

## [F] Coverage of the legal repertoire

Distinct `|offset|` classes observed per piece (color-folded):

| piece | observed | true |
|---|---|---|
| pawn | 3 | 3 |
| knight | 2 | 2 |
| bishop | 6 | 7 |
| rook | 14 | 14 |
| queen | 21 | 21 |
| king | 4 | 4 |

Near-complete — the varied/weak bots explore the movement repertoire well; only
the single longest bishop diagonal (rare) is missed.

## [G] Prediction accuracy (not just AUC)

`python -m chess_rules.proximity`, threshold 0.5.

**Legality** (note pin/check: the 0.867 raw accuracy is just the legal base rate
— balanced accuracy 0.50 confirms chance):

| legal vs. | accuracy | balanced acc | AUC |
|---|---|---|---|
| random | 0.995 | 0.994 | 0.995 |
| self-capture | 0.955 | 0.925 | 0.964 |
| pin/check | 0.867 | **0.499** | 0.459 |

**Move / strategy** top-1 (predict the bot's move among the legal set):

| bot | top-1 | random-legal baseline |
|---|---|---|
| pooled | 0.168 | 0.059 (≈2.9×) |
| capture | 0.295 | |
| mobility | 0.210 | |
| minimax | 0.165 | |
| center | 0.149 | |
| random | 0.029 | (≈chance — unpredictable, as it should be) |

## [H] Proximity — exact match is meaningless, so measure distance

Skeptic's point: positions need not be *identical* to a training position to be
effectively memorized — being *near* one is enough. So measure, for each eval
position, the Hamming distance (number of differing cells, 0–64) to its
**nearest** training position:

| min | p5 | median | p95 | max |
|---|---|---|---|---|
| 2 | 6 | **16** | 24 | 27 |

Only **0.3%** of eval positions are within 2 cells of any training position. On
average **16 of 64 squares differ** from the closest training position — these
are genuinely novel boards, not near-duplicates.

**Does accuracy fall off with distance from training data?** (flat ⇒
generalization; falling ⇒ interpolation/memorization)

| NN-distance | n | legality acc | move top-1 | (move chance) |
|---|---|---|---|---|
| 5–9 | 80 | 0.980 | 0.075 | 0.056 |
| 10–14 | 196 | 0.963 | 0.168 | 0.053 |
| 15–19 | 246 | 0.957 | 0.215 | 0.062 |
| 20–64 | 168 | 0.952 | 0.131 | 0.058 |

- **Legality accuracy is essentially flat** (0.98 → 0.95) across the whole
  distance range — it does not depend on being near a training position. Strong
  evidence the recoverable rule layers *generalize* rather than interpolate.
- **Move accuracy is, if anything, lowest for positions closest to training** —
  the opposite of the memorization signature. Move predictability tracks
  position type (branching, bot determinism), not proximity to training data.

Caveat: Hamming distance addresses the *near-duplicate / memorization* concern.
It does not capture *strategic* similarity ("obvious what to do anyway") — that
is a difficulty axis, not a memorization axis, and is listed as open work in
`RESEARCH_NOTES.md`.

## Verdict

| claim | holds up? |
|---|---|
| not memorizing positions | ✅ 0% overlap |
| piece kinds / movement geometry recovered | ✅ ARI 1.0, beats lookup on hard negs |
| capture/occupancy rule recovered | ✅ 0.96 vs lookup 0.55 |
| king-safety rule recovered | ❌ 0.47 (chance) — encoding can't carry it |
| unbounded sliding rule induced | ❌ anchors to observed magnitudes |
| "legality AUC 0.995" as a rule-discovery claim | ⚠️ inflated by easy negatives; 0.96 on hard negatives is the honest number |
| rules invariant across bots / strategy bot-specific | ✅ the strongest finding, unaffected |

The recovered "rules" are best described as the **static, geometric, occupancy
layer** of chess — which the model genuinely generalizes — not the full dynamic
rulebook. The disentanglement of rules (invariant across bots) from strategy
(bot-specific) stands regardless.
