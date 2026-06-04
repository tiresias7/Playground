# Open research directions

Captured from review discussion. These are the next problems worth chasing;
each is about *whether the data and model responses make sense*, not about
inflating a metric.

## 1. Similarity beyond exact match (partly started)

Exact-match overlap (0%) is meaningless — positions can be *near* a training
example, or *strategically obvious*, and look "learned" without generalization.

- **Done:** Hamming distance to nearest training position (median 16/64) and
  accuracy-vs-distance bins (legality flat ⇒ generalizes). See `SANITY.md` [H].
  Move-prediction *ceiling* per bot (`SANITY.md` [I]) and the rule-vs-subset
  generalization test (`SANITY.md` [J]) are also done — they bound the target
  scope and confirm legality learned the rule, not the bot's subset.
- **Open:** Hamming captures *near-duplicate* similarity, not *strategic*
  similarity. Two boards can be far in Hamming yet tactically identical (same
  pawn skeleton, same forced recapture). Better notions to try:
  - difficulty proxies that flag "obvious" positions: branching factor, move
    entropy of the bot, presence of a forced/only-legal move, material swing;
  - learned position embeddings; distance in embedding space;
  - condition every accuracy number on difficulty, not just on NN-distance.
- **Question:** how much of move-prediction accuracy is just "obvious" positions
  (only-move, forced recapture) vs. genuine preference modeling?

## 2. Representation & target choice as inductive bias (the deep one)

> "By encoding the board in such a method, are we doing a favor for which model,
> in which prediction / which rule?"

The encoding is itself a hypothesis. Our choice — a 64-cell **occupancy** grid +
an **absolute `(from, to)`** target — makes some rules trivially readable and
others invisible:

- *favored:* movement geometry and capture/occupancy legality (a displacement +
  a target-cell symbol is all you need — hence the easy 0.96).
- *hidden:* king-safety (no check signal), side-to-move (absent), move history.

Research plan: hold the bots/data fixed and **sweep the representation**, then
measure per-rule recoverability for each:
- board encodings: occupancy grid vs. piece-centric lists vs. **relative**
  coordinates vs. attack/threat maps vs. raw move-sequence (history) vs. image.
- target formulations: legality classification vs. next-move vs. masked-cell
  reconstruction vs. predicting the full legal *set*.
- a matrix of (representation × rule) recoverability would show explicitly which
  representation "does a favor" for which rule. This reframes "can chess be
  learned?" as "which rules are identifiable under which representation?"

## 3. Data-generation-induced difficulty (enumerate the hard/impossible rules)

Some rules are hard or **impossible** to recover from our data *by construction*,
and it is worth stating exactly why and what minimal change fixes each:

| rule / fact | why hard with current data | minimal fix |
|---|---|---|
| king safety (pins, moving into check) | needs global reasoning; no check signal in occupancy; only legal moves shown (no "illegal-because-check" examples) | add attack/king-danger features or move-history |
| side-to-move | not encoded → same occupancy admits either side → legality genuinely ambiguous | add a turn channel |
| en passant | depends on the *previous* move, not the frame | add last-move / history |
| castling rights | depends on whether king/rook moved before | add right flags / history |
| promotion choice | our `(from, to)` target **drops the promotion piece** → promotions conflated | richer target `(from, to, promo)` |
| 50-move / threefold repetition / draws | depend on history & counters | add counters / history |
| checkmate / stalemate / game-over | terminal, rarely sampled; needs check | sample terminal states; add check signal |

Also: the data is **opening/midgame-heavy** (mean 26 pieces, endgames thin), so
endgame-specific patterns are undersampled regardless of representation.

**Through-line:** "how much of chess can be recovered" is not one number — it is
*per-rule*, and each rule's recoverability is jointly set by (representation,
target, data-generating distribution). The current setup recovers the static,
single-frame, geometric/occupancy layer well and the dynamic/history-dependent
layer not at all — and that split is predictable from the three factors above.
