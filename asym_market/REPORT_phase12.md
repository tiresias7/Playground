# Phase 1–2 checkpoint report

**Status:** environment + order book + full logging + finite-action solver
(best response / iterative best response) + behavioural diagnostics + tests.
Phases 3–5 (fictitious play / regret matching, ablation suite, final report)
are intentionally **not** in this checkpoint — see *Next steps*.

> This report describes the **framework** and reports **emergent, caveated
> observations** from one run. It deliberately does **not** declare answers to
> the research questions. The point of the checkpoint is to confirm the
> machinery measures the right things before we trust any conclusion.

---

## 1. Game formalization

Three players, one risky asset, discrete horizon `T`.

* Hidden draws: `state ∈ {FIX, RANDOM}` (`P(FIX)=0.5`), `x ∈ {0,5,…,100}`
  (uniform), `k ∈ {1,2,3}` (uniform). Terminal value `V = 30` if FIX else `x`.
* Private signals: **S** sees `state`, **X** sees `x`, **B** sees `k`.
  - S knows `V` only in the FIX branch (`V=30`); under RANDOM, S knows the
    *regime* but not the number.
  - X knows the number that *would* apply, but not whether it applies.
  - So full knowledge of `V` is **split across S and X**; neither is fully
    informed. B is uninformed but has a forced-demand obligation.
* Market microstructure: each round, all three act once in a freshly drawn
  random order. Actions `{WAIT, POST_BID(p), POST_ASK(p), TAKE_BID, TAKE_ASK,
  CANCEL(id)}`. Orders are size 1, persist until filled/cancelled/end, snap to a
  tick grid. Marketable limit orders execute at the resting (maker) price.
* Settlement: `PnL_i = position_i · V + cash_i`; for B,
  `− penalty_per_unit · max(0, k − gross_buys_B)`.

The obligation is on **gross buys**, not net position — buying `k` then selling
them back satisfies it. (Whether to instead require *holding* `k` is a config
question flagged for Phase 4.)

## 2. Method

* **Engine** (`env.py`). One strategy-agnostic episode driver takes an
  `act_fn(player, Observation, legal_actions) → Action`. The same engine serves
  baselines, tabular policies and the best-response explorer, so behaviour and
  payoff always refer to identical mechanics. Every decision, book snapshot,
  trade, first-poster and first-taker is logged into an `EpisodeLog`.
* **Information sets** (`policies.info_key`). The abstraction over which the
  solver reasons: private info + remaining rounds + best bid/ask + own resting
  order + clipped position + (for B) remaining obligation. Two knobs:
  `x_bucket` (group `x`) and `pos_clip` (clip inventory). This is the
  "narrowing" lever — exact on a tiny config, abstracted on larger ones.
* **Legality** (`env.legal_actions`) is the single source of truth and encodes
  *rules only* (no self-trade, one resting order, short-selling gate).

## 3. Solver used

* `best_response(player, fixed_strategies, config, …)`: against fixed opponents,
  the target faces a single-agent POMDP whose sufficient statistic is the
  info-set key. Solved by **every-visit Monte-Carlo control** with terminal
  reward and decaying ε-greedy exploration; greedy improvement gives a
  `TabularPolicy`. Returns the policy, the Q-table, visit counts, and a clean
  greedy value re-estimated on a fresh seed stream.
* `iterative_best_response(config, …)`: cyclic (Gauss–Seidel) IBR from an
  **all-random** start (unbiased — not silence, not aggression). Per iteration
  it reports an **exploitability proxy** = the maximum single-player gain from
  best-responding to the current profile. ≈0 would indicate an approximate
  equilibrium.

Honesty about scope: this is **sampling-based and approximate**. Estimates carry
Monte-Carlo noise; the greedy policy falls back to WAIT on info sets it never
visited; and IBR is not guaranteed to converge (see §6).

## 4. Key tables (this run)

Game: `T=3`, posting grid `{0,20,40,60,80,100}`, `penalty/unit=50`, `E[V]=40`.
Solve: `x_bucket=20`, `pos_clip=2`, `30000` episodes/BR, `6` IBR iterations.

**Convergence (exploitability proxy = max single-player BR gain):**

| iter | S | X | B | max BR gain |
|------|------|------|------|------|
| 0 | +4.45 | +16.17 | −39.28 | +67.32 |
| 1 | +4.82 | +12.55 | −34.17 | +24.65 |
| 2 | +6.79 | +19.26 | −47.53 | +25.28 |
| 3 | +13.40 | +13.37 | −61.49 | +25.37 |
| 4 | +9.23 | +10.61 | −53.81 | +26.52 |
| 5 | +9.38 | +19.69 | −57.92 | +22.55 |

**Post-IBR profile, evaluated on 20 000 fresh episodes:**

* Mean PnL: **S = +9.15**, **X = +20.26**, **B = −59.13** (±~0.3).
* `no_trade = 5.9%`, fully `silent = 0%`.
* First **poster**: S 33.7% / X 32.9% / B 33.4% / none 0% — essentially uniform.
* First **taker**: **B 63.2%** / S 17.6% / X 13.4% / none 5.9%.
* First-poster premium: **+1.59** (poster −8.85 vs others −10.44).
* B posts at `t=0` ~100% (k=1,2) and 97% (k=3); S posts-ever ~98–99%;
  X posts-ever ~82–97%, lowest at `x=100`.
* First-action mode: S `POST_ASK` 67%, X `POST_ASK` 71%, B `POST_BID` 78%.

## 5. Main observations (emergent, NOT conclusions)

Read directly off the diagnostics of the post-IBR iterate. **Descriptive only**
— this iterate is not a solved equilibrium (§6).

1. **No silent market in this regime.** With the obligation + penalty=50, the
   market never goes fully silent (`silent=0%`); someone always posts. This is a
   data point for research question 7, *not* a general claim.
2. **B drives execution, not posting.** First-*posting* is ~uniform across the
   three, but B is the first *taker* 63% of the time and posts a bid at `t=0`
   almost always. The forced demand shows up as **aggression/lifting**, not as a
   monopoly on providing the first quote (question 1/8).
3. **First-posting is not clearly punished here.** The first-poster premium is
   ≈ 0 (+1.6) in this iterate — and it was **−6.3** in the shorter `--quick`
   run. The sign is unstable across runs, so "informed-poster-leaks" (questions
   2/3) is **not** supported as a robust effect yet; it needs the Phase-3
   averaged equilibrium before any sign can be trusted.
4. **X (number player) earns the most, B pays for its obligation.** X +20 vs
   B −59. B's loss is essentially the cost of guaranteed demand against
   better-informed counterparties — quantifying that cost vs penalty is the
   Phase-4(B) sweep.
5. **X quotes least when `x=100`** (post-ever 82% vs ~95% elsewhere): a hint
   that an extreme number changes X's willingness to show — exactly the kind of
   `x`-conditioned effect Phase-4(F) will map. Treat as a lead, not a result.

## 6. Counterexamples / caveats found so far

* **IBR does not converge on this config.** The exploitability proxy stays
  large and oscillates across iterations — direct evidence that *naïve cyclic
  best response cycles* here, which is itself a finding: the game likely has no
  pure-strategy fixed point reachable this way, motivating the Phase-3
  fictitious-play / regret-matching averaging.
* Therefore **no claim about equilibrium behaviour should be drawn yet.** The
  "first-poster premium" and "post-probability | private info" numbers describe
  the *current iterate*, not a solved equilibrium.

## 7. Limitations

* Approximate solver; results carry sampling noise and depend on the info-set
  abstraction (`x_bucket`, `pos_clip`) and posting-grid coarseness.
* `max_outstanding_per_player = 1` and tiny `T` for tractability.
* Greedy tabular policies under-explore rare info sets (WAIT fallback).
* No equilibrium-averaging yet, so non-convergence ≠ no equilibrium.

## 8. Next steps (Phases 3–5)

1. **Phase 3** — fictitious play + regret-matching / simplified CFR for proper
   approximate-equilibrium averaging; compare convergence to IBR.
2. **Phase 4** — ablation suite as configs: (A) no buyer obligation, (B)
   penalty sweep, (C) horizon sweep, (D) `k=1` vs `k=3`, (E) tick fine/coarse,
   (F) `x` buckets near 0/30/50/100.
3. **Phase 5** — full markdown report with plots/tables and the parameter
   regions where each behaviour (post early / wait / silent) becomes
   approximately optimal.
