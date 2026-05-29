# Asymmetric-information trading game — research framework

A framework for studying a **three-player game with private information** over a
limit order book. The goal is **not** to hand-code strategies and confirm a
human guess. It is to give a solver enough machinery to *tell us* who provides
liquidity, whether informed players leak by posting, and when a silent market
forms — and to let us find counterexamples.

> **Design rule:** no behavioural prior is hard-coded. The solver follows
> estimated payoffs only. Whatever emerges (post early / wait / provide
> liquidity / snipe) is a property of the game, not of the code.

## The game

| Player | Private info | Payoff of one unit held to the end (`V`) |
|--------|--------------|------------------------------------------|
| **S** (State)  | `state ∈ {FIX, RANDOM}` | `30` if FIX, else `x` |
| **X** (Number) | `x ∈ {0..100}`          | `x` if RANDOM, else `30` |
| **B** (Buyer)  | `k ∈ {1,2,3}`           | must buy ≥ `k` units or pay a penalty |

Priors: `P(FIX)=P(RANDOM)=0.5`, `x ~ U{0..100}`, `k ~ U{1,2,3}`. Neither S nor
X alone knows `V`; only together do they pin it down. B knows neither.

**Market.** Discrete `T` rounds; each round all three act once in a fresh random
order. Actions: `WAIT, POST_BID(p), POST_ASK(p), TAKE_BID, TAKE_ASK,
CANCEL(id)`. Size 1, limit orders persist, configurable tick. Final PnL =
`position·V + cash − penalty`.

## Layout

```
asym_market/
  types.py          enums + Order/Trade/Action dataclasses
  config.py         GameConfig — every ablation is just a different config
  order_book.py     single-unit price-time-priority book (marketable limits)
  env.py            GameEnv: episode driver, Observation, legal_actions, logging
  policies.py       Policy ABC, Random/Wait baselines, TabularPolicy, info_key
  evaluate.py       evaluate_policy(profile, ...) -> mean PnL + full logs
  best_response.py  best_response(...) and iterative_best_response(...)
  diagnostics.py    behavioural metrics (first poster, post-prob | private, ...)
  logging_utils.py  CSV export
  tests/            pytest: order matching + PnL settlement
  scripts/          run_phase12.py driver
```

## Quick start

```bash
python -m pytest asym_market/tests -q            # 16 tests: matching + PnL
python -m asym_market.scripts.run_phase12 --quick   # fast end-to-end demo
python -m asym_market.scripts.run_phase12           # full run -> CSVs
```

### Programmatic use

```python
from asym_market import GameConfig, iterative_best_response, evaluate_policy, behavioral_report

cfg = GameConfig(horizon=3, post_price_grid=(0,20,40,60,80,100))
ibr = iterative_best_response(cfg, n_iters=6, num_episodes=30000,
                              x_bucket=20, pos_clip=2)
res = evaluate_policy(ibr.profile, cfg, num_simulations=20000, keep_logs=True)
report = behavioral_report(res.logs, cfg)   # who posts first, post-prob | x, ...
```

## Solver notes (honest scope)

* `best_response` solves, for one player against **fixed** opponents, the
  induced single-agent POMDP via **sampling-based Monte-Carlo control**
  (every-visit, terminal reward, ε-greedy). The information-set key
  (`policies.info_key`) is the abstraction knob: `x_bucket` groups `x`,
  `pos_clip` clips inventory.
* `iterative_best_response` is cyclic (Gauss–Seidel) IBR from an **all-random**
  (unbiased) start, tracking an exploitability proxy = max single-player BR gain.
* This is **approximate**. IBR can cycle without converging — that is reported,
  not hidden. Phase 3 adds fictitious play / regret-matching for better
  convergence to approximate equilibria.

See `REPORT_phase12.md` for the formalization, method, and preliminary
(emergent, caveated) observations from the checkpoint run.
