"""Self-play parameter tuning for SharpBot via the cross-entropy method.

This is the pragmatic "RL" given no deep-learning stack: treat the bot's four
free parameters as a policy and optimize them by repeated self-play.

Fitness of a candidate = its net/round, averaged over two fields:
  * vs three copies of the current incumbent (best-response / don't overpay),
  * vs a weak field (Value, Heuristic, Random) (exploit weaker opponents),
weighted 0.6 / 0.4.

Common random numbers: every candidate in a generation plays the *same* deck
seeds, so comparisons are paired and low-variance. Candidates are evaluated in
parallel across CPU cores.
"""

from __future__ import annotations

import argparse
import math
import random
from multiprocessing import Pool

from .bots import HeuristicBot, RandomBot, SharpBot, ValueBot
from .game import FiggieGame

PARAMS = ("buy_frac", "sell_frac", "flow_lam", "flow_cap")
BOUNDS = {
    "buy_frac": (0.2, 1.0),
    "sell_frac": (1.0, 2.5),
    "flow_lam": (0.0, 1.5),
    "flow_cap": (1.0, 14.0),
}
INIT_MEAN = {"buy_frac": 0.65, "sell_frac": 1.35, "flow_lam": 0.45, "flow_cap": 6.0}
INIT_STD = {"buy_frac": 0.25, "sell_frac": 0.4, "flow_lam": 0.4, "flow_cap": 4.0}


def _clip(name, v):
    lo, hi = BOUNDS[name]
    return max(lo, min(hi, v))


def _game(factories, seed, total_ticks):
    grng = random.Random(seed)
    bots = [factories[s](random.Random(seed * 1009 + s)) for s in range(len(factories))]
    return FiggieGame(bots, total_ticks=total_ticks, rng=grng).play()


def eval_candidate(args):
    """Top-level so it is picklable for multiprocessing."""
    cand, incumbent, seeds, total_ticks = args
    cp = dict(zip(PARAMS, cand))
    ip = dict(zip(PARAMS, incumbent))

    def cf(r): return SharpBot(rng=r, **cp)
    def inf(r): return SharpBot(rng=r, **ip)
    def vf(r): return ValueBot(rng=r)
    def hf(r): return HeuristicBot(rng=r)
    def rf(r): return RandomBot(rng=r)

    A = B = 0.0
    for seed in seeds:
        A += _game([cf, inf, inf, inf], seed, total_ticks).net[0]
        B += _game([cf, vf, hf, rf], seed + 777, total_ticks).net[0]
    k = len(seeds)
    return 0.6 * (A / k) + 0.4 * (B / k)


def cem(generations=8, pop=24, elite_frac=0.25, rounds=140, total_ticks=800,
        seed=0, workers=4):
    rng = random.Random(seed)
    mean = dict(INIT_MEAN)
    std = dict(INIT_STD)
    incumbent = tuple(mean[p] for p in PARAMS)
    best_fit = -1e9
    best = incumbent

    with Pool(workers) as pool:
        for gen in range(generations):
            gseeds = [rng.randrange(1 << 30) for _ in range(rounds)]
            cands = [incumbent]  # always re-evaluate incumbent under new seeds
            for _ in range(pop - 1):
                cands.append(tuple(
                    _clip(p, rng.gauss(mean[p], std[p])) for p in PARAMS))
            args = [(c, incumbent, gseeds, total_ticks) for c in cands]
            fits = pool.map(eval_candidate, args)

            ranked = sorted(zip(fits, cands), key=lambda x: -x[0])
            n_elite = max(2, int(elite_frac * pop))
            elite = [c for _, c in ranked[:n_elite]]
            for i, p in enumerate(PARAMS):
                vals = [c[i] for c in elite]
                mean[p] = sum(vals) / len(vals)
                var = sum((v - mean[p]) ** 2 for v in vals) / len(vals)
                std[p] = max(0.05, math.sqrt(var))

            gen_best_fit, gen_best = ranked[0]
            incumbent = gen_best
            if gen_best_fit > best_fit:
                best_fit, best = gen_best_fit, gen_best
            print(f"gen {gen}: best_fit={gen_best_fit:7.2f}  "
                  f"mean={{{', '.join(f'{p}={mean[p]:.2f}' for p in PARAMS)}}}",
                  flush=True)

    print("\nBEST PARAMS:")
    bp = dict(zip(PARAMS, best))
    for p in PARAMS:
        print(f"  {p} = {bp[p]:.3f}")
    print(f"  fitness = {best_fit:.2f}")
    return bp, best_fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--rounds", type=int, default=140)
    ap.add_argument("--ticks", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cem(args.generations, args.pop, rounds=args.rounds, total_ticks=args.ticks,
        seed=args.seed, workers=args.workers)


if __name__ == "__main__":
    main()
