"""Evolution-strategies training of the RL policy, with league self-play.

OpenAI-style ES: perturb the mean weights with antithetic Gaussian noise, score
each perturbation by self-play return, rank-normalize, and step the mean along
the estimated gradient (Adam). Exploration is in *parameter* space, so the
policy is evaluated greedily and the sparse end-of-round reward is handled
natively (no per-step credit assignment needed).

To avoid the Rock-Paper-Scissors trap of naive self-play, fitness is the average
net across fields whose 3 opponents are sampled from a *league*: the hand-coded
bots plus periodically-frozen checkpoints of the policy itself. The agent must
beat a diverse population, not just its current self.
"""

from __future__ import annotations

import argparse
import random
from multiprocessing import Pool

import numpy as np

from ..bots import (AdaptiveCollectorBot, CollectorBot, HeuristicBot, MMBot,
                    RandomBot, ValueBot)
from ..game import FiggieGame
from .features import N_ACTIONS, N_FEATURES
from .policy import Policy
from .rl_bot import RLBot

BASE_LEAGUE = [("Random",), ("Heur",), ("Value",), ("MM",), ("Collector",), ("AC",)]


def build_opponent(tag, rng):
    r = random.Random(rng.random())
    kind = tag[0]
    if kind == "MM":
        return MMBot(rng=r)
    if kind == "Heur":
        return HeuristicBot(rng=r)
    if kind == "Random":
        return RandomBot(rng=r)
    if kind == "Value":
        return ValueBot(rng=r)
    if kind == "Collector":
        return CollectorBot(rng=r)
    if kind == "AC":
        return AdaptiveCollectorBot(rng=r)
    if kind == "RL":
        _, theta, spec = tag
        return RLBot(Policy(*spec), theta, greedy=True, seed=r.randrange(1 << 30))
    raise ValueError(tag)


def eval_theta(args):
    """Top-level (picklable). Mean net of the candidate policy across CRN fields."""
    theta, spec, league, seeds, ticks = args
    pol = Policy(*spec)
    total = 0.0
    for sd in seeds:
        rng = random.Random(sd)
        opps = [build_opponent(league[rng.randrange(len(league))], rng) for _ in range(3)]
        cand = RLBot(pol, theta, greedy=True, seed=sd)
        res = FiggieGame([cand, *opps], total_ticks=ticks,
                         rng=random.Random(rng.random())).play()
        total += res.net[0]
    return total / len(seeds)


def _ranks(fits):
    order = np.argsort(np.argsort(fits))  # rank of each element
    return order / (len(fits) - 1) - 0.5  # centered in [-0.5, 0.5]


def train(gens=60, npert=100, sigma=0.1, lr=0.05, rounds=18, ticks=300,
          hidden=16, workers=4, seed=0, league_every=12, out="figgie/rl/trained_theta.npy"):
    rng = np.random.default_rng(seed)
    pyrng = random.Random(seed)
    pol = Policy(N_FEATURES, N_ACTIONS, hidden)
    spec = (N_FEATURES, N_ACTIONS, hidden)
    mu = pol.init_theta(rng, 0.1)
    league = list(BASE_LEAGUE)
    half = npert // 2
    m = np.zeros_like(mu); v = np.zeros_like(mu); step = 0

    with Pool(workers) as pool:
        for gen in range(gens):
            seeds = [pyrng.randrange(1 << 30) for _ in range(rounds)]
            eps = rng.standard_normal((half, pol.dim)).astype(np.float32)
            thetas = []
            for i in range(half):
                thetas.append(mu + sigma * eps[i])
                thetas.append(mu - sigma * eps[i])
            fits = np.array(pool.map(eval_theta, [(t, spec, league, seeds, ticks) for t in thetas]))
            r = _ranks(fits)
            g = np.zeros(pol.dim, dtype=np.float32)
            for i in range(half):
                g += (r[2 * i] - r[2 * i + 1]) * eps[i]
            g /= (npert * sigma)
            step += 1
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * (g * g)
            mu += lr * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)

            print(f"gen {gen:3d}  mean_fit {fits.mean():7.2f}  max {fits.max():7.2f}  "
                  f"league {len(league)}", flush=True)
            if (gen + 1) % league_every == 0:
                league.append(("RL", mu.copy(), spec))

    np.save(out, mu)
    print(f"saved policy to {out}  (dim {pol.dim}, hidden {hidden})")
    return mu, spec


def main():
    ap = argparse.ArgumentParser()
    for k, d, t in [("gens", 60, int), ("npert", 100, int), ("rounds", 18, int),
                    ("ticks", 300, int), ("hidden", 16, int), ("workers", 4, int),
                    ("seed", 0, int), ("league_every", 12, int)]:
        ap.add_argument(f"--{k}", type=t, default=d)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--out", type=str, default="figgie/rl/trained_theta.npy")
    a = ap.parse_args()
    train(a.gens, a.npert, a.sigma, a.lr, a.rounds, a.ticks, a.hidden,
          a.workers, a.seed, a.league_every, a.out)


if __name__ == "__main__":
    main()
