"""Evaluate trained RL agents against the bot pool in the ecology tournament.

    python -m figgie.rl.evaluate --es figgie/rl/trained_theta.npy \
                                 --ppo figgie/rl/ppo_policy.pt --rounds 1500
"""

from __future__ import annotations

import argparse
import random

from ..bots import (AdaptiveCollectorBot, CollectorBot, HeuristicBot, MMBot,
                    RandomBot, ValueBot)
from ..tournament import ecology
from .load import rl_factory


def ppo_factory(path, hidden=64):
    import torch  # lazy: PPO eval needs torch, ES does not
    from .ppo import ActorCritic, PPOBot
    net = ActorCritic(hidden=hidden)
    net.load_state_dict(torch.load(path))
    net.eval()
    return lambda r: PPOBot(net, greedy=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--es", type=str, default=None, help="ES policy .npy")
    ap.add_argument("--ppo", type=str, default=None, help="PPO policy .pt")
    ap.add_argument("--ppo_hidden", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--ticks", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=2024)
    a = ap.parse_args()

    pool = {
        "MM": lambda r: MMBot(rng=random.Random(r.random())),
        "AdaptiveColl": lambda r: AdaptiveCollectorBot(rng=random.Random(r.random())),
        "Heuristic": lambda r: HeuristicBot(rng=random.Random(r.random())),
        "Collector": lambda r: CollectorBot(rng=random.Random(r.random())),
        "Value": lambda r: ValueBot(rng=random.Random(r.random())),
        "Random": lambda r: RandomBot(rng=random.Random(r.random())),
    }
    if a.es:
        pool["RL-ES"] = rl_factory(a.es, greedy=True)
    if a.ppo:
        pool["RL-PPO"] = ppo_factory(a.ppo, a.ppo_hidden)
    ecology(pool, a.rounds, seed=a.seed, total_ticks=a.ticks)


if __name__ == "__main__":
    main()
