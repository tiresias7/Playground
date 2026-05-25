"""Behavior cloning: warm-start the RL policy by imitating MMBot, then let PPO
finetune from there. Cold-start RL plateaued below the engineered bots; cloning
the strong hand-coded policy first gives PPO a good basin to improve from.

We run MMBot in seat 0 against a league, and at each of its decisions record the
RL feature vector, the action mask, and the *discrete action id* nearest to the
order MMBot actually played (matched by side+suit, then closest price). Then we
train the actor-critic's policy head by masked cross-entropy to reproduce it.
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
import torch.nn.functional as Fn

from ..bots import MMBot
from ..bots.base import Ask, Bid, Bot
from ..game import FiggieGame
from .agent_state import AgentState
from .ppo import ActorCritic, base_league


def map_action(mm_action, actions):
    """Nearest discrete action id to MMBot's order (None -> pass=0)."""
    if mm_action is None:
        return 0
    best, bid = None, None
    for i, a in enumerate(actions):
        if a is None or type(a) is not type(mm_action) or a.suit is not mm_action.suit:
            continue
        d = abs(a.price - mm_action.price)
        if best is None or d < best:
            best, bid = d, i
    return bid if bid is not None else 0


class RecorderBot(Bot):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.st = AgentState()
        self.data = []

    def reset(self, obs):
        self.st.reset()
        self.inner.reset(obs)

    def act(self, obs):
        feat, mask, actions = self.st.observe(obs)
        a = self.inner.act(obs)
        self.data.append((feat, mask, map_action(a, actions)))
        return a


def collect(n_games, ticks, seed):
    rng = random.Random(seed)
    league = base_league()
    F, M, Y = [], [], []
    for _ in range(n_games):
        rec = RecorderBot(MMBot(rng=random.Random(rng.random())))
        opps = [random.choice(league)(rng) for _ in range(3)]
        FiggieGame([rec, *opps], total_ticks=ticks,
                   rng=random.Random(rng.random())).play()
        for feat, mask, tid in rec.data:
            F.append(feat); M.append(mask); Y.append(tid)
    return np.array(F, dtype=np.float32), np.array(M), np.array(Y)


def train_bc(F, M, Y, hidden=64, epochs=10, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    net = ActorCritic(hidden=hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Ft = torch.from_numpy(F)
    Mt = torch.from_numpy(M)
    Yt = torch.from_numpy(Y).long()
    n = len(Y)
    idx = np.arange(n)
    for ep in range(epochs):
        np.random.shuffle(idx)
        tot, correct, nb = 0.0, 0, 0
        for s in range(0, n, 1024):
            b = idx[s:s + 1024]
            logits, _ = net(Ft[b])
            ml = logits.masked_fill(~Mt[b], -1e9)
            loss = Fn.cross_entropy(ml, Yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
            correct += int((ml.argmax(1) == Yt[b]).sum())
        print(f"bc epoch {ep:2d}  loss {tot/nb:.3f}  acc {correct/n:.3f}", flush=True)
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--ticks", type=int, default=250)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="figgie/rl/bc_policy.pt")
    a = ap.parse_args()
    print(f"collecting from {a.games} MMBot games...", flush=True)
    F, M, Y = collect(a.games, a.ticks, a.seed)
    print(f"dataset: {len(Y)} samples", flush=True)
    net = train_bc(F, M, Y, a.hidden, a.epochs, seed=a.seed)
    torch.save(net.state_dict(), a.out)
    print(f"saved BC policy to {a.out}", flush=True)


if __name__ == "__main__":
    main()
