"""PPO (gradient RL) for Figgie, the comparison arm to ES.

Actor-critic MLP over the same 34 features / 33 masked actions. Trains on the
dense-reward FiggieEnv against a league of opponents (hand-coded bots + frozen
PPO snapshots, for RPS robustness). Single process; CPU torch.
"""

from __future__ import annotations

import argparse
import copy
import random

import numpy as np
import torch
import torch.nn as nn

from ..bots import (AdaptiveCollectorBot, CollectorBot, HeuristicBot, MMBot,
                    RandomBot, ValueBot)
from ..bots.base import Bot
from .agent_state import AgentState
from .env import FiggieEnv
from .features import N_ACTIONS, N_FEATURES

torch.set_num_threads(2)


class ActorCritic(nn.Module):
    def __init__(self, n_in=N_FEATURES, n_out=N_ACTIONS, hidden=64):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(),
                                   nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.Linear(hidden, n_out)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.pi(h), self.v(h).squeeze(-1)


def _masked(logits, mask):
    return logits.masked_fill(~mask, -1e9)


class PPOBot(Bot):
    def __init__(self, net, name=None, greedy=True, seed=0):
        super().__init__(name)
        self.net = net
        self.greedy = greedy
        self.st = AgentState()

    def reset(self, obs):
        self.st.reset()

    @torch.no_grad()
    def act(self, obs):
        f, mask, actions = self.st.observe(obs)
        logits, _ = self.net(torch.from_numpy(f))
        ml = _masked(logits, torch.from_numpy(mask))
        if self.greedy:
            a = int(torch.argmax(ml))
        else:
            a = int(torch.distributions.Categorical(logits=ml).sample())
        return actions[a]


def base_league():
    def w(cls, **k):
        return lambda r: cls(rng=random.Random(r.random()), **k)
    return [w(RandomBot), w(HeuristicBot), w(ValueBot), w(MMBot),
            w(CollectorBot), w(AdaptiveCollectorBot)]


@torch.no_grad()
def run_episode(net, env, seed):
    st = AgentState()
    obs = env.reset(seed)
    st.reset()
    F, M, A, LP, V, R = [], [], [], [], [], []
    done = False
    while not done:
        f, mask, actions = st.observe(obs)
        ft, mt = torch.from_numpy(f), torch.from_numpy(mask)
        logits, v = net(ft)
        dist = torch.distributions.Categorical(logits=_masked(logits, mt))
        a = dist.sample()
        obs, r, done = env.step(actions[int(a)])
        F.append(f); M.append(mask); A.append(int(a))
        LP.append(float(dist.log_prob(a))); V.append(float(v)); R.append(r)
    return F, M, A, LP, V, R


def gae(rewards, values, gamma=0.997, lam=0.95):
    adv = np.zeros(len(rewards), dtype=np.float32)
    last = 0.0
    for t in reversed(range(len(rewards))):
        nextv = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * nextv - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    ret = adv + np.array(values, dtype=np.float32)
    return adv, ret


def train(updates=150, episodes_per=16, ticks=250, hidden=64, lr=3e-4,
          clip=0.2, epochs=4, ent=0.01, league_every=40, seed=0,
          init=None, out="figgie/rl/ppo_policy.pt"):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    net = ActorCritic(hidden=hidden)
    if init:
        net.load_state_dict(torch.load(init))
        print(f"warm-started from {init}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    league = base_league()

    for u in range(updates):
        F, M, A, LP, ADV, RET = [], [], [], [], [], []
        ep_returns = []
        for _ in range(episodes_per):
            opps = [random.choice(league) for _ in range(3)]  # factories
            env = FiggieEnv(opps, total_ticks=ticks)
            f, m, a, lp, v, r = run_episode(net, env, rng.randrange(1 << 30))
            adv, ret = gae(r, v)
            F += f; M += m; A += a; LP += lp; ADV += list(adv); RET += list(ret)
            ep_returns.append(sum(r))
        F = torch.tensor(np.array(F))
        M = torch.tensor(np.array(M))
        A = torch.tensor(A)
        LP = torch.tensor(LP, dtype=torch.float32)
        ADV = torch.tensor(np.array(ADV))
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-6)
        RET = torch.tensor(np.array(RET))

        n = len(A)
        idx = np.arange(n)
        for _ in range(epochs):
            np.random.shuffle(idx)
            for s in range(0, n, 512):
                b = idx[s:s + 512]
                logits, v = net(F[b])
                dist = torch.distributions.Categorical(logits=_masked(logits, M[b]))
                lp = dist.log_prob(A[b])
                ratio = torch.exp(lp - LP[b])
                a1 = ratio * ADV[b]
                a2 = torch.clamp(ratio, 1 - clip, 1 + clip) * ADV[b]
                pi_loss = -torch.min(a1, a2).mean()
                v_loss = ((v - RET[b]) ** 2).mean()
                loss = pi_loss + 0.5 * v_loss - ent * dist.entropy().mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()

        print(f"update {u:3d}  ep_return {np.mean(ep_returns):7.2f}  "
              f"league {len(league)}", flush=True)
        if (u + 1) % league_every == 0:
            snap = copy.deepcopy(net).eval()
            league.append(lambda r, s=snap: PPOBot(s, greedy=False, seed=r.randrange(1 << 30)))

    torch.save(net.state_dict(), out)
    print(f"saved PPO policy to {out}", flush=True)
    return net


def main():
    ap = argparse.ArgumentParser()
    for k, d in [("updates", 150), ("episodes_per", 16), ("ticks", 250),
                 ("hidden", 64), ("epochs", 4), ("league_every", 40), ("seed", 0)]:
        ap.add_argument(f"--{k}", type=int, default=d)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--init", type=str, default=None)
    ap.add_argument("--out", type=str, default="figgie/rl/ppo_policy.pt")
    a = ap.parse_args()
    train(a.updates, a.episodes_per, a.ticks, a.hidden, a.lr, epochs=a.epochs,
          league_every=a.league_every, seed=a.seed, init=a.init, out=a.out)


if __name__ == "__main__":
    main()
