"""Best-response and iterative best-response (approximate equilibrium) search.

Against *fixed* opponent policies, one player faces a single-agent POMDP whose
sufficient statistic is the information-set key (private info + public history,
since opponents are fixed). We solve it with sampling-based Monte-Carlo control
(every-visit, terminal reward, epsilon-greedy exploration). This is
approximate: quality depends on episode count and the info-set abstraction.

Nothing here biases *which* action is good. The greedy improvement simply
follows the estimated Q-values, so whatever behaviour emerges (post early, stay
silent, provide liquidity, snipe) is a property of the game, not the code.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .config import GameConfig
from .env import GameEnv, Observation
from .evaluate import EvalResult, evaluate_policy
from .policies import (
    AlwaysWaitPolicy, ActionTemplate, InfoKey, Policy, RandomPolicy,
    TabularPolicy, info_key, legal_templates, realize,
)
from .types import Action, ActionType


@dataclass
class BRResult:
    player: int
    policy: TabularPolicy
    q_mean: Dict[InfoKey, Dict[ActionTemplate, float]]
    q_count: Dict[InfoKey, Dict[ActionTemplate, int]]
    value: float                       # greedy BR value vs fixed opponents
    n_info_sets: int


def best_response(
    player: int,
    fixed_strategies: Dict[int, Policy],
    config: GameConfig,
    num_episodes: int = 20000,
    seed: int = 0,
    epsilon: float = 0.25,
    epsilon_final: Optional[float] = None,
    x_bucket: int = 1,
    pos_clip: int = 3,
    q_init: float = 0.0,
    eval_sims: int = 4000,
) -> BRResult:
    """Find an approximate best response for ``player`` to ``fixed_strategies``.

    ``fixed_strategies`` must provide the *other two* players' policies (the
    target player's entry, if present, is ignored).
    """
    env = GameEnv(config)
    rng = random.Random(seed)

    q_sum: Dict[InfoKey, Dict[ActionTemplate, float]] = defaultdict(lambda: defaultdict(float))
    q_cnt: Dict[InfoKey, Dict[ActionTemplate, int]] = defaultdict(lambda: defaultdict(int))

    eps_final = epsilon if epsilon_final is None else epsilon_final
    visited: List[Tuple[InfoKey, ActionTemplate]] = []
    cur_eps = [epsilon]

    def q_mean(key: InfoKey, t: ActionTemplate) -> float:
        c = q_cnt[key][t]
        return q_sum[key][t] / c if c > 0 else q_init

    def greedy(key: InfoKey, legal_t: List[ActionTemplate]) -> ActionTemplate:
        best, best_v = legal_t[0], float("-inf")
        for t in legal_t:
            v = q_mean(key, t)
            if v > best_v:
                best_v, best = v, t
        return best

    def target_act(obs: Observation, legal: List[Action]) -> Action:
        key = info_key(obs, config, x_bucket, pos_clip)
        legal_t = legal_templates(legal)
        if rng.random() < cur_eps[0]:
            t = rng.choice(legal_t)
        else:
            t = greedy(key, legal_t)
        visited.append((key, t))
        return realize(t, obs)

    def act_fn(p: int, obs: Observation, legal: List[Action]) -> Action:
        if p == player:
            return target_act(obs, legal)
        return fixed_strategies[p].act(obs, legal, rng)

    for ep in range(num_episodes):
        frac = ep / max(1, num_episodes - 1)
        cur_eps[0] = epsilon + (eps_final - epsilon) * frac
        visited.clear()
        log = env.play_episode(act_fn, rng)
        ret = log.pnl[player]
        for key, t in visited:
            q_sum[key][t] += ret
            q_cnt[key][t] += 1

    # Build greedy tabular policy from the learned Q.
    br = TabularPolicy(config, x_bucket=x_bucket, pos_clip=pos_clip)
    q_mean_table: Dict[InfoKey, Dict[ActionTemplate, float]] = {}
    for key in q_sum:
        means = {t: q_sum[key][t] / q_cnt[key][t] for t in q_sum[key]}
        q_mean_table[key] = means
        br.set(key, max(means, key=means.get))

    # Clean greedy value vs the fixed opponents (fresh seed stream).
    profile = dict(fixed_strategies)
    profile[player] = br
    ev = evaluate_policy(profile, config, num_simulations=eval_sims,
                         seed=seed + 1, keep_logs=False)

    return BRResult(player=player, policy=br, q_mean=q_mean_table,
                    q_count={k: dict(v) for k, v in q_cnt.items()},
                    value=ev.mean_pnl[player], n_info_sets=len(q_sum))


@dataclass
class IBRHistory:
    iteration: int
    values: Dict[int, float]
    br_gains: Dict[int, float]         # value gained by best-responding vs current
    max_gain: float                    # exploitability proxy


@dataclass
class IBRResult:
    profile: Dict[int, Policy]
    history: List[IBRHistory] = field(default_factory=list)


def iterative_best_response(
    config: GameConfig,
    n_iters: int = 6,
    num_episodes: int = 20000,
    eval_sims: int = 4000,
    seed: int = 0,
    init_profile: Optional[Dict[int, Policy]] = None,
    x_bucket: int = 1,
    pos_clip: int = 3,
    epsilon: float = 0.25,
    epsilon_final: float = 0.02,
    verbose: bool = True,
) -> IBRResult:
    """Cyclic iterative best response.

    Starts from ``init_profile`` (default: all-random, an *unbiased* start --
    not silence, not aggression) and repeatedly replaces each player's policy
    with an approximate best response to the others. Tracks an exploitability
    proxy (max single-player improvement) per iteration.
    """
    if init_profile is None:
        profile: Dict[int, Policy] = {p: RandomPolicy() for p in (0, 1, 2)}
    else:
        profile = dict(init_profile)

    result = IBRResult(profile=profile)
    for it in range(n_iters):
        gains: Dict[int, float] = {}
        for p in (0, 1, 2):
            base = evaluate_policy(profile, config, num_simulations=eval_sims,
                                   seed=seed + 100 * it + p, keep_logs=False)
            br = best_response(
                p, profile, config, num_episodes=num_episodes,
                seed=seed + 1000 * it + p, epsilon=epsilon,
                epsilon_final=epsilon_final, x_bucket=x_bucket,
                pos_clip=pos_clip, eval_sims=eval_sims,
            )
            gains[p] = br.value - base.mean_pnl[p]
            profile[p] = br.policy

        final = evaluate_policy(profile, config, num_simulations=eval_sims,
                                seed=seed + 99 * (it + 1), keep_logs=False)
        hist = IBRHistory(iteration=it,
                          values={p: final.mean_pnl[p] for p in (0, 1, 2)},
                          br_gains=gains, max_gain=max(gains.values()))
        result.history.append(hist)
        if verbose:
            names = {0: "S", 1: "X", 2: "B"}
            vals = "  ".join(f"{names[p]}={final.mean_pnl[p]:+.2f}" for p in (0, 1, 2))
            print(f"[IBR it={it}] {vals}  | max_BR_gain={hist.max_gain:+.2f}")

    result.profile = profile
    return result
