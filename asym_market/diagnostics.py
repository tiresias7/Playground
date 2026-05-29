"""Behavioural diagnostics extracted from episode logs.

These functions answer the Phase-2 questions directly and *descriptively* --
they measure what a profile does, they do not assert what it should do:

  * who is the first poster / first taker (and how often nobody posts)
  * P(post) conditional on private info (state for S, x for X, k for B)
  * P(post at t=0) and first-action distribution
  * aggression (take rate) as a function of remaining time
  * the buyer's resting bid price over time
  * the ex-ante value of *being* the first poster vs waiting

Everything is computed from the same ``EpisodeLog`` objects produced by
``evaluate_policy`` so behaviour and payoff always refer to identical runs.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from .config import GameConfig
from .env import EpisodeLog
from .types import Player

_POST = {"POST_BID", "POST_ASK"}
_TAKE = {"TAKE_BID", "TAKE_ASK"}
NAMES = {0: "S", 1: "X", 2: "B", None: "none"}


def _first_decision(log: EpisodeLog, player: int):
    for d in log.decisions:
        if d.player == player:
            return d
    return None


def _player_posts(log: EpisodeLog, player: int) -> bool:
    return any(d.player == player and d.action_type in _POST for d in log.decisions)


def first_poster_distribution(logs: List[EpisodeLog]) -> Dict[Optional[int], float]:
    counts: Dict[Optional[int], int] = defaultdict(int)
    for log in logs:
        counts[log.first_poster] += 1
    n = len(logs)
    return {k: counts[k] / n for k in (0, 1, 2, None)}


def first_taker_distribution(logs: List[EpisodeLog]) -> Dict[Optional[int], float]:
    counts: Dict[Optional[int], int] = defaultdict(int)
    for log in logs:
        counts[log.first_taker] += 1
    n = len(logs)
    return {k: counts[k] / n for k in (0, 1, 2, None)}


def first_action_distribution(logs: List[EpisodeLog]) -> Dict[int, Dict[str, float]]:
    """Per player: distribution over the *type* of their first action."""
    counts: Dict[int, Dict[str, int]] = {p: defaultdict(int) for p in (0, 1, 2)}
    totals: Dict[int, int] = defaultdict(int)
    for log in logs:
        for p in (0, 1, 2):
            d = _first_decision(log, p)
            if d is not None:
                counts[p][d.action_type] += 1
                totals[p] += 1
    out: Dict[int, Dict[str, float]] = {}
    for p in (0, 1, 2):
        tot = max(1, totals[p])
        out[p] = {a: c / tot for a, c in counts[p].items()}
    return out


def posting_prob_by_private(logs: List[EpisodeLog]) -> Dict:
    """P(player ever posts) and P(posts at t=0) conditioned on private info."""
    # accumulators keyed by (player, private_value)
    ever = defaultdict(int)
    at_t0 = defaultdict(int)
    is_first = defaultdict(int)
    total = defaultdict(int)

    for log in logs:
        priv = {Player.S: log.state, Player.X: log.x, Player.B: log.k}
        for p in (0, 1, 2):
            key = (p, priv[p])
            total[key] += 1
            if _player_posts(log, p):
                ever[key] += 1
            posted_t0 = any(d.player == p and d.t == 0 and d.action_type in _POST
                            for d in log.decisions)
            if posted_t0:
                at_t0[key] += 1
            if log.first_poster == p:
                is_first[key] += 1

    out: Dict = {0: {}, 1: {}, 2: {}}
    for (p, val), tot in total.items():
        out[p][val] = {
            "n": tot,
            "p_ever_post": ever[(p, val)] / tot,
            "p_post_t0": at_t0[(p, val)] / tot,
            "p_first_poster": is_first[(p, val)] / tot,
        }
    return out


def aggression_by_rounds_left(logs: List[EpisodeLog], horizon: int) -> Dict[int, Dict[int, float]]:
    """Per player: take-rate as a function of remaining rounds.

    ``rounds_left`` is recovered from ``t`` (= horizon - t).
    """
    take = {p: defaultdict(int) for p in (0, 1, 2)}
    tot = {p: defaultdict(int) for p in (0, 1, 2)}
    for log in logs:
        for d in log.decisions:
            rl = horizon - d.t
            tot[d.player][rl] += 1
            if d.action_type in _TAKE:
                take[d.player][rl] += 1
    out: Dict[int, Dict[int, float]] = {}
    for p in (0, 1, 2):
        out[p] = {rl: take[p][rl] / tot[p][rl] for rl in sorted(tot[p]) if tot[p][rl]}
    return out


def buyer_bid_over_time(logs: List[EpisodeLog], horizon: int) -> Dict[int, Optional[float]]:
    """Average best-bid price *posted by B* at each round (None if never)."""
    total = defaultdict(float)
    count = defaultdict(int)
    for log in logs:
        for d in log.decisions:
            if d.player == Player.B and d.action_type == "POST_BID" and d.action_price is not None:
                total[d.t] += d.action_price
                count[d.t] += 1
    return {t: (total[t] / count[t] if count[t] else None) for t in range(horizon)}


def first_poster_value(logs: List[EpisodeLog]) -> Dict:
    """Ex-ante value of *being* the first poster vs not.

    Returns mean PnL of the player who posted first, mean PnL of players who did
    NOT post first in those same games, and the per-player conditional means.
    """
    fp_pnls: List[float] = []
    other_pnls: List[float] = []
    by_player_fp = defaultdict(list)
    by_player_wait = defaultdict(list)

    for log in logs:
        fp = log.first_poster
        if fp is None:
            continue
        fp_pnls.append(log.pnl[fp])
        by_player_fp[fp].append(log.pnl[fp])
        for p in (0, 1, 2):
            if p != fp:
                other_pnls.append(log.pnl[p])
                by_player_wait[p].append(log.pnl[p])

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    return {
        "n_with_poster": len(fp_pnls),
        "mean_pnl_first_poster": mean(fp_pnls),
        "mean_pnl_non_first_poster": mean(other_pnls),
        "first_poster_premium": mean(fp_pnls) - mean(other_pnls),
        "by_player_first_poster_mean": {p: mean(by_player_fp[p]) for p in (0, 1, 2)},
        "by_player_when_not_first_mean": {p: mean(by_player_wait[p]) for p in (0, 1, 2)},
    }


def behavioral_report(logs: List[EpisodeLog], config: GameConfig) -> Dict:
    """Bundle all diagnostics for a profile into one dict."""
    return {
        "n_episodes": len(logs),
        "silent_fraction": sum(1 for l in logs if l.first_poster is None and not l.trades) / max(1, len(logs)),
        "no_trade_fraction": sum(1 for l in logs if not l.trades) / max(1, len(logs)),
        "first_poster_dist": first_poster_distribution(logs),
        "first_taker_dist": first_taker_distribution(logs),
        "first_action_dist": first_action_distribution(logs),
        "posting_by_private": posting_prob_by_private(logs),
        "aggression_by_rounds_left": aggression_by_rounds_left(logs, config.horizon),
        "buyer_bid_over_time": buyer_bid_over_time(logs, config.horizon),
        "first_poster_value": first_poster_value(logs),
    }


def print_report(report: Dict) -> None:
    """Human-readable dump of a behavioural report."""
    print(f"episodes={report['n_episodes']}  "
          f"no_trade={report['no_trade_fraction']:.2%}  "
          f"silent={report['silent_fraction']:.2%}")

    fp = report["first_poster_dist"]
    print("first poster : " + "  ".join(f"{NAMES[k]}={fp[k]:.2%}" for k in (0, 1, 2, None)))
    ft = report["first_taker_dist"]
    print("first taker  : " + "  ".join(f"{NAMES[k]}={ft[k]:.2%}" for k in (0, 1, 2, None)))

    fpv = report["first_poster_value"]
    print(f"first-poster premium = {fpv['first_poster_premium']:+.2f} "
          f"(poster {fpv['mean_pnl_first_poster']:+.2f} vs "
          f"others {fpv['mean_pnl_non_first_poster']:+.2f})")

    print("S post-prob by state:")
    for val, m in sorted(report["posting_by_private"][0].items(), key=lambda kv: str(kv[0])):
        print(f"  state={val:<7} ever={m['p_ever_post']:.2%} t0={m['p_post_t0']:.2%} first={m['p_first_poster']:.2%}")
    print("X post-prob by x (sampled):")
    items = sorted(report["posting_by_private"][1].items())
    for val, m in items[:: max(1, len(items) // 8)]:
        print(f"  x={val:<4} ever={m['p_ever_post']:.2%} t0={m['p_post_t0']:.2%} first={m['p_first_poster']:.2%}")
    print("B post-prob by k:")
    for val, m in sorted(report["posting_by_private"][2].items()):
        print(f"  k={val} ever={m['p_ever_post']:.2%} t0={m['p_post_t0']:.2%} first={m['p_first_poster']:.2%}")
