"""Sanity-check diagnostics for the Figgie engine, inference, and bots.

Usage:
    python -m figgie.analyze --rounds 300 --seed 1

Reports:
  * hand-only inference accuracy (does the posterior point at the true goal?)
  * price discovery (does the goal suit trade at a premium?)
  * goal-suit accumulation by bot type
  * P&L decomposition (trading vs pot) and net distribution
  * conservation check (net must sum to 0 every round)
  * an all-heuristic symmetry run (no seat should have an edge)
"""

from __future__ import annotations

import argparse
import random
import statistics as stats

from .bots import HeuristicBot, RandomBot
from .bots.base import Bot
from .cards import ALL_SUITS
from .game import FiggieGame
from .inference import goal_posterior
from .sim import default_bots


def _argmax_suit(post):
    return max(ALL_SUITS, key=lambda s: post[s])


def inference_only(rounds: int, rng: random.Random, players: int = 4):
    """Deal hands (no trading) and test whether the hand-only posterior finds
    the goal suit. Pure check of inference.py."""
    per = 40 // players
    hits = color_hits = 0
    p_true = []
    samples = 0
    for _ in range(rounds):
        game = FiggieGame(default_bots(rng), rng=random.Random(rng.random()))
        game._deal()
        goal = game.deck.goal_suit
        for p in range(players):
            hand = game.hands[p]
            post = goal_posterior(hand)
            guess = _argmax_suit(post)
            hits += guess is goal
            color_hits += guess.is_black == goal.is_black
            p_true.append(post[goal])
            samples += 1
    print("== hand-only goal inference ==")
    print(f"  samples           : {samples}")
    print(f"  MAP accuracy      : {hits/samples:6.1%}   (random guess 25.0%)")
    print(f"  correct color     : {color_hits/samples:6.1%}   (random 50.0%)")
    print(f"  avg P(true goal)  : {stats.fmean(p_true):6.3f}   (uninformed 0.25)")
    print()


def play_rounds(make_bots, rounds: int, rng: random.Random):
    n = len(make_bots(rng))
    results = []
    for _ in range(rounds):
        game = FiggieGame(make_bots(rng), rng=random.Random(rng.random()))
        results.append(game.play())
    return results, n


def report(results, n, names):
    rounds = len(results)

    # --- conservation: net must sum to 0 each round -----------------
    bad = [i for i, r in enumerate(results) if sum(r.net.values()) != 0]
    print("== conservation ==")
    print(f"  rounds with nonzero net sum : {len(bad)}  (expect 0)")
    print()

    # --- trading activity -------------------------------------------
    trades = [r.num_trades for r in results]
    no_trade = sum(t == 0 for t in trades)
    print("== trading activity ==")
    print(f"  avg trades/round  : {stats.fmean(trades):6.1f}")
    print(f"  min / max         : {min(trades)} / {max(trades)}")
    print(f"  rounds w/ 0 trades: {no_trade}")
    print()

    # --- price discovery: goal vs non-goal traded prices ------------
    goal_prices, other_prices = [], []
    for r in results:
        goal = r.deck.goal_suit
        for t in r.trades:
            (goal_prices if t.suit is goal else other_prices).append(t.price)
    print("== price discovery (traded prices) ==")
    if goal_prices:
        print(f"  goal suit  : n={len(goal_prices):5d}  mean={stats.fmean(goal_prices):5.2f}")
    if other_prices:
        print(f"  non-goal   : n={len(other_prices):5d}  mean={stats.fmean(other_prices):5.2f}")
    print()

    # --- per-seat P&L decomposition and distribution ----------------
    print("== per-seat results ==")
    hdr = f"  {'seat':<4}{'bot':<14}{'net/rd':>8}{'trade/rd':>9}{'pot/rd':>8}"
    hdr += f"{'goal@end':>9}{'std':>8}{'%pos':>7}"
    print(hdr)
    for p in range(n):
        net = [r.net[p] for r in results]
        tr = [r.trading_pnl[p] for r in results]
        pot = [r.payouts[p] for r in results]
        goal_held = [r.final_hands[p][r.deck.goal_suit] for r in results]
        pos = sum(x > 0 for x in net) / rounds
        print(f"  {p:<4}{names[p]:<14}{stats.fmean(net):>8.2f}{stats.fmean(tr):>9.2f}"
              f"{stats.fmean(pot):>8.2f}{stats.fmean(goal_held):>9.2f}"
              f"{stats.pstdev(net):>8.1f}{pos:>7.1%}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Figgie sanity-check diagnostics")
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    inference_only(args.rounds, rng)

    names = [b.name for b in default_bots(rng)]
    results, n = play_rounds(default_bots, args.rounds, rng)
    print(f"== main matchup: {args.rounds} rounds, {n} players ==\n")
    report(results, n, names)

    # symmetry: four identical value traders -> no seat should dominate
    def all_heuristic(r: random.Random) -> list[Bot]:
        return [HeuristicBot(f"heur-{i}", random.Random(r.random())) for i in range(4)]

    hnames = [b.name for b in all_heuristic(rng)]
    hres, hn = play_rounds(all_heuristic, args.rounds, rng)
    print(f"== symmetry check: 4 identical heuristics, {args.rounds} rounds ==\n")
    report(hres, hn, hnames)


if __name__ == "__main__":
    main()
