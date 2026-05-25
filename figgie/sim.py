"""Run many Figgie rounds between bots and report results.

Usage:
    python -m figgie.sim --rounds 500 --seed 1
"""

from __future__ import annotations

import argparse
import random

from .bots import HeuristicBot, RandomBot
from .bots.base import Bot
from .game import FiggieGame


def run(make_bots, rounds: int, seed: int | None = None, verbose: bool = False):
    rng = random.Random(seed)
    names = [b.name for b in make_bots(rng)]
    n = len(names)
    totals = [0] * n
    wins = [0] * n  # rounds where the bot had the top net result (ties shared)

    for r in range(rounds):
        bots = make_bots(rng)
        game = FiggieGame(bots, rng=random.Random(rng.random()))
        result = game.play()
        best = max(result.net.values())
        leaders = [p for p in range(n) if result.net[p] == best]
        for p in range(n):
            totals[p] += result.net[p]
            if p in leaders:
                wins[p] += 1 / len(leaders)
        if verbose:
            print(f"round {r}: goal={result.deck.goal_suit.symbol} "
                  f"({result.deck.goal_count}) trades={result.num_trades} "
                  f"net={[result.net[p] for p in range(n)]}")

    print(f"\n{rounds} rounds, {n} players\n")
    print(f"{'seat':<6}{'bot':<18}{'total net':>12}{'avg/round':>12}{'win share':>12}")
    for p in range(n):
        print(f"{p:<6}{names[p]:<18}{totals[p]:>12}{totals[p]/rounds:>12.2f}"
              f"{wins[p]/rounds:>12.2%}")


def default_bots(rng: random.Random) -> list[Bot]:
    # Three value traders against one noise trader.
    return [
        HeuristicBot("heuristic-0", random.Random(rng.random())),
        HeuristicBot("heuristic-1", random.Random(rng.random())),
        HeuristicBot("heuristic-2", random.Random(rng.random())),
        RandomBot("random-0", rng=random.Random(rng.random())),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Figgie bot simulation")
    ap.add_argument("--rounds", type=int, default=500)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    run(default_bots, args.rounds, args.seed, args.verbose)


if __name__ == "__main__":
    main()
