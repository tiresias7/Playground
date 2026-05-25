"""Evaluate bots head-to-head with seat rotation.

`evaluate` takes a roster of (label, factory) seats, rotates the roster across
seats each round so no label is advantaged by position, and aggregates results
per label (labels may repeat -- e.g. two "Sharp" seats are pooled).
"""

from __future__ import annotations

import argparse
import random
import statistics as st

from .game import FiggieGame


def evaluate(roster, rounds: int, seed: int = 0, total_ticks: int = 2000):
    """roster: list of (label, factory) where factory(rng) -> Bot. Returns
    {label: {net, trade, pot, goal, win, n}} aggregated over rotated seats."""
    labels = [lbl for lbl, _ in roster]
    facs = [f for _, f in roster]
    n = len(facs)
    rng = random.Random(seed)

    agg: dict[str, dict] = {}
    for lbl in labels:
        agg.setdefault(lbl, {"net": [], "trade": [], "pot": [], "goal": [], "win": 0.0})

    for r in range(rounds):
        shift = r % n
        seat_id = [(s - shift) % n for s in range(n)]  # identity sitting at seat s
        bots = [facs[seat_id[s]](rng) for s in range(n)]
        g = FiggieGame(bots, total_ticks=total_ticks, rng=random.Random(rng.random()))
        res = g.play()
        gs = res.deck.goal_suit
        best = max(res.net.values())
        leaders = [s for s in range(n) if res.net[s] == best]
        for s in range(n):
            a = agg[labels[seat_id[s]]]
            a["net"].append(res.net[s])
            a["trade"].append(res.trading_pnl[s])
            a["pot"].append(res.payouts[s])
            a["goal"].append(res.final_hands[s][gs])
            if s in leaders:
                a["win"] += 1.0 / len(leaders)
    return agg, rounds


def report(roster, rounds: int, seed: int = 0, total_ticks: int = 2000, title: str = ""):
    agg, R = evaluate(roster, rounds, seed, total_ticks)
    if title:
        print(f"\n=== {title} ({R} rounds, {total_ticks} ticks) ===")
    print(f"{'label':<12}{'seats':>6}{'net/rd':>9}{'trade/rd':>9}"
          f"{'pot/rd':>8}{'goal@end':>9}{'win%':>7}")
    # All per-seat-per-round averages (agg lists hold one entry per seat-round).
    for lbl, a in sorted(agg.items(), key=lambda kv: -st.fmean(kv[1]["net"])):
        seats = len(a["net"]) // R
        print(f"{lbl:<12}{seats:>6}{st.fmean(a['net']):>9.2f}{st.fmean(a['trade']):>9.2f}"
              f"{st.fmean(a['pot']):>8.2f}{st.fmean(a['goal']):>9.2f}"
              f"{a['win']/len(a['net']):>7.1%}")
    return agg


def ecology(pool, rounds: int, seed: int = 0, total_ticks: int = 2000):
    """Each round samples a random 4-bot field from `pool` (a {label: factory}
    dict) and records net per type. Average net per type is its expected result
    in a random field -- a robust overall ranking. Returns {label: {net, win, n}}."""
    names = list(pool)
    rng = random.Random(seed)
    agg = {n: {"net": [], "win": 0.0} for n in names}
    for _ in range(rounds):
        seats = [rng.choice(names) for _ in range(4)]
        bots = [pool[seats[i]](rng) for i in range(4)]
        res = FiggieGame(bots, total_ticks=total_ticks,
                         rng=random.Random(rng.random())).play()
        best = max(res.net.values())
        leaders = [i for i in range(4) if res.net[i] == best]
        for i in range(4):
            agg[seats[i]]["net"].append(res.net[i])
            if i in leaders:
                agg[seats[i]]["win"] += 1.0 / len(leaders)

    print(f"\n=== ecology: {rounds} random 4-bot fields, {total_ticks} ticks ===")
    print(f"{'bot':<10}{'games':>7}{'net/game':>10}{'win%':>8}")
    for n in sorted(names, key=lambda x: -(st.fmean(agg[x]['net']) if agg[x]['net'] else 0)):
        g = len(agg[n]["net"])
        if g:
            print(f"{n:<10}{g:>7}{st.fmean(agg[n]['net']):>10.2f}{agg[n]['win']/g:>8.1%}")
    return agg


def _default_pool():
    from .bots import HeuristicBot, MMBot, RandomBot, SharpBot, ValueBot
    return {
        "MM": lambda r: MMBot(rng=random.Random(r.random())),
        "Sharp": lambda r: SharpBot(rng=random.Random(r.random())),
        "Value": lambda r: ValueBot(rng=random.Random(r.random())),
        "Heur": lambda r: HeuristicBot(rng=random.Random(r.random())),
        "Random": lambda r: RandomBot(rng=random.Random(r.random())),
    }


def main():
    ap = argparse.ArgumentParser(description="Figgie ecology tournament")
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--ticks", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    ecology(_default_pool(), args.rounds, args.seed, args.ticks)


if __name__ == "__main__":
    main()
