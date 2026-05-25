"""Evaluate bots head-to-head with seat rotation.

`evaluate` takes a roster of (label, factory) seats, rotates the roster across
seats each round so no label is advantaged by position, and aggregates results
per label (labels may repeat -- e.g. two "Sharp" seats are pooled).
"""

from __future__ import annotations

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
