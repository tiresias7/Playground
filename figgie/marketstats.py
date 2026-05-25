"""Market-microstructure diagnostics: trade frequency, traded-price
distributions by suit role, price discovery over the round, and per-bot trade
activity. Sanity-checks that the simulated market behaves like a real Figgie
market (goal suit trades at a premium, prices discover the goal over time, etc).

    python -m figgie.marketstats --rounds 400 --ticks 2000 --chart /tmp/figgie_market.png
"""

from __future__ import annotations

import argparse
import random
import statistics as st

from .bots import HeuristicBot, MMBot, RandomBot, ValueBot
from .cards import ALL_SUITS
from .game import FiggieGame


def _roster(field="mixed"):
    mm = ("MM", lambda r: MMBot(rng=random.Random(r.random())))
    val = ("Value", lambda r: ValueBot(rng=random.Random(r.random())))
    heur = ("Heur", lambda r: HeuristicBot(rng=random.Random(r.random())))
    rnd = ("Random", lambda r: RandomBot(rng=random.Random(r.random())))
    if field == "skilled":   # no noise trader
        return [mm, val, heur, ("MM2", lambda r: MMBot(rng=random.Random(r.random())))]
    return [mm, val, heur, rnd]


def _pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))
    return xs[k]


def collect(rounds, ticks, seed=0, field="mixed"):
    roster = _roster(field)
    labels = [l for l, _ in roster]
    facs = [f for _, f in roster]
    n = len(facs)
    rng = random.Random(seed)

    trades_per_round = []
    price = {"goal": [], "common": [], "off": []}
    # price by progress bucket (fifths) for goal and common
    nb = 5
    prog = {"goal": [[] for _ in range(nb)], "common": [[] for _ in range(nb)]}
    timing = [0] * nb  # share of trades by round-progress bucket
    perbot = {l: {"trades": 0, "aggress": 0, "buy_px": [], "sell_px": [],
                  "bought": 0, "sold": 0} for l in labels}
    dealt_goal = {l: [] for l in labels}
    final_goal = {l: [] for l in labels}

    for r in range(rounds):
        shift = r % n
        seat_lbl = [labels[(s - shift) % n] for s in range(n)]
        bots = [facs[(s - shift) % n](rng) for s in range(n)]
        res = FiggieGame(bots, total_ticks=ticks,
                         rng=random.Random(rng.random())).play()
        T = res.trades
        m = len(T)
        trades_per_round.append(m)
        goal, common = res.deck.goal_suit, res.deck.common_suit
        for l in labels:
            seat = seat_lbl.index(l)
            dealt_goal[l].append(res.dealt_hands[seat][goal])
            final_goal[l].append(res.final_hands[seat][goal])
        for i, t in enumerate(T):
            role = "goal" if t.suit is goal else ("common" if t.suit is common else "off")
            price[role].append(t.price)
            b = min(nb - 1, int(nb * i / m)) if m else 0
            timing[b] += 1
            if role in prog:
                prog[role][b].append(t.price)
            for who, side in ((t.buyer, "buy"), (t.seller, "sell")):
                lbl = seat_lbl[who]
                perbot[lbl]["trades"] += 1
                perbot[lbl][f"{side}_px"].append(t.price)
                perbot[lbl]["bought" if side == "buy" else "sold"] += 1
            perbot[seat_lbl[t.aggressor]]["aggress"] += 1

    return dict(labels=labels, rounds=rounds, ticks=ticks,
                trades_per_round=trades_per_round, price=price, prog=prog,
                timing=timing, perbot=perbot, dealt_goal=dealt_goal,
                final_goal=final_goal)


def report(d):
    tpr = d["trades_per_round"]
    print(f"\n=== market stats: {d['rounds']} rounds, {d['ticks']} ticks, "
          f"field={'/'.join(d['labels'])} ===\n")

    print("-- trade frequency (per round) --")
    print(f"  mean {st.fmean(tpr):.1f}  median {_pct(tpr,50)}  "
          f"p10 {_pct(tpr,10)}  p90 {_pct(tpr,90)}  min {min(tpr)}  max {max(tpr)}")
    total = sum(d["timing"]) or 1
    shares = "  ".join(f"{100*c/total:4.1f}%" for c in d["timing"])
    print(f"  trades by round progress (early->late, fifths): {shares}\n")

    print("-- traded price by suit role --")
    print(f"  {'role':<8}{'n':>7}{'mean':>7}{'med':>6}{'p10':>6}{'p90':>6}")
    for role in ("goal", "common", "off"):
        xs = d["price"][role]
        if xs:
            print(f"  {role:<8}{len(xs):>7}{st.fmean(xs):>7.2f}{_pct(xs,50):>6}"
                  f"{_pct(xs,10):>6}{_pct(xs,90):>6}")
    print()

    print("-- price discovery (mean price by round progress, fifths) --")
    for role in ("goal", "common"):
        cells = []
        for b in range(len(d["prog"][role])):
            xs = d["prog"][role][b]
            cells.append(f"{st.fmean(xs):5.1f}" if xs else "  -  ")
        print(f"  {role:<8} {'  '.join(cells)}")
    print()

    print("-- per-bot trade activity --")
    print(f"  {'bot':<8}{'trades':>8}{'aggress%':>9}{'avgBuy':>8}{'avgSell':>8}"
          f"{'goalDealt':>10}{'goalEnd':>9}")
    for l in d["labels"]:
        p = d["perbot"][l]
        tr = p["trades"]  # number of trades this bot participated in
        agg = 100 * p["aggress"] / tr if tr else 0
        ab = st.fmean(p["buy_px"]) if p["buy_px"] else float("nan")
        as_ = st.fmean(p["sell_px"]) if p["sell_px"] else float("nan")
        print(f"  {l:<8}{tr:>8}{agg:>8.0f}%{ab:>8.2f}{as_:>8.2f}"
              f"{st.fmean(d['dealt_goal'][l]):>10.2f}{st.fmean(d['final_goal'][l]):>9.2f}")
    print()


def chart(d, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    # 1) price distribution by role
    bins = range(0, 21)
    for role, c in (("goal", "tab:green"), ("common", "tab:red"), ("off", "tab:gray")):
        xs = [min(20, x) for x in d["price"][role]]
        if xs:
            ax[0].hist(xs, bins=bins, alpha=0.55, label=role, color=c, density=True)
    ax[0].set_title("Traded price distribution by suit role")
    ax[0].set_xlabel("price"); ax[0].set_ylabel("density"); ax[0].legend()

    # 2) price discovery over the round
    nb = len(d["prog"]["goal"])
    x = [(b + 0.5) / nb for b in range(nb)]
    for role, c in (("goal", "tab:green"), ("common", "tab:red")):
        y = [st.fmean(d["prog"][role][b]) if d["prog"][role][b] else None for b in range(nb)]
        ax[1].plot(x, y, marker="o", label=role, color=c)
    ax[1].set_title("Mean price vs round progress")
    ax[1].set_xlabel("round progress (early->late)"); ax[1].set_ylabel("mean price")
    ax[1].legend()

    # 3) trades-per-round histogram
    ax[2].hist(d["trades_per_round"], bins=20, color="tab:blue", alpha=0.8)
    ax[2].set_title("Trades per round")
    ax[2].set_xlabel("trades"); ax[2].set_ylabel("rounds")

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"chart saved to {path}")


def main():
    ap = argparse.ArgumentParser(description="Figgie market-microstructure stats")
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--ticks", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chart", type=str, default=None)
    ap.add_argument("--field", choices=["mixed", "skilled"], default="mixed")
    args = ap.parse_args()
    d = collect(args.rounds, args.ticks, args.seed, args.field)
    report(d)
    if args.chart:
        chart(d, args.chart)


if __name__ == "__main__":
    main()
