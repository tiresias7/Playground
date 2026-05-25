"""Order-flow inference: reweight the goal-suit belief using observed trades.

Players accumulate the goal suit and dump the abundant common suit, which is the
goal's same-color partner. So a suit that opponents aggressively *buy* is more
likely the goal, and a suit they aggressively *sell* is more likely the common
suit (evidence its partner is the goal). We summarize this as a net aggressive
buy count per suit and fold it into the hand-only posterior multiplicatively:

    weight(s) ∝ P_hand(s) * exp(lam * clip(net[s] - net[partner(s)], ±cap))

The partner term sharpens the signal: the goal stands out from the common suit
of the same color. `lam` and `cap` are tuned by self-play.
"""

from __future__ import annotations

import math

from .cards import ALL_SUITS


def combine_goal_size(hand_joint, net, lam: float, cap: float):
    """Reweight a joint {suit:{G:prob}} posterior by net aggressive flow."""
    mult = {}
    for s in ALL_SUITS:
        score = net[s] - net[s.partner]
        score = max(-cap, min(cap, score))
        mult[s] = math.exp(lam * score)

    out = {s: {} for s in ALL_SUITS}
    total = 0.0
    for s in ALL_SUITS:
        for G, p in hand_joint[s].items():
            v = p * mult[s]
            out[s][G] = v
            total += v
    if total > 0:
        for s in ALL_SUITS:
            for G in out[s]:
                out[s][G] /= total
    return out
