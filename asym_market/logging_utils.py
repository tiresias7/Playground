"""CSV export of episode logs and diagnostics (stdlib only)."""

from __future__ import annotations

import csv
import os
from typing import Dict, List

from .env import EpisodeLog

NAMES = {0: "S", 1: "X", 2: "B", None: "none"}


def write_episode_logs_csv(logs: List[EpisodeLog], out_dir: str, prefix: str = "") -> Dict[str, str]:
    """Write episodes / decisions / trades CSVs. Returns the paths written."""
    os.makedirs(out_dir, exist_ok=True)
    ep_path = os.path.join(out_dir, f"{prefix}episodes.csv")
    dec_path = os.path.join(out_dir, f"{prefix}decisions.csv")
    tr_path = os.path.join(out_dir, f"{prefix}trades.csv")

    with open(ep_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "state", "x", "value", "k", "first_poster",
                    "first_taker", "n_trades",
                    "pos_S", "pos_X", "pos_B",
                    "buys_B", "pnl_S", "pnl_X", "pnl_B"])
        for i, log in enumerate(logs):
            w.writerow([i, log.state, log.x, log.value, log.k,
                        NAMES[log.first_poster], NAMES[log.first_taker],
                        len(log.trades),
                        log.positions[0], log.positions[1], log.positions[2],
                        log.gross_buys[2],
                        round(log.pnl[0], 3), round(log.pnl[1], 3), round(log.pnl[2], 3)])

    with open(dec_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "decision_index", "t", "player",
                    "private_state", "private_x", "private_k",
                    "best_bid", "best_ask", "own_position",
                    "action_type", "action_price", "coerced", "trades_this_turn"])
        for i, log in enumerate(logs):
            for d in log.decisions:
                w.writerow([i, d.decision_index, d.t, NAMES[d.player],
                            d.private_state, d.private_x, d.private_k,
                            d.best_bid, d.best_ask, d.own_position,
                            d.action_type, d.action_price, int(d.coerced),
                            d.trades_this_turn])

    with open(tr_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "ts", "price", "buyer", "seller", "aggressor",
                    "maker_order_id"])
        for i, log in enumerate(logs):
            for tr in log.trades:
                w.writerow([i, tr.ts, tr.price, NAMES[tr.buyer], NAMES[tr.seller],
                            NAMES[tr.aggressor], tr.maker_order_id])

    return {"episodes": ep_path, "decisions": dec_path, "trades": tr_path}
