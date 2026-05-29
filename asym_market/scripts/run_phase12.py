"""Phase 1-2 driver.

Runs the environment + best-response/iterative-best-response solver on a small,
tractable game instance, prints the behavioural diagnostics that answer the
Phase-2 questions, and writes CSV logs.

This script does NOT assert conclusions. It reports what the search found.

Usage::

    python -m asym_market.scripts.run_phase12 [--quick]
"""

from __future__ import annotations

import argparse
import os
import sys

# allow running as a plain script too
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from asym_market import (  # noqa: E402
    GameConfig, RandomPolicy, behavioral_report, best_response,
    evaluate_policy, iterative_best_response, print_report,
    write_episode_logs_csv,
)
from asym_market.diagnostics import NAMES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer episodes (smoke run)")
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "output"))
    args = ap.parse_args()

    episodes = 8000 if args.quick else 30000
    iters = 4 if args.quick else 6
    eval_sims = 3000 if args.quick else 6000

    # A deliberately small, tractable instance. Coarse posting grid + bucketed x
    # keep the info-set space searchable; the tick grid stays fine.
    cfg = GameConfig(
        horizon=args.horizon,
        post_price_grid=(0, 20, 40, 60, 80, 100),
        penalty_per_unit=50.0,
    )
    x_bucket, pos_clip = 20, 2

    print("=" * 70)
    print(f"GAME  T={cfg.horizon}  post_grid={cfg.post_price_grid}  "
          f"penalty/unit={cfg.penalty_per_unit}  E[V]={cfg.ev_value:.1f}")
    print(f"SOLVE x_bucket={x_bucket}  pos_clip={pos_clip}  "
          f"episodes/BR={episodes}  IBR_iters={iters}")
    print("=" * 70)

    # ---- Iterative best response from an UNBIASED (all-random) start --------
    print("\n## Iterative best response (approximate equilibrium search)\n")
    ibr = iterative_best_response(
        cfg, n_iters=iters, num_episodes=episodes, eval_sims=eval_sims, seed=0,
        x_bucket=x_bucket, pos_clip=pos_clip,
    )
    profile = ibr.profile

    print("\nConvergence (exploitability proxy = max single-player BR gain):")
    for h in ibr.history:
        print(f"  it={h.iteration}  max_BR_gain={h.max_gain:+.2f}")

    # ---- Evaluate the resulting profile + behavioural diagnostics ----------
    print("\n## Profile evaluation + behavioural diagnostics\n")
    res = evaluate_policy(profile, cfg, num_simulations=20000, seed=777, keep_logs=True)
    print("Mean PnL:", res.summary())
    print()
    report = behavioral_report(res.logs, cfg)
    print_report(report)

    # first-action distribution (answers "is t=0 a post?")
    print("\nFirst-action distribution (per player):")
    fad = report["first_action_dist"]
    for p in (0, 1, 2):
        items = sorted(fad[p].items(), key=lambda kv: -kv[1])
        line = "  ".join(f"{a}:{v:.0%}" for a, v in items[:4])
        print(f"  {NAMES[p]}: {line}")

    # ---- CSV output --------------------------------------------------------
    out_dir = os.path.abspath(args.out)
    paths = write_episode_logs_csv(res.logs[:5000], out_dir, prefix="ibr_")
    print("\nCSV written:")
    for k, v in paths.items():
        print(f"  {k}: {v}")

    print("\nNOTE: findings are emergent from search; see report.md for the")
    print("structured write-up and caveats. No behaviour is hard-coded.")


if __name__ == "__main__":
    main()
