"""End-to-end settlement / PnL tests using scripted act functions."""

import random

from asym_market.config import GameConfig
from asym_market.env import GameEnv
from asym_market.evaluate import evaluate_policy
from asym_market.policies import RandomPolicy
from asym_market.types import Action, ActionType, Player, State


def _scripted_trade_act(player, obs, legal):
    """X posts an ask at 50; B lifts it; S waits. Deterministic vs any order."""
    legal_types = {a.type for a in legal}
    if player == Player.X:
        if obs.own_order is None and obs.best_ask is None:
            return Action(ActionType.POST_ASK, price=50)
        return Action(ActionType.WAIT)
    if player == Player.B:
        if obs.best_ask is not None and ActionType.TAKE_ASK in legal_types:
            return Action(ActionType.TAKE_ASK)
        return Action(ActionType.WAIT)
    return Action(ActionType.WAIT)


def test_random_value_trade_pnl():
    cfg = GameConfig(horizon=2)
    env = GameEnv(cfg)
    rng = random.Random(123)
    # RANDOM regime, x=80 -> V=80, k=1
    log = env.play_episode(_scripted_trade_act, rng,
                           instance=(State.RANDOM, 80, 1, 80))
    assert len(log.trades) == 1
    assert log.trades[0].price == 50
    # B bought 1 @50, holds to V=80
    assert log.positions[Player.B] == 1
    assert log.pnl[Player.B] == 1 * 80 - 50          # +30
    assert log.pnl[Player.X] == -1 * 80 + 50         # -30
    assert log.pnl[Player.S] == 0
    assert log.gross_buys[Player.B] == 1


def test_fix_value_overrides_x():
    cfg = GameConfig(horizon=2)
    env = GameEnv(cfg)
    rng = random.Random(7)
    # FIX regime: V=30 regardless of x=80
    log = env.play_episode(_scripted_trade_act, rng,
                           instance=(State.FIX, 80, 1, 30))
    assert log.pnl[Player.B] == 1 * 30 - 50          # -20
    assert log.pnl[Player.X] == -1 * 30 + 50         # +20


def test_buyer_penalty_when_obligation_unmet():
    cfg = GameConfig(horizon=3, penalty_per_unit=50.0, buyer_obligation=True)
    env = GameEnv(cfg)
    rng = random.Random(1)

    def all_wait(player, obs, legal):
        return Action(ActionType.WAIT)

    log = env.play_episode(all_wait, rng, instance=(State.RANDOM, 50, 2, 50))
    # B bought 0 of k=2 required -> penalty 2 * 50
    assert log.gross_buys[Player.B] == 0
    assert log.pnl[Player.B] == -100.0
    assert log.pnl[Player.S] == 0 and log.pnl[Player.X] == 0


def test_no_penalty_when_obligation_disabled():
    cfg = GameConfig(horizon=3, penalty_per_unit=50.0, buyer_obligation=False)
    env = GameEnv(cfg)
    rng = random.Random(1)

    def all_wait(player, obs, legal):
        return Action(ActionType.WAIT)

    log = env.play_episode(all_wait, rng, instance=(State.RANDOM, 50, 2, 50))
    assert log.pnl[Player.B] == 0.0


def test_cash_position_identity_holds():
    """Zero-sum check on V-settlement (ignoring penalty): sum of position*V+cash = 0."""
    cfg = GameConfig(horizon=4, buyer_obligation=False)
    res = evaluate_policy({0: RandomPolicy(), 1: RandomPolicy(), 2: RandomPolicy()},
                          cfg, num_simulations=200, seed=5)
    for log in res.logs:
        total = sum(log.positions[p] * log.value + log.cash[p] for p in (0, 1, 2))
        assert abs(total) < 1e-9        # every trade is matched -> conserved


def test_reproducible_seed():
    cfg = GameConfig(horizon=4)
    prof = {0: RandomPolicy(), 1: RandomPolicy(), 2: RandomPolicy()}
    r1 = evaluate_policy(prof, cfg, num_simulations=300, seed=42, keep_logs=False)
    r2 = evaluate_policy(prof, cfg, num_simulations=300, seed=42, keep_logs=False)
    assert r1.mean_pnl == r2.mean_pnl
