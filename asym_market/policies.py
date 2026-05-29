"""Policies and the information-set abstraction.

A *policy* maps an information set to an action. The information set is the
abstraction returned by :func:`info_key` -- this is the single knob that
controls how much state the solver reasons over (Phase-1 "narrowing").

Actions are stored as *templates* ``(ActionType, price)`` so a tabular policy
learned in one episode can be replayed in another: CANCEL templates get their
concrete ``order_id`` realised against the live observation.

No policy here encodes a strategic prior. ``RandomPolicy`` and
``AlwaysWaitPolicy`` exist only as baselines / exploration drivers; the
interesting policies are produced by the solver.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from .config import GameConfig
from .env import Observation
from .types import Action, ActionType, Player

# An action template is hashable and obs-independent.
ActionTemplate = Tuple[ActionType, Optional[int]]
InfoKey = Tuple


# ------------------------------------------------------------ action helpers
def template_of(action: Action) -> ActionTemplate:
    """Reduce a concrete action to its obs-independent template."""
    if action.type in (ActionType.POST_BID, ActionType.POST_ASK):
        return (action.type, action.price)
    return (action.type, None)


def realize(template: ActionTemplate, obs: Observation) -> Action:
    """Turn a template back into a concrete action against ``obs``."""
    atype, price = template
    if atype in (ActionType.POST_BID, ActionType.POST_ASK):
        return Action(atype, price=price)
    if atype is ActionType.CANCEL:
        oid = obs.own_order.order_id if obs.own_order else None
        return Action(ActionType.CANCEL, order_id=oid)
    return Action(atype)


def legal_templates(legal: List[Action]) -> List[ActionTemplate]:
    return [template_of(a) for a in legal]


# ----------------------------------------------------- info-set abstraction
def info_key(
    obs: Observation,
    config: GameConfig,
    x_bucket: int = 1,
    pos_clip: int = 3,
) -> InfoKey:
    """Hashable abstraction of an information set.

    ``x_bucket`` groups the number player's private value into buckets of this
    width (1 = no bucketing). ``pos_clip`` clips own position into
    ``[-pos_clip, pos_clip]``. Everything here is *observable* to the player.
    """
    if obs.player == Player.S:
        priv = ("S", obs.private_state.value)
    elif obs.player == Player.X:
        priv = ("X", obs.private_x // x_bucket)
    else:
        priv = ("B", obs.private_k)

    own = None
    if obs.own_order is not None:
        own = (obs.own_order.side.value, obs.own_order.price)

    pos = max(-pos_clip, min(pos_clip, obs.own_position))
    buys_left = obs.buys_left if obs.player == Player.B else None

    return (
        priv,
        obs.rounds_left,
        obs.best_bid,
        obs.best_ask,
        own,
        pos,
        buys_left,
    )


# -------------------------------------------------------------- policy types
class Policy(ABC):
    @abstractmethod
    def act(self, obs: Observation, legal: List[Action], rng: random.Random) -> Action:
        ...


class AlwaysWaitPolicy(Policy):
    """Baseline: never trades. Useful for the silent-market ablation."""

    def act(self, obs, legal, rng):
        return Action(ActionType.WAIT)


class RandomPolicy(Policy):
    """Uniform over legal actions. Pure exploration driver, no prior."""

    def act(self, obs, legal, rng):
        return rng.choice(legal)


class TabularPolicy(Policy):
    """Deterministic map: info_key -> action template.

    Unseen info sets fall back to ``default`` (WAIT). If a stored template is
    not currently legal it also falls back, so a policy is always well-defined.
    """

    def __init__(self, config: GameConfig, x_bucket: int = 1, pos_clip: int = 3,
                 default: ActionTemplate = (ActionType.WAIT, None)) -> None:
        self.config = config
        self.x_bucket = x_bucket
        self.pos_clip = pos_clip
        self.default = default
        self.table: Dict[InfoKey, ActionTemplate] = {}

    def key(self, obs: Observation) -> InfoKey:
        return info_key(obs, self.config, self.x_bucket, self.pos_clip)

    def set(self, key: InfoKey, template: ActionTemplate) -> None:
        self.table[key] = template

    def act(self, obs, legal, rng):
        key = self.key(obs)
        template = self.table.get(key, self.default)
        legal_t = legal_templates(legal)
        if template not in legal_t:
            template = (ActionType.WAIT, None)
        return realize(template, obs)


def make_act_fn(profile: Dict[int, Policy], rng: random.Random):
    """Combine three per-player policies into an environment ``act_fn``."""
    def act_fn(player: int, obs: Observation, legal: List[Action]) -> Action:
        return profile[player].act(obs, legal, rng)
    return act_fn
