"""The game environment.

Drives a full episode: samples hidden variables, runs ``T`` rounds in which all
three players act once each in a freshly randomised order, applies actions to
the order book, and settles final PnL. Produces an exhaustive ``EpisodeLog``.

The environment is strategy-agnostic. It takes an ``act_fn`` callable
``(player, Observation, legal_actions) -> Action`` so the same engine serves
hand-written policies, tabular policies and the best-response explorer.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .config import GameConfig
from .order_book import OrderBook
from .types import Action, ActionType, Order, Player, State, Trade

ActFn = Callable[[int, "Observation", List[Action]], Action]


# --------------------------------------------------------------------- views
@dataclass
class Observation:
    """Everything a player legitimately knows on their turn = its info set."""

    player: int
    t: int                       # current round index (0-based)
    rounds_left: int
    decision_index: int          # global turn counter (price-time priority)

    # private info -- exactly one of these is non-None per player role
    private_state: Optional[State]
    private_x: Optional[int]
    private_k: Optional[int]

    # public book view
    best_bid: Optional[int]
    best_ask: Optional[int]
    book: dict

    # own private accounting
    own_position: int
    own_cash: float
    own_order: Optional[Order]

    # buyer-only obligation tracking (None for S and X)
    buys_done: Optional[int]
    buys_left: Optional[int]


# --------------------------------------------------------------------- logging
@dataclass
class DecisionRecord:
    decision_index: int
    t: int
    player: int
    private_state: Optional[str]
    private_x: Optional[int]
    private_k: Optional[int]
    best_bid: Optional[int]
    best_ask: Optional[int]
    own_position: int
    action_type: str
    action_price: Optional[int]
    coerced: bool                # True if an illegal action was coerced to WAIT
    trades_this_turn: int


@dataclass
class EpisodeLog:
    config_horizon: int
    state: str
    x: int
    value: int                   # realised V
    k: int
    decisions: List[DecisionRecord] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    book_snapshots: List[dict] = field(default_factory=list)
    first_poster: Optional[int] = None
    first_taker: Optional[int] = None
    first_post_decision_index: Optional[int] = None
    positions: Dict[int, int] = field(default_factory=dict)
    cash: Dict[int, float] = field(default_factory=dict)
    gross_buys: Dict[int, int] = field(default_factory=dict)
    pnl: Dict[int, float] = field(default_factory=dict)


# --------------------------------------------------------------- player state
@dataclass
class _PlayerState:
    position: int = 0
    cash: float = 0.0
    gross_buys: int = 0


# -------------------------------------------------------------- legal actions
def legal_actions(obs: Observation, config: GameConfig) -> List[Action]:
    """Enumerate the legal actions for ``obs.player`` given the public book.

    This is the single source of truth for action legality and is used both by
    the environment (to validate) and by the solver (to enumerate). It encodes
    *rules*, never strategy.
    """
    acts: List[Action] = [Action(ActionType.WAIT)]
    has_order = obs.own_order is not None
    can_sell = config.allow_short or obs.own_position > 0

    if not has_order:  # max_outstanding_per_player == 1 in Phase 1
        for p in config.post_price_grid:
            acts.append(Action(ActionType.POST_BID, price=p))
            if can_sell:
                acts.append(Action(ActionType.POST_ASK, price=p))

    if obs.best_ask is not None and not _best_is_own(obs, side="ask"):
        acts.append(Action(ActionType.TAKE_ASK))  # buy from best ask
    if obs.best_bid is not None and not _best_is_own(obs, side="bid") and can_sell:
        acts.append(Action(ActionType.TAKE_BID))  # sell into best bid

    if has_order and obs.own_order is not None:
        acts.append(Action(ActionType.CANCEL, order_id=obs.own_order.order_id))

    return acts


def _best_is_own(obs: Observation, side: str) -> bool:
    bucket = obs.book["asks"] if side == "ask" else obs.book["bids"]
    if not bucket:
        return False
    _, owner, _ = bucket[0]
    return owner == obs.player


# ------------------------------------------------------------------- the game
class GameEnv:
    def __init__(self, config: GameConfig) -> None:
        self.config = config

    def sample_instance(self, rng: random.Random) -> tuple:
        """Draw (state, x, k, V) from the prior."""
        state = State.FIX if rng.random() < self.config.p_fix else State.RANDOM
        x = rng.choice(self.config.x_values)
        k = rng.choice(self.config.k_values)
        value = self.config.fix_value if state is State.FIX else x
        return state, x, k, value

    def play_episode(
        self,
        act_fn: ActFn,
        rng: random.Random,
        instance: Optional[tuple] = None,
    ) -> EpisodeLog:
        """Run one full episode. ``instance`` may pin (state, x, k, V)."""
        cfg = self.config
        if instance is None:
            state, x, k, value = self.sample_instance(rng)
        else:
            state, x, k, value = instance

        book = OrderBook()
        pstate = {p: _PlayerState() for p in (0, 1, 2)}
        log = EpisodeLog(config_horizon=cfg.horizon, state=state.value, x=x,
                         value=value, k=k)

        decision_index = 0
        for t in range(cfg.horizon):
            order = [0, 1, 2]
            rng.shuffle(order)
            for player in order:
                obs = self._observe(player, t, decision_index, book, pstate,
                                     state, x, k)
                log.book_snapshots.append(book.snapshot())
                legal = legal_actions(obs, cfg)
                action = act_fn(player, obs, legal)
                coerced = action not in legal
                if coerced:
                    action = Action(ActionType.WAIT)

                trades = self._apply(action, player, book, pstate, decision_index)
                self._settle_trades(trades, pstate, log)

                if action.type in (ActionType.POST_BID, ActionType.POST_ASK):
                    if log.first_poster is None:
                        log.first_poster = player
                        log.first_post_decision_index = decision_index

                log.decisions.append(DecisionRecord(
                    decision_index=decision_index, t=t, player=player,
                    private_state=state.value if player == Player.S else None,
                    private_x=x if player == Player.X else None,
                    private_k=k if player == Player.B else None,
                    best_bid=obs.best_bid, best_ask=obs.best_ask,
                    own_position=obs.own_position,
                    action_type=action.type.value, action_price=action.price,
                    coerced=coerced, trades_this_turn=len(trades),
                ))
                decision_index += 1

        self._settle(log, pstate, value, k)
        return log

    # ----------------------------------------------------------- internals
    def _observe(self, player, t, decision_index, book, pstate, state, x, k):
        cfg = self.config
        own_orders = book.owner_orders(player)
        own_order = own_orders[0] if own_orders else None
        ps = pstate[player]
        buys_done = ps.gross_buys if player == Player.B else None
        buys_left = (max(0, k - ps.gross_buys)
                     if (player == Player.B and cfg.buyer_obligation) else
                     (0 if player == Player.B else None))
        return Observation(
            player=player, t=t, rounds_left=cfg.horizon - t,
            decision_index=decision_index,
            private_state=state if player == Player.S else None,
            private_x=x if player == Player.X else None,
            private_k=k if player == Player.B else None,
            best_bid=book.best_bid_price(), best_ask=book.best_ask_price(),
            book=book.snapshot(),
            own_position=ps.position, own_cash=ps.cash, own_order=own_order,
            buys_done=buys_done, buys_left=buys_left,
        )

    def _apply(self, action, player, book, pstate, ts) -> List[Trade]:
        if action.type is ActionType.WAIT:
            return []
        if action.type is ActionType.POST_BID:
            _, trades = book.post_bid(player, action.price, ts)
            return trades
        if action.type is ActionType.POST_ASK:
            _, trades = book.post_ask(player, action.price, ts)
            return trades
        if action.type is ActionType.TAKE_ASK:
            tr = book.take_ask(player, ts)
            return [tr] if tr else []
        if action.type is ActionType.TAKE_BID:
            tr = book.take_bid(player, ts)
            return [tr] if tr else []
        if action.type is ActionType.CANCEL:
            book.cancel(action.order_id, player)
            return []
        return []

    def _settle_trades(self, trades, pstate, log):
        """Apply executed trades to positions/cash/gross_buys immediately."""
        for tr in trades:
            pstate[tr.buyer].position += 1
            pstate[tr.buyer].cash -= tr.price
            pstate[tr.buyer].gross_buys += 1
            pstate[tr.seller].position -= 1
            pstate[tr.seller].cash += tr.price
            log.trades.append(tr)
            if log.first_taker is None:
                log.first_taker = tr.aggressor

    def _settle(self, log, pstate, value, k):
        cfg = self.config
        for p in (0, 1, 2):
            ps = pstate[p]
            pnl = ps.position * value + ps.cash
            if p == Player.B and cfg.buyer_obligation:
                shortfall = max(0, k - ps.gross_buys)
                pnl -= cfg.penalty_per_unit * shortfall
            log.positions[p] = ps.position
            log.cash[p] = ps.cash
            log.gross_buys[p] = ps.gross_buys
            log.pnl[p] = pnl
