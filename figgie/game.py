"""Figgie game: dealing, running the trading period, and settlement.

Defaults follow the standard 4-player game: $50 ante each ($200 pot), 10 cards
dealt to each player. The pot is configurable. At settlement each goal-suit
card pays its holder $10, and the player holding the most goal-suit cards wins
the remaining pot (ties split it).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .cards import ALL_SUITS, Deck, Suit
from .bots.base import Bot, Observation, Quote, Bid, Ask
from .market import Market, Trade

GOAL_CARD_BONUS = 10


@dataclass
class RoundResult:
    deck: Deck
    starting_cash: dict[int, int]
    trading_pnl: dict[int, int]   # cash change from trading only
    payouts: dict[int, int]       # pot winnings (card bonuses + majority share)
    final_cash: dict[int, int]
    net: dict[int, int]           # final_cash - (start - ante); profit for the round
    final_hands: dict[int, dict[Suit, int]]
    dealt_hands: dict[int, dict[Suit, int]]  # hands as originally dealt
    trades: list                             # full Trade log for the round
    num_trades: int


class FiggieGame:
    def __init__(
        self,
        bots: list[Bot],
        pot: int = 200,
        starting_cash: int = 350,
        total_ticks: int = 2000,
        trade_history: int = 12,
        rng: random.Random | None = None,
    ):
        n = len(bots)
        if n not in (4, 5):
            raise ValueError("Figgie is played with 4 or 5 players")
        if 40 % n != 0:
            raise ValueError("Players must divide 40 cards evenly")
        self.bots = bots
        self.n = n
        self.pot = pot
        self.ante = pot // n
        self.starting_cash = starting_cash
        self.total_ticks = total_ticks
        self.trade_history = trade_history
        self.rng = rng or random.Random()

        self.deck = Deck.random(self.rng)
        self.hands: dict[int, dict[Suit, int]] = {}
        self.cash: dict[int, int] = {}

    # --- market hooks -------------------------------------------------
    def _owns(self, player: int, suit: Suit) -> int:
        return self.hands[player][suit]

    def _can_afford(self, player: int, price: int) -> bool:
        return self.cash[player] >= price

    def _settle(self, trade: Trade) -> None:
        self.cash[trade.buyer] -= trade.price
        self.cash[trade.seller] += trade.price
        self.hands[trade.buyer][trade.suit] += 1
        self.hands[trade.seller][trade.suit] -= 1

    # --- setup --------------------------------------------------------
    def _deal(self) -> None:
        cards: list[Suit] = []
        for suit in ALL_SUITS:
            cards += [suit] * self.deck.counts[suit]
        self.rng.shuffle(cards)
        per = 40 // self.n
        for p in range(self.n):
            chunk = cards[p * per : (p + 1) * per]
            self.hands[p] = {s: chunk.count(s) for s in ALL_SUITS}
            self.cash[p] = self.starting_cash - self.ante

    def _observation(self, player: int, market: Market, tick: int) -> Observation:
        quotes = {}
        for s in ALL_SUITS:
            book = market.books[s]
            bb = book.best_bid()
            ba = book.best_ask()
            quotes[s] = Quote(bb.price if bb else None, ba.price if ba else None)
        return Observation(
            player=player,
            num_players=self.n,
            pot=self.pot,
            hand=dict(self.hands[player]),
            cash=self.cash[player],
            market=quotes,
            last_trades=tuple(market.trades[-self.trade_history :]),
            tick=tick,
            total_ticks=self.total_ticks,
        )

    # --- run ----------------------------------------------------------
    def play(self) -> RoundResult:
        self._deal()
        self._dealt_hands = {p: dict(self.hands[p]) for p in range(self.n)}
        start = dict(self.cash)
        market = Market(self._owns, self._can_afford, self._settle)

        for p, bot in enumerate(self.bots):
            bot.reset(self._observation(p, market, 0))

        order = list(range(self.n))
        for tick in range(self.total_ticks):
            self.rng.shuffle(order)
            for p in order:
                action = self.bots[p].act(self._observation(p, market, tick))
                if action is None:
                    continue
                if isinstance(action, Bid):
                    market.submit(p, action.suit, True, action.price)
                elif isinstance(action, Ask):
                    market.submit(p, action.suit, False, action.price)

        return self._settle_round(start, market)

    def _settle_round(self, start: dict[int, int], market: Market) -> RoundResult:
        trading_pnl = {p: self.cash[p] - start[p] for p in range(self.n)}

        goal = self.deck.goal_suit
        payouts = {p: 0 for p in range(self.n)}
        for p in range(self.n):
            payouts[p] += self.hands[p][goal] * GOAL_CARD_BONUS

        remainder = self.pot - self.deck.goal_count * GOAL_CARD_BONUS
        most = max(self.hands[p][goal] for p in range(self.n))
        winners = [p for p in range(self.n) if self.hands[p][goal] == most]
        # Split remainder among majority holders; distribute any odd cents to
        # the earliest winners so the pot is fully and exactly paid out.
        share, extra = divmod(remainder, len(winners))
        for i, p in enumerate(winners):
            payouts[p] += share + (1 if i < extra else 0)

        for p in range(self.n):
            self.cash[p] += payouts[p]

        net = {p: self.cash[p] - self.starting_cash for p in range(self.n)}
        return RoundResult(
            deck=self.deck,
            starting_cash={p: self.starting_cash for p in range(self.n)},
            trading_pnl=trading_pnl,
            payouts=payouts,
            final_cash=dict(self.cash),
            net=net,
            final_hands={p: dict(self.hands[p]) for p in range(self.n)},
            dealt_hands=self._dealt_hands,
            trades=list(market.trades),
            num_trades=len(market.trades),
        )
