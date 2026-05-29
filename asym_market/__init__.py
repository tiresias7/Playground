"""Three-player asymmetric-information trading game: research framework.

A research framework (not a fixed-strategy simulator) for studying who provides
liquidity, whether informed players leak by posting, and when a silent
equilibrium forms. The solver searches for approximate best responses and
equilibria; it does not encode any behavioural prior.

Public API
----------
* ``GameConfig``                       -- defines a game instance
* ``GameEnv``                          -- runs episodes, full logging
* ``OrderBook``                        -- single-unit price-time priority book
* ``evaluate_policy``                  -- Monte-Carlo profile evaluation
* ``best_response`` / ``iterative_best_response`` -- (approximate) solvers
* ``behavioral_report``                -- Phase-2 behavioural diagnostics
"""

from .config import GameConfig
from .env import EpisodeLog, GameEnv, Observation, legal_actions
from .order_book import OrderBook
from .types import Action, ActionType, Player, Side, State, Trade
from .policies import (
    AlwaysWaitPolicy, Policy, RandomPolicy, TabularPolicy, info_key, make_act_fn,
)
from .evaluate import EvalResult, evaluate_policy
from .best_response import (
    BRResult, IBRResult, best_response, iterative_best_response,
)
from .diagnostics import behavioral_report, print_report
from .logging_utils import write_episode_logs_csv

__all__ = [
    "GameConfig", "GameEnv", "EpisodeLog", "Observation", "legal_actions",
    "OrderBook", "Action", "ActionType", "Player", "Side", "State", "Trade",
    "Policy", "RandomPolicy", "AlwaysWaitPolicy", "TabularPolicy", "info_key",
    "make_act_fn", "EvalResult", "evaluate_policy", "BRResult", "IBRResult",
    "best_response", "iterative_best_response", "behavioral_report",
    "print_report", "write_episode_logs_csv",
]
