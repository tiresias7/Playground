"""Recovering hidden chess rules from observations alone.

A thought experiment: bots emit (position, move) pairs; the position is a
generic 64-symbol vector and the move is a (from, to) index pair. No rules,
geometry, piece identities, or domain knowledge are given to the learner. How
much of the hidden generative process -- the structure and rules of chess --
can be recovered from data?

See ``experiment.py`` for the full pipeline and ``README.md`` for findings.
"""

from .encoding import make_codec
from .datagen import generate

__all__ = ["make_codec", "generate"]
