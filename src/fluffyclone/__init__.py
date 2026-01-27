"""
Handling the weighted uniform distribution over spanning trees.

Weighted uniform distribution
-----------------------------

Reference: R. Lyons and Y. Peres (2021),
Probability on trees and networks, Cambridge University Press, New York.

Random trees
------------

The sampling procedure is based on Wilson's method,
which consists in using loop-erased random walks from a Markov chain.

See https://github.com/ulysseherbach/fluffyclone for documentation.
"""
from importlib.metadata import version as _version
from fluffyclone.sampling import random_tree
from fluffyclone.utils import distribution_exact, distribution_estim

__all__ = ['distribution_estim', 'distribution_exact', 'random_tree']

try:
    __version__ = _version('fluffyclone')
except Exception:
    __version__ = 'unknown version'
