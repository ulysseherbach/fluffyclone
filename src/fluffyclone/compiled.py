"""
Just-in-time compiled version of useful functions.
"""
from numba import njit
from .sampling import random_walk_step as random_walk_step_jit

# Export compiled function to the namespace
random_walk_step_jit = njit(random_walk_step_jit)
