"""
RCOA - Rice-Crab Optimization Algorithm

A heterogeneous multi-agent metaheuristic for regenerative optimization problems.

Authors: Steven Ritz, Claude, Gemini
"""

from rcoa.core import RCOA, RCOAConfig, RCOAResult, minimize
from rcoa.agents import RiceAgent, CrabAgent, CrabState
from rcoa.operators import weeding, fertilization, bioturbation, molting

__version__ = "0.1.0"
__all__ = [
    "RCOA",
    "RCOAConfig",
    "RCOAResult",
    "minimize",
    "RiceAgent",
    "CrabAgent",
    "CrabState",
    "weeding",
    "fertilization",
    "bioturbation",
    "molting",
]
