"""
RCOA Agent Definitions

Rice agents (stationary) and Crab agents (mobile) with their state variables.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


class CrabState(Enum):
    """Behavioral states for Crab agents."""
    FORAGING = "foraging"
    SYMBIOSIS = "symbiosis"
    MOLTING = "molting"


@dataclass
class RiceAgent:
    """
    Stationary Rice agent representing a candidate solution.
    
    Attributes:
        position: Solution vector in R^D
        health: Structural integrity [0, 1], decays without crab attention
        fitness: Raw fitness value f(position)
        yield_value: Effective fitness = fitness * health
        feature_mask: Binary mask for active dimensions (for weeding)
        crab_density: Number of crabs currently servicing this agent
        stagnation: Counter for iterations without improvement
        locked: Whether protected by molting (cannot be modified)
        lock_timer: Iterations remaining in locked state
    """
    position: np.ndarray
    health: float = 1.0
    fitness: float = 0.0
    yield_value: float = 0.0
    feature_mask: Optional[np.ndarray] = None
    crab_density: int = 0
    stagnation: int = 0
    locked: bool = False
    lock_timer: int = 0
    
    def __post_init__(self):
        if self.feature_mask is None:
            self.feature_mask = np.ones(len(self.position), dtype=np.float64)
    
    def update_yield(self):
        """Update yield value based on current fitness and health."""
        self.yield_value = self.fitness * self.health
    
    def degrade(self, lambda_deg: float):
        """Apply degradation to health if not locked."""
        if not self.locked:
            self.health *= np.exp(-lambda_deg)
            self.update_yield()
    
    def restore_health(self, amount: float):
        """Restore health by a given amount."""
        self.health = min(1.0, self.health + amount)
        self.update_yield()


@dataclass
class CrabAgent:
    """
    Mobile Crab agent that services Rice agents.
    
    Attributes:
        position: Current position in search space
        velocity: Velocity vector for PSO-like movement
        energy: Energy level [0, 1]
        state: Current behavioral state
        personal_best: Best position found by this crab
        personal_best_fitness: Fitness at personal best
        target_rice_idx: Index of Rice agent being targeted
        recent_improvement: Sum of improvements made (for molting selection)
        molt_timer: Iterations remaining in molting state
    """
    position: np.ndarray
    velocity: Optional[np.ndarray] = None
    energy: float = 1.0
    state: CrabState = CrabState.FORAGING
    personal_best: Optional[np.ndarray] = None
    personal_best_fitness: float = float('-inf')
    target_rice_idx: int = 0
    recent_improvement: float = 0.0
    molt_timer: int = 0
    
    def __post_init__(self):
        if self.velocity is None:
            self.velocity = np.zeros_like(self.position)
        if self.personal_best is None:
            self.personal_best = self.position.copy()
    
    def update_personal_best(self, position: np.ndarray, fitness: float):
        """Update personal best if new fitness is better."""
        if fitness > self.personal_best_fitness:
            self.personal_best = position.copy()
            self.personal_best_fitness = fitness
            return True
        return False
    
    def start_molting(self, duration: int = 5):
        """Enter molting state."""
        self.state = CrabState.MOLTING
        self.molt_timer = duration
    
    def update_molt(self) -> bool:
        """Update molting timer. Returns True if molting ended."""
        if self.state == CrabState.MOLTING:
            self.molt_timer -= 1
            if self.molt_timer <= 0:
                self.state = CrabState.FORAGING
                self.energy = 1.0
                self.recent_improvement = 0.0
                return True
        return False
