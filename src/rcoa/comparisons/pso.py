"""
Particle Swarm Optimization (PSO)

Standard PSO with constriction factor for comparison with RCOA.

Reference:
    Kennedy, J., & Eberhart, R. (1995). Particle Swarm Optimization.
    Proceedings of ICNN'95 - International Conference on Neural Networks.
"""

import numpy as np
from typing import Callable, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class PSOResult:
    """Results from PSO optimization run."""
    best_position: np.ndarray
    best_fitness: float
    convergence_history: List[float]
    function_evaluations: int
    iterations: int


class PSO:
    """
    Particle Swarm Optimization with constriction factor.
    
    Uses the constriction factor variant (Clerc & Kennedy, 2002)
    which is equivalent to standard PSO with properly tuned parameters.
    
    Example:
        >>> from rcoa.comparisons import PSO
        >>> import numpy as np
        >>> 
        >>> def sphere(x):
        ...     return -np.sum(x**2)  # Negative for maximization
        >>> 
        >>> pso = PSO(n_particles=30)
        >>> result = pso.optimize(
        ...     objective_fn=sphere,
        ...     bounds=(np.full(10, -5.0), np.full(10, 5.0)),
        ...     max_iterations=100
        ... )
    """
    
    def __init__(
        self,
        n_particles: int = 30,
        omega: float = 0.729,      # Inertia weight (constriction)
        c1: float = 1.49618,       # Cognitive coefficient
        c2: float = 1.49618,       # Social coefficient
        v_max_ratio: float = 0.2,  # Max velocity as ratio of range
    ):
        """
        Initialize PSO.
        
        Args:
            n_particles: Number of particles
            omega: Inertia weight
            c1: Cognitive coefficient (personal best attraction)
            c2: Social coefficient (global best attraction)
            v_max_ratio: Maximum velocity as ratio of search range
        """
        self.n_particles = n_particles
        self.omega = omega
        self.c1 = c1
        self.c2 = c2
        self.v_max_ratio = v_max_ratio
        
        # State
        self.positions: np.ndarray = np.array([])
        self.velocities: np.ndarray = np.array([])
        self.personal_best_pos: np.ndarray = np.array([])
        self.personal_best_fit: np.ndarray = np.array([])
        self.global_best_pos: np.ndarray = np.array([])
        self.global_best_fit: float = float('-inf')
        self.convergence_history: List[float] = []
        self.function_evaluations: int = 0
    
    def _evaluate(self, position: np.ndarray, objective_fn: Callable) -> float:
        """Evaluate fitness and track function evaluations."""
        self.function_evaluations += 1
        return objective_fn(position)
    
    def _initialize(
        self,
        bounds: Tuple[np.ndarray, np.ndarray],
        objective_fn: Callable,
    ) -> None:
        """Initialize particle positions and velocities."""
        lb, ub = bounds
        D = len(lb)
        
        # Random uniform initialization
        self.positions = np.random.uniform(
            lb, ub, (self.n_particles, D)
        )
        
        # Initialize velocities to zero
        self.velocities = np.zeros((self.n_particles, D))
        
        # Evaluate initial fitness
        self.personal_best_pos = self.positions.copy()
        self.personal_best_fit = np.array([
            self._evaluate(p, objective_fn) for p in self.positions
        ])
        
        # Find global best
        best_idx = np.argmax(self.personal_best_fit)
        self.global_best_pos = self.personal_best_pos[best_idx].copy()
        self.global_best_fit = self.personal_best_fit[best_idx]
    
    def _step(
        self,
        objective_fn: Callable,
        bounds: Tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Execute one PSO iteration."""
        lb, ub = bounds
        v_max = self.v_max_ratio * (ub - lb)
        
        for i in range(self.n_particles):
            # Random coefficients
            r1 = np.random.random(len(lb))
            r2 = np.random.random(len(lb))
            
            # Velocity update
            cognitive = self.c1 * r1 * (self.personal_best_pos[i] - self.positions[i])
            social = self.c2 * r2 * (self.global_best_pos - self.positions[i])
            
            self.velocities[i] = (
                self.omega * self.velocities[i] + cognitive + social
            )
            
            # Clamp velocity
            self.velocities[i] = np.clip(self.velocities[i], -v_max, v_max)
            
            # Position update
            self.positions[i] = self.positions[i] + self.velocities[i]
            self.positions[i] = np.clip(self.positions[i], lb, ub)
            
            # Evaluate fitness
            fitness = self._evaluate(self.positions[i], objective_fn)
            
            # Update personal best
            if fitness > self.personal_best_fit[i]:
                self.personal_best_fit[i] = fitness
                self.personal_best_pos[i] = self.positions[i].copy()
                
                # Update global best
                if fitness > self.global_best_fit:
                    self.global_best_fit = fitness
                    self.global_best_pos = self.positions[i].copy()
        
        self.convergence_history.append(self.global_best_fit)
    
    def optimize(
        self,
        objective_fn: Callable[[np.ndarray], float],
        bounds: Tuple[np.ndarray, np.ndarray],
        max_iterations: Optional[int] = None,
        max_fe: Optional[int] = None,
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> PSOResult:
        """
        Run PSO optimization.
        
        Args:
            objective_fn: Objective function f(x) -> float to MAXIMIZE
            bounds: Tuple of (lower_bounds, upper_bounds)
            max_iterations: Maximum iterations
            max_fe: Maximum function evaluations
            callback: Optional callback(iteration, best_fitness)
            
        Returns:
            PSOResult with best solution and statistics
        """
        if max_iterations is None and max_fe is None:
            max_iterations = 1000
        
        # Reset state
        self.convergence_history = []
        self.function_evaluations = 0
        self.global_best_fit = float('-inf')
        
        # Initialize
        self._initialize(bounds, objective_fn)
        
        iteration = 0
        while True:
            # Check termination
            if max_fe is not None and self.function_evaluations >= max_fe:
                break
            if max_iterations is not None and iteration >= max_iterations:
                break
            
            self._step(objective_fn, bounds)
            iteration += 1
            
            if callback is not None:
                callback(iteration, self.global_best_fit)
        
        return PSOResult(
            best_position=self.global_best_pos,
            best_fitness=self.global_best_fit,
            convergence_history=self.convergence_history,
            function_evaluations=self.function_evaluations,
            iterations=iteration,
        )


def minimize(
    objective_fn: Callable[[np.ndarray], float],
    bounds: Tuple[np.ndarray, np.ndarray],
    n_particles: int = 30,
    **kwargs,
) -> PSOResult:
    """
    Convenience function to minimize using PSO.
    
    Wraps objective to convert minimization to maximization.
    """
    pso = PSO(n_particles=n_particles)
    
    def negated(x):
        return -objective_fn(x)
    
    result = pso.optimize(negated, bounds, **kwargs)
    result.best_fitness = -result.best_fitness
    result.convergence_history = [-f for f in result.convergence_history]
    
    return result
