"""
Differential Evolution (DE)

DE/rand/1/bin variant for comparison with RCOA.

Reference:
    Storn, R., & Price, K. (1997). Differential Evolution – A Simple and
    Efficient Heuristic for Global Optimization over Continuous Spaces.
    Journal of Global Optimization, 11(4), 341-359.
"""

import numpy as np
from typing import Callable, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class DEResult:
    """Results from DE optimization run."""
    best_position: np.ndarray
    best_fitness: float
    convergence_history: List[float]
    function_evaluations: int
    iterations: int


class DifferentialEvolution:
    """
    Differential Evolution (DE/rand/1/bin).
    
    Classic DE variant with:
    - rand/1: Random base vector, one difference vector
    - bin: Binomial crossover
    
    Example:
        >>> from rcoa.comparisons import DifferentialEvolution
        >>> import numpy as np
        >>> 
        >>> def sphere(x):
        ...     return -np.sum(x**2)  # Negative for maximization
        >>> 
        >>> de = DifferentialEvolution(pop_size=30)
        >>> result = de.optimize(
        ...     objective_fn=sphere,
        ...     bounds=(np.full(10, -5.0), np.full(10, 5.0)),
        ...     max_iterations=100
        ... )
    """
    
    def __init__(
        self,
        pop_size: int = 30,
        F: float = 0.5,   # Mutation factor
        CR: float = 0.9,  # Crossover rate
    ):
        """
        Initialize DE.
        
        Args:
            pop_size: Population size
            F: Mutation factor (scaling for difference vectors)
            CR: Crossover rate (probability of taking mutant gene)
        """
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        
        # State
        self.population: np.ndarray = np.array([])
        self.fitness: np.ndarray = np.array([])
        self.best_position: np.ndarray = np.array([])
        self.best_fitness: float = float('-inf')
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
        """Initialize population."""
        lb, ub = bounds
        D = len(lb)
        
        # Random uniform initialization
        self.population = np.random.uniform(
            lb, ub, (self.pop_size, D)
        )
        
        # Evaluate initial fitness
        self.fitness = np.array([
            self._evaluate(p, objective_fn) for p in self.population
        ])
        
        # Find best
        best_idx = np.argmax(self.fitness)
        self.best_position = self.population[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]
    
    def _mutate(self, target_idx: int, bounds: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Generate mutant vector using DE/rand/1.
        
        v = x_r1 + F * (x_r2 - x_r3)
        """
        lb, ub = bounds
        
        # Select three distinct random indices (different from target)
        candidates = list(range(self.pop_size))
        candidates.remove(target_idx)
        r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
        
        # Mutation
        mutant = (
            self.population[r1] +
            self.F * (self.population[r2] - self.population[r3])
        )
        
        # Bound handling (bounce back)
        mutant = np.clip(mutant, lb, ub)
        
        return mutant
    
    def _crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """
        Binomial crossover.
        
        Each dimension taken from mutant with probability CR.
        At least one dimension always taken from mutant.
        """
        D = len(target)
        trial = target.copy()
        
        # Ensure at least one dimension from mutant
        j_rand = np.random.randint(D)
        
        for j in range(D):
            if np.random.random() < self.CR or j == j_rand:
                trial[j] = mutant[j]
        
        return trial
    
    def _step(
        self,
        objective_fn: Callable,
        bounds: Tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Execute one DE generation."""
        for i in range(self.pop_size):
            # Mutation
            mutant = self._mutate(i, bounds)
            
            # Crossover
            trial = self._crossover(self.population[i], mutant)
            
            # Selection
            trial_fitness = self._evaluate(trial, objective_fn)
            
            if trial_fitness > self.fitness[i]:
                self.population[i] = trial
                self.fitness[i] = trial_fitness
                
                # Update best
                if trial_fitness > self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_position = trial.copy()
        
        self.convergence_history.append(self.best_fitness)
    
    def optimize(
        self,
        objective_fn: Callable[[np.ndarray], float],
        bounds: Tuple[np.ndarray, np.ndarray],
        max_iterations: Optional[int] = None,
        max_fe: Optional[int] = None,
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> DEResult:
        """
        Run DE optimization.
        
        Args:
            objective_fn: Objective function f(x) -> float to MAXIMIZE
            bounds: Tuple of (lower_bounds, upper_bounds)
            max_iterations: Maximum generations
            max_fe: Maximum function evaluations
            callback: Optional callback(iteration, best_fitness)
            
        Returns:
            DEResult with best solution and statistics
        """
        if max_iterations is None and max_fe is None:
            max_iterations = 1000
        
        # Reset state
        self.convergence_history = []
        self.function_evaluations = 0
        self.best_fitness = float('-inf')
        
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
                callback(iteration, self.best_fitness)
        
        return DEResult(
            best_position=self.best_position,
            best_fitness=self.best_fitness,
            convergence_history=self.convergence_history,
            function_evaluations=self.function_evaluations,
            iterations=iteration,
        )


def minimize(
    objective_fn: Callable[[np.ndarray], float],
    bounds: Tuple[np.ndarray, np.ndarray],
    pop_size: int = 30,
    **kwargs,
) -> DEResult:
    """Convenience function to minimize using DE."""
    de = DifferentialEvolution(pop_size=pop_size)
    
    def negated(x):
        return -objective_fn(x)
    
    result = de.optimize(negated, bounds, **kwargs)
    result.best_fitness = -result.best_fitness
    result.convergence_history = [-f for f in result.convergence_history]
    
    return result
