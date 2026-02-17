"""
Genetic Algorithm (GA)

Real-coded GA with tournament selection and blend crossover.

Reference:
    Holland, J.H. (1992). Adaptation in Natural and Artificial Systems.
    MIT Press.
"""

import numpy as np
from typing import Callable, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class GAResult:
    """Results from GA optimization run."""
    best_position: np.ndarray
    best_fitness: float
    convergence_history: List[float]
    function_evaluations: int
    iterations: int


class GeneticAlgorithm:
    """
    Real-coded Genetic Algorithm.
    
    Features:
    - Tournament selection
    - BLX-alpha crossover (blend crossover)
    - Gaussian mutation
    - Elitism
    
    Example:
        >>> from rcoa.comparisons import GeneticAlgorithm
        >>> import numpy as np
        >>> 
        >>> def sphere(x):
        ...     return -np.sum(x**2)  # Negative for maximization
        >>> 
        >>> ga = GeneticAlgorithm(pop_size=30)
        >>> result = ga.optimize(
        ...     objective_fn=sphere,
        ...     bounds=(np.full(10, -5.0), np.full(10, 5.0)),
        ...     max_iterations=100
        ... )
    """
    
    def __init__(
        self,
        pop_size: int = 30,
        tournament_size: int = 3,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
        mutation_sigma: float = 0.1,
        elitism: int = 2,
        blx_alpha: float = 0.5,
    ):
        """
        Initialize GA.
        
        Args:
            pop_size: Population size
            tournament_size: Number of individuals in tournament
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation per gene
            mutation_sigma: Standard deviation for Gaussian mutation
            elitism: Number of best individuals to preserve
            blx_alpha: Alpha parameter for BLX crossover
        """
        self.pop_size = pop_size
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.elitism = elitism
        self.blx_alpha = blx_alpha
        
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
    
    def _tournament_select(self) -> int:
        """Select parent using tournament selection."""
        candidates = np.random.choice(
            self.pop_size, self.tournament_size, replace=False
        )
        winner = candidates[np.argmax(self.fitness[candidates])]
        return winner
    
    def _blx_crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        bounds: Tuple[np.ndarray, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        BLX-alpha (blend) crossover.
        
        Creates offspring in extended range around parents.
        """
        lb, ub = bounds
        
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        D = len(parent1)
        child1 = np.zeros(D)
        child2 = np.zeros(D)
        
        for j in range(D):
            min_val = min(parent1[j], parent2[j])
            max_val = max(parent1[j], parent2[j])
            diff = max_val - min_val
            
            low = min_val - self.blx_alpha * diff
            high = max_val + self.blx_alpha * diff
            
            child1[j] = np.random.uniform(low, high)
            child2[j] = np.random.uniform(low, high)
        
        # Bound handling
        child1 = np.clip(child1, lb, ub)
        child2 = np.clip(child2, lb, ub)
        
        return child1, child2
    
    def _mutate(
        self,
        individual: np.ndarray,
        bounds: Tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """Gaussian mutation."""
        lb, ub = bounds
        mutant = individual.copy()
        
        for j in range(len(individual)):
            if np.random.random() < self.mutation_rate:
                sigma = self.mutation_sigma * (ub[j] - lb[j])
                mutant[j] += np.random.normal(0, sigma)
        
        return np.clip(mutant, lb, ub)
    
    def _step(
        self,
        objective_fn: Callable,
        bounds: Tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Execute one GA generation."""
        # Sort population by fitness (descending)
        sorted_indices = np.argsort(self.fitness)[::-1]
        
        # Elitism: preserve best individuals
        new_population = [self.population[sorted_indices[i]].copy() 
                        for i in range(self.elitism)]
        new_fitness = [self.fitness[sorted_indices[i]] 
                      for i in range(self.elitism)]
        
        # Generate rest of population
        while len(new_population) < self.pop_size:
            # Selection
            parent1_idx = self._tournament_select()
            parent2_idx = self._tournament_select()
            
            # Crossover
            child1, child2 = self._blx_crossover(
                self.population[parent1_idx],
                self.population[parent2_idx],
                bounds
            )
            
            # Mutation
            child1 = self._mutate(child1, bounds)
            child2 = self._mutate(child2, bounds)
            
            # Evaluate
            fit1 = self._evaluate(child1, objective_fn)
            fit2 = self._evaluate(child2, objective_fn)
            
            new_population.append(child1)
            new_fitness.append(fit1)
            
            if len(new_population) < self.pop_size:
                new_population.append(child2)
                new_fitness.append(fit2)
        
        # Update population
        self.population = np.array(new_population[:self.pop_size])
        self.fitness = np.array(new_fitness[:self.pop_size])
        
        # Update best
        best_idx = np.argmax(self.fitness)
        if self.fitness[best_idx] > self.best_fitness:
            self.best_fitness = self.fitness[best_idx]
            self.best_position = self.population[best_idx].copy()
        
        self.convergence_history.append(self.best_fitness)
    
    def optimize(
        self,
        objective_fn: Callable[[np.ndarray], float],
        bounds: Tuple[np.ndarray, np.ndarray],
        max_iterations: Optional[int] = None,
        max_fe: Optional[int] = None,
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> GAResult:
        """
        Run GA optimization.
        
        Args:
            objective_fn: Objective function f(x) -> float to MAXIMIZE
            bounds: Tuple of (lower_bounds, upper_bounds)
            max_iterations: Maximum generations
            max_fe: Maximum function evaluations
            callback: Optional callback(iteration, best_fitness)
            
        Returns:
            GAResult with best solution and statistics
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
        
        return GAResult(
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
) -> GAResult:
    """Convenience function to minimize using GA."""
    ga = GeneticAlgorithm(pop_size=pop_size)
    
    def negated(x):
        return -objective_fn(x)
    
    result = ga.optimize(negated, bounds, **kwargs)
    result.best_fitness = -result.best_fitness
    result.convergence_history = [-f for f in result.convergence_history]
    
    return result
