"""
RCOA Core Algorithm

Rice-Crab Optimization Algorithm main class implementing the complete
optimization loop with all four operators.
"""

import numpy as np
from typing import Callable, Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field

from rcoa.agents import RiceAgent, CrabAgent, CrabState
from rcoa.operators import (
    weeding, fertilization, bioturbation, molting,
    update_locks, update_molts
)


@dataclass
class RCOAConfig:
    """
    Configuration parameters for RCOA.
    
    Default values are based on parameter sensitivity analysis
    from the RCOA paper (Section 6).
    """
    # Population sizes
    n_rice: int = 30
    n_crabs: int = 20
    
    # Core parameters
    gamma: float = 0.5       # Cannibalism coefficient (density penalty)
    eta: float = 0.25        # Fertilization rate
    alpha: float = 0.2       # Bioturbation scale
    epsilon: float = 0.15    # Weeding sensitivity threshold
    
    # Timing parameters
    tau_stag: int = 6        # Stagnation threshold for bioturbation
    tau_molt: int = 15       # Molting period
    
    # Movement parameters (PSO-inherited)
    omega: float = 0.65      # Inertia weight
    c1: float = 1.5          # Cognitive weight
    c2: float = 2.0          # Social weight
    v_max_ratio: float = 0.2 # Max velocity as ratio of search range
    
    # Interaction
    r_int_ratio: float = 0.1 # Interaction radius as ratio of search range
    
    # Degradation
    lambda_deg: float = 0.01 # Degradation rate
    
    # Operator toggles
    enable_weeding: bool = True
    enable_fertilization: bool = True
    enable_bioturbation: bool = True
    enable_molting: bool = True
    
    # Weeding optimization
    stochastic_weeding_dims: Optional[int] = None  # If set, only test this many dims
    
    # Instant mode (skip travel time)
    instant_mode: bool = True


@dataclass
class RCOAResult:
    """Results from RCOA optimization run."""
    best_position: np.ndarray
    best_fitness: float
    convergence_history: List[float]
    function_evaluations: int
    iterations: int
    operator_stats: Dict[str, int]


class RCOA:
    """
    Rice-Crab Optimization Algorithm.
    
    A heterogeneous dual-population metaheuristic for regenerative
    optimization problems.
    
    Example:
        >>> from rcoa import RCOA, RCOAConfig
        >>> import numpy as np
        >>> 
        >>> def sphere(x):
        ...     return -np.sum(x**2)  # Negative because RCOA maximizes
        >>> 
        >>> config = RCOAConfig(n_rice=20, n_crabs=10)
        >>> optimizer = RCOA(config)
        >>> result = optimizer.optimize(
        ...     objective_fn=sphere,
        ...     bounds=(np.full(10, -5.0), np.full(10, 5.0)),
        ...     max_iterations=100
        ... )
        >>> print(f"Best fitness: {result.best_fitness}")
    """
    
    def __init__(self, config: Optional[RCOAConfig] = None):
        """
        Initialize RCOA optimizer.
        
        Args:
            config: Configuration parameters. Uses defaults if None.
        """
        self.config = config or RCOAConfig()
        self.rice_population: List[RiceAgent] = []
        self.crab_population: List[CrabAgent] = []
        self.best_position: Optional[np.ndarray] = None
        self.best_fitness: float = float('-inf')
        self.convergence_history: List[float] = []
        self.function_evaluations: int = 0
        self.iteration: int = 0
        
        # Operator statistics
        self.stats = {
            'weeding_tests': 0,
            'dimensions_pruned': 0,
            'fertilizations': 0,
            'bioturbations': 0,
            'molts': 0,
        }
    
    def _evaluate(self, position: np.ndarray, objective_fn: Callable) -> float:
        """Evaluate fitness and track function evaluations."""
        self.function_evaluations += 1
        return objective_fn(position)
    
    def _initialize_populations(
        self,
        bounds: Tuple[np.ndarray, np.ndarray],
        objective_fn: Callable,
    ) -> None:
        """
        Initialize Rice and Crab populations using Latin Hypercube Sampling.
        """
        lb, ub = bounds
        D = len(lb)
        cfg = self.config
        
        # Latin Hypercube Sampling for Rice agents
        self.rice_population = []
        for i in range(cfg.n_rice):
            # LHS: divide each dimension into n_rice intervals
            position = np.zeros(D)
            for d in range(D):
                interval = (ub[d] - lb[d]) / cfg.n_rice
                position[d] = lb[d] + (i + np.random.random()) * interval
            
            # Shuffle to break correlation
            np.random.shuffle(position)
            position = np.clip(position, lb, ub)
            
            rice = RiceAgent(position=position)
            rice.fitness = self._evaluate(position, objective_fn)
            rice.update_yield()
            self.rice_population.append(rice)
            
            # Track global best
            if rice.yield_value > self.best_fitness:
                self.best_fitness = rice.yield_value
                self.best_position = rice.position.copy()
        
        # Initialize Crab agents near Rice agents
        self.crab_population = []
        for j in range(cfg.n_crabs):
            # Seed near a Rice agent
            rice_idx = j % cfg.n_rice
            base_position = self.rice_population[rice_idx].position
            noise = np.random.normal(0, 0.1, D)
            position = np.clip(base_position + noise, lb, ub)
            
            crab = CrabAgent(position=position)
            crab.personal_best = position.copy()
            crab.personal_best_fitness = self.rice_population[rice_idx].yield_value
            self.crab_population.append(crab)
    
    def _compute_attractiveness(self, rice: RiceAgent) -> float:
        """
        Compute density-penalized attractiveness for target selection.
        
        Psi = (1 - yield + 0.5) / (1 + gamma * density)
        
        Higher urgency (lower yield) = more attractive
        Higher density = less attractive (density penalty)
        """
        cfg = self.config
        urgency = (1.0 - rice.yield_value) + 0.5
        return urgency / (1.0 + cfg.gamma * rice.crab_density)
    
    def _select_target(self, crab: CrabAgent) -> int:
        """Select target Rice agent for a Crab using density-penalized attractiveness."""
        best_attr = float('-inf')
        best_idx = 0
        
        for i, rice in enumerate(self.rice_population):
            attr = self._compute_attractiveness(rice)
            if attr > best_attr:
                best_attr = attr
                best_idx = i
        
        return best_idx
    
    def _move_crab(
        self,
        crab: CrabAgent,
        target_rice: RiceAgent,
        bounds: Tuple[np.ndarray, np.ndarray],
    ) -> float:
        """
        Move crab toward target rice using PSO-like velocity update.
        
        Returns distance to target after move.
        """
        cfg = self.config
        lb, ub = bounds
        
        # Calculate max velocity
        v_max = cfg.v_max_ratio * (ub - lb)
        
        # PSO velocity update
        r1, r2 = np.random.random(2)
        
        cognitive = cfg.c1 * r1 * (crab.personal_best - crab.position)
        social = cfg.c2 * r2 * (target_rice.position - crab.position)
        
        crab.velocity = cfg.omega * crab.velocity + cognitive + social
        crab.velocity = np.clip(crab.velocity, -v_max, v_max)
        
        # Update position
        crab.position = crab.position + crab.velocity
        crab.position = np.clip(crab.position, lb, ub)
        
        # Return distance to target
        return np.linalg.norm(crab.position - target_rice.position)
    
    def _step(
        self,
        objective_fn: Callable,
        bounds: Tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Execute one iteration of RCOA."""
        cfg = self.config
        lb, ub = bounds
        search_range = np.linalg.norm(ub - lb)
        r_int = cfg.r_int_ratio * search_range
        
        # Phase 1: Degradation (unattended Rice loses health)
        for rice in self.rice_population:
            if rice.crab_density == 0 and not rice.locked:
                rice.degrade(cfg.lambda_deg)
        
        # Phase 2: Reset density counts
        for rice in self.rice_population:
            rice.crab_density = 0
        
        # Phase 3: Crab movement and operator application
        for crab in self.crab_population:
            # Skip molting crabs
            if crab.state == CrabState.MOLTING:
                continue
            
            # Select target
            target_idx = self._select_target(crab)
            crab.target_rice_idx = target_idx
            target_rice = self.rice_population[target_idx]
            
            # Move or teleport to target
            if cfg.instant_mode:
                # Instant mode: teleport near target
                noise = np.random.normal(0, 0.01, len(target_rice.position))
                crab.position = np.clip(target_rice.position + noise, lb, ub)
                distance = 0.0
            else:
                # Normal mode: PSO-like movement
                distance = self._move_crab(crab, target_rice, bounds)
            
            # Check if in symbiosis range
            if distance < r_int or cfg.instant_mode:
                crab.state = CrabState.SYMBIOSIS
                target_rice.crab_density += 1
                
                # Store previous yield for stagnation detection
                prev_yield = target_rice.yield_value
                
                # Operator A: Weeding
                if cfg.enable_weeding and not target_rice.locked:
                    stoch_dims = cfg.stochastic_weeding_dims
                    if stoch_dims is None:
                        stoch_dims = max(1, len(target_rice.position) // 3)
                    
                    tested, pruned = weeding(
                        target_rice, 
                        lambda x: self._evaluate(x, objective_fn),
                        cfg.epsilon,
                        stoch_dims
                    )
                    self.stats['weeding_tests'] += tested
                    self.stats['dimensions_pruned'] += pruned
                
                # Operator B: Fertilization
                if cfg.enable_fertilization and not target_rice.locked:
                    fertilization(target_rice, crab, cfg.eta, cfg.gamma)
                    target_rice.fitness = self._evaluate(target_rice.position, objective_fn)
                    target_rice.update_yield()
                    self.stats['fertilizations'] += 1
                    
                    # Update crab's personal best
                    crab.update_personal_best(target_rice.position, target_rice.yield_value)
                
                # Operator C: Bioturbation (stagnation escape)
                if cfg.enable_bioturbation:
                    if abs(target_rice.yield_value - prev_yield) < 0.001:
                        target_rice.stagnation += 1
                    else:
                        target_rice.stagnation = 0
                    
                    if target_rice.stagnation >= cfg.tau_stag and not target_rice.locked:
                        bioturbation(target_rice, cfg.alpha, bounds)
                        target_rice.fitness = self._evaluate(target_rice.position, objective_fn)
                        target_rice.update_yield()
                        self.stats['bioturbations'] += 1
            else:
                crab.state = CrabState.FORAGING
        
        # Phase 4: Molting
        if cfg.enable_molting and self.iteration % cfg.tau_molt == 0 and self.iteration > 0:
            crabs_molted, rice_locked = molting(
                self.rice_population, self.crab_population
            )
            self.stats['molts'] += crabs_molted
        
        # Update lock and molt timers
        update_locks(self.rice_population)
        update_molts(self.crab_population)
        
        # Phase 5: Update global best
        for rice in self.rice_population:
            if rice.yield_value > self.best_fitness:
                self.best_fitness = rice.yield_value
                self.best_position = rice.position.copy()
        
        self.convergence_history.append(self.best_fitness)
        self.iteration += 1
    
    def optimize(
        self,
        objective_fn: Callable[[np.ndarray], float],
        bounds: Tuple[np.ndarray, np.ndarray],
        max_iterations: Optional[int] = None,
        max_fe: Optional[int] = None,
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> RCOAResult:
        """
        Run RCOA optimization.
        
        Args:
            objective_fn: Objective function f(x) -> float to MAXIMIZE
            bounds: Tuple of (lower_bounds, upper_bounds) as numpy arrays
            max_iterations: Maximum number of iterations (default: 1000)
            max_fe: Maximum function evaluations (overrides max_iterations if set)
            callback: Optional callback(iteration, best_fitness) called each iteration
            
        Returns:
            RCOAResult with best solution, convergence history, and statistics
        """
        lb, ub = bounds
        D = len(lb)
        
        if max_iterations is None and max_fe is None:
            max_iterations = 1000
        
        # Reset state
        self.best_position = None
        self.best_fitness = float('-inf')
        self.convergence_history = []
        self.function_evaluations = 0
        self.iteration = 0
        self.stats = {k: 0 for k in self.stats}
        
        # Initialize populations
        self._initialize_populations(bounds, objective_fn)
        
        # Main optimization loop
        while True:
            # Check termination
            if max_fe is not None and self.function_evaluations >= max_fe:
                break
            if max_iterations is not None and self.iteration >= max_iterations:
                break
            
            self._step(objective_fn, bounds)
            
            if callback is not None:
                callback(self.iteration, self.best_fitness)
        
        return RCOAResult(
            best_position=self.best_position,
            best_fitness=self.best_fitness,
            convergence_history=self.convergence_history,
            function_evaluations=self.function_evaluations,
            iterations=self.iteration,
            operator_stats=self.stats.copy(),
        )


def minimize(
    objective_fn: Callable[[np.ndarray], float],
    bounds: Tuple[np.ndarray, np.ndarray],
    config: Optional[RCOAConfig] = None,
    **kwargs,
) -> RCOAResult:
    """
    Convenience function to minimize an objective using RCOA.
    
    Wraps the objective to convert minimization to maximization.
    
    Args:
        objective_fn: Function to minimize
        bounds: Search bounds
        config: RCOA configuration
        **kwargs: Additional arguments passed to RCOA.optimize()
        
    Returns:
        RCOAResult (note: best_fitness will be negated)
    """
    optimizer = RCOA(config)
    
    # Wrap to negate for maximization
    def negated(x):
        return -objective_fn(x)
    
    result = optimizer.optimize(negated, bounds, **kwargs)
    
    # Convert back to minimization result
    result.best_fitness = -result.best_fitness
    result.convergence_history = [-f for f in result.convergence_history]
    
    return result
