"""
Selective Maintenance Problem (SMP) Benchmark

A 20-component series-parallel system with Weibull degradation.

Reference:
    Cassady, C.R., Murdock, W.P., & Pohl, E.A. (2001). Selective Maintenance
    Modeling for Industrial Systems. Journal of Quality in Maintenance
    Engineering, 7(2), 104-117.
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class Component:
    """A system component with degradation characteristics."""
    id: int
    subsystem: int
    shape: float  # Weibull shape parameter
    scale: float  # Weibull scale parameter
    health: float = 1.0
    repair_level: int = 0  # 0=none, 1=minimal, 2=imperfect, 3=perfect


class SelectiveMaintenanceProblem:
    """
    Selective Maintenance Problem for RCOA evaluation.
    
    Models a 20-component series-parallel system where:
    - Components degrade over time (Weibull distribution)
    - Repair crews can apply different repair levels
    - Budget and crew constraints limit repairs
    - Objective: maximize system reliability
    
    This is a regenerative optimization problem that naturally
    maps to RCOA's architecture.
    
    Example:
        >>> from rcoa.benchmarks import SelectiveMaintenanceProblem
        >>> smp = SelectiveMaintenanceProblem(n_components=20, n_crews=6)
        >>> state = smp.reset()
        >>> for t in range(100):
        ...     actions = smp.sample_action()
        ...     state, reward = smp.step(actions)
        >>> print(f"Final reliability: {smp.get_reliability():.3f}")
    """
    
    # Repair levels: [cost, health_restoration]
    REPAIR_LEVELS = {
        0: (0, 0.0),      # No repair
        1: (1, 0.05),     # Minimal repair
        2: (3, 0.15),     # Imperfect repair
        3: (7, 0.30),     # Perfect repair
    }
    
    def __init__(
        self,
        n_components: int = 20,
        n_subsystems: int = 4,
        n_crews: int = 6,
        budget: int = 40,
        severity: float = 5.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize SMP instance.
        
        Args:
            n_components: Number of system components
            n_subsystems: Number of subsystems (series connection)
            n_crews: Maximum simultaneous repair crews
            budget: Total repair budget per episode
            severity: Degradation severity multiplier
            seed: Random seed for reproducibility
        """
        self.n_components = n_components
        self.n_subsystems = n_subsystems
        self.n_crews = n_crews
        self.budget = budget
        self.severity = severity
        
        if seed is not None:
            np.random.seed(seed)
        
        self.components: List[Component] = []
        self.iteration = 0
        self.budget_used = 0
        
        self._init_components()
    
    def _init_components(self):
        """Initialize component characteristics."""
        self.components = []
        comps_per_subsystem = self.n_components // self.n_subsystems
        
        for i in range(self.n_components):
            subsystem = i // comps_per_subsystem
            self.components.append(Component(
                id=i,
                subsystem=min(subsystem, self.n_subsystems - 1),
                shape=np.random.uniform(1.5, 3.0),
                scale=np.random.uniform(0.02, 0.06),
                health=1.0,
                repair_level=0,
            ))
    
    def reset(self) -> np.ndarray:
        """
        Reset the system to initial state.
        
        Returns:
            State vector (component health values)
        """
        for comp in self.components:
            comp.health = 1.0
            comp.repair_level = 0
        self.iteration = 0
        self.budget_used = 0
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """Get current health state of all components."""
        return np.array([c.health for c in self.components])
    
    def get_reliability(self) -> float:
        """
        Calculate system reliability.
        
        Uses series-parallel model:
        - Subsystems are in series (all must work)
        - Components within subsystem are k-out-of-n redundant
        """
        subsystem_reliabilities = []
        comps_per_subsystem = self.n_components // self.n_subsystems
        
        for s in range(self.n_subsystems):
            subsystem_comps = [
                c for c in self.components 
                if c.subsystem == s
            ]
            
            # k-out-of-n: at least 2 out of 5 must work
            k = max(1, len(subsystem_comps) // 2)
            
            # Simplified: product of component healths
            # (In practice, would use proper k-out-of-n calculation)
            healths = [c.health for c in subsystem_comps]
            subsystem_rel = np.mean(healths)  # Average health
            subsystem_reliabilities.append(subsystem_rel)
        
        # Series connection: product of subsystem reliabilities
        return np.prod(subsystem_reliabilities)
    
    def _weibull_degradation(self, comp: Component) -> float:
        """Calculate Weibull degradation for a component."""
        u = np.random.random()
        if u == 0:
            u = 1e-10
        degradation = comp.scale * (-np.log(1 - u)) ** (1 / comp.shape)
        return degradation * (self.severity / 8)
    
    def step(self, repair_actions: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Execute one time step.
        
        Args:
            repair_actions: Array of repair levels (0-3) for each component
            
        Returns:
            Tuple of (new_state, reward)
        """
        self.iteration += 1
        
        # Apply degradation to all components
        for comp in self.components:
            deg = self._weibull_degradation(comp)
            comp.health = max(0.0, comp.health - deg)
        
        # Apply repairs (respecting crew and budget constraints)
        crews_used = 0
        sorted_actions = sorted(
            enumerate(repair_actions),
            key=lambda x: (1 - self.components[x[0]].health, x[1]),
            reverse=True
        )
        
        for comp_idx, level in sorted_actions:
            if crews_used >= self.n_crews:
                break
            
            level = int(np.clip(level, 0, 3))
            cost, restoration = self.REPAIR_LEVELS[level]
            
            if cost > 0 and self.budget_used + cost <= self.budget:
                comp = self.components[comp_idx]
                comp.health = min(1.0, comp.health + restoration)
                comp.repair_level = level
                self.budget_used += cost
                crews_used += 1
        
        reward = self.get_reliability()
        return self.get_state(), reward
    
    def sample_action(self) -> np.ndarray:
        """Sample a random repair action."""
        return np.random.randint(0, 4, self.n_components)
    
    def get_info(self) -> dict:
        """Get current problem state info."""
        return {
            'iteration': self.iteration,
            'reliability': self.get_reliability(),
            'mean_health': np.mean(self.get_state()),
            'budget_used': self.budget_used,
            'budget_remaining': self.budget - self.budget_used,
        }
    
    def evaluate_plan(self, plan: np.ndarray, iterations: int = 100) -> float:
        """
        Evaluate a maintenance plan over multiple iterations.
        
        Args:
            plan: Repair level for each component (applied each iteration)
            iterations: Number of iterations to simulate
            
        Returns:
            Mean reliability over all iterations
        """
        self.reset()
        reliabilities = []
        
        for _ in range(iterations):
            _, reward = self.step(plan)
            reliabilities.append(reward)
        
        return np.mean(reliabilities)
    
    def as_optimization_problem(self) -> Tuple[callable, Tuple[np.ndarray, np.ndarray]]:
        """
        Convert to standard optimization problem format.
        
        Returns:
            Tuple of (objective_function, bounds)
        """
        def objective(x: np.ndarray) -> float:
            """Objective: maximize mean reliability."""
            plan = np.round(x).astype(int)
            plan = np.clip(plan, 0, 3)
            return self.evaluate_plan(plan, iterations=50)
        
        lb = np.zeros(self.n_components)
        ub = np.full(self.n_components, 3.0)
        
        return objective, (lb, ub)
