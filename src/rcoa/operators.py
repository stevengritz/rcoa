"""
RCOA Operators

The four biologically-motivated operators:
- Weeding: Embedded dimensionality reduction via sensitivity analysis
- Fertilization: Density-penalized gradient injection
- Bioturbation: Lévy-flight perturbation under stagnation
- Molting: Periodic elitism archiving
"""

import numpy as np
from typing import Callable, Tuple, Optional
from rcoa.agents import RiceAgent, CrabAgent, CrabState


def levy_step(beta: float = 1.5) -> float:
    """
    Generate a Lévy flight step using Mantegna's algorithm.
    
    Args:
        beta: Lévy exponent (1 < beta <= 2), default 1.5
        
    Returns:
        Single Lévy step value
        
    Reference:
        Mantegna, R.N. (1994). Fast, accurate algorithm for numerical 
        simulation of Lévy stable stochastic processes. Physical Review E.
    """
    from scipy.special import gamma
    
    sigma_u = (
        gamma(1 + beta) * np.sin(np.pi * beta / 2) /
        (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    
    u = np.random.normal(0, sigma_u)
    v = np.abs(np.random.normal(0, 1))
    
    return u / (v ** (1 / beta))


def weeding(
    rice: RiceAgent,
    objective_fn: Callable[[np.ndarray], float],
    epsilon: float = 0.15,
    stochastic_dims: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Weeding operator: Embedded online dimensionality reduction.
    
    Performs per-dimension sensitivity analysis and prunes dimensions
    with sensitivity below threshold epsilon.
    
    Args:
        rice: Rice agent to apply weeding to
        objective_fn: Fitness function f(x) -> float
        epsilon: Sensitivity threshold for pruning
        stochastic_dims: If set, only test this many random dimensions
                        (reduces computational cost from O(D) to O(stochastic_dims))
    
    Returns:
        Tuple of (dimensions_tested, dimensions_pruned, function_evaluations)
        
    Computational cost: O(dims_tested) function evaluations
    """
    if rice.locked:
        return 0, 0
    
    D = len(rice.position)
    active_dims = np.where(rice.feature_mask > 0.5)[0]
    
    if len(active_dims) == 0:
        return 0, 0
    
    # Determine which dimensions to test
    if stochastic_dims is not None and stochastic_dims < len(active_dims):
        dims_to_test = np.random.choice(active_dims, size=stochastic_dims, replace=False)
    else:
        dims_to_test = active_dims
    
    # Current fitness
    current_fitness = rice.fitness
    pruned_count = 0
    fe_count = 0
    
    for d in dims_to_test:
        # Create trial position with dimension d zeroed
        trial_position = rice.position.copy()
        trial_position[d] = 0.0
        
        # Evaluate sensitivity
        trial_fitness = objective_fn(trial_position)
        fe_count += 1
        
        sensitivity = current_fitness - trial_fitness
        
        # Prune if sensitivity is below threshold
        if sensitivity < epsilon:
            rice.feature_mask[d] = 0.0
            pruned_count += 1
    
    return len(dims_to_test), pruned_count


def fertilization(
    rice: RiceAgent,
    crab: CrabAgent,
    eta: float = 0.25,
    gamma: float = 0.5,
    health_restore_rate: float = 0.1,
) -> float:
    """
    Fertilization operator: Density-penalized gradient injection.
    
    Improves rice position by injecting information from crab's personal best,
    with density penalty to prevent crowding.
    
    Args:
        rice: Rice agent to fertilize
        crab: Crab agent providing gradient information
        eta: Fertilization rate (learning rate)
        gamma: Cannibalism coefficient for density penalty
        health_restore_rate: Rate of health restoration
        
    Returns:
        Amount of position change (L2 norm)
    """
    if rice.locked:
        return 0.0
    
    # Density penalty factor
    density_factor = 1.0 / (1.0 + gamma * rice.crab_density)
    
    # Update position using gradient from crab's personal best
    old_position = rice.position.copy()
    
    for d in range(len(rice.position)):
        if rice.feature_mask[d] > 0.5:  # Only update active dimensions
            delta = eta * density_factor * (crab.personal_best[d] - rice.position[d])
            rice.position[d] += delta
    
    # Restore health
    rice.restore_health(eta * health_restore_rate)
    
    # Calculate improvement for crab tracking
    improvement = np.linalg.norm(rice.position - old_position)
    crab.recent_improvement += improvement
    
    return improvement


def bioturbation(
    rice: RiceAgent,
    alpha: float = 0.2,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> bool:
    """
    Bioturbation operator: Lévy-flight perturbation for stagnation escape.
    
    Applied when rice agent's fitness has stagnated for tau_stag iterations.
    Perturbs active dimensions using Lévy flights.
    
    Args:
        rice: Rice agent to perturb
        alpha: Perturbation scale factor
        bounds: Optional (lower_bounds, upper_bounds) for clamping
        
    Returns:
        True if perturbation was applied
    """
    if rice.locked:
        return False
    
    for d in range(len(rice.position)):
        if rice.feature_mask[d] > 0.5:  # Only perturb active dimensions
            step = levy_step(1.5)
            rice.position[d] += alpha * step
    
    # Clamp to bounds if provided
    if bounds is not None:
        lb, ub = bounds
        rice.position = np.clip(rice.position, lb, ub)
    
    # Reset stagnation counter
    rice.stagnation = 0
    
    return True


def molting(
    rice_population: list[RiceAgent],
    crab_population: list[CrabAgent],
    molt_fraction: float = 0.1,
    lock_duration: int = 5,
) -> Tuple[int, int]:
    """
    Molting operator: Periodic elitism archiving.
    
    Low-performing crabs enter molting state near elite rice agents,
    which are temporarily locked from modification.
    
    Args:
        rice_population: List of all Rice agents
        crab_population: List of all Crab agents
        molt_fraction: Fraction of crabs to molt (default 10%)
        lock_duration: Iterations to keep crabs molting and rice locked
        
    Returns:
        Tuple of (crabs_molted, rice_locked)
    """
    # Sort crabs by recent improvement (ascending - worst performers first)
    active_crabs = [c for c in crab_population if c.state != CrabState.MOLTING]
    if not active_crabs:
        return 0, 0
    
    sorted_crabs = sorted(active_crabs, key=lambda c: c.recent_improvement)
    molt_count = max(1, int(len(sorted_crabs) * molt_fraction))
    
    # Sort rice by yield (descending - best performers first)
    sorted_rice = sorted(rice_population, key=lambda r: r.yield_value, reverse=True)
    
    crabs_molted = 0
    rice_locked = 0
    
    for i in range(min(molt_count, len(sorted_crabs), len(sorted_rice))):
        crab = sorted_crabs[i]
        elite_rice = sorted_rice[i]
        
        # Crab enters molting near elite rice
        crab.start_molting(lock_duration)
        crab.position = elite_rice.position.copy() + np.random.normal(0, 0.01, len(elite_rice.position))
        crabs_molted += 1
        
        # Elite rice becomes locked
        if not elite_rice.locked:
            elite_rice.locked = True
            elite_rice.lock_timer = lock_duration
            rice_locked += 1
    
    return crabs_molted, rice_locked


def update_locks(rice_population: list[RiceAgent]) -> int:
    """
    Update lock timers for all rice agents.
    
    Args:
        rice_population: List of all Rice agents
        
    Returns:
        Number of rice agents that became unlocked
    """
    unlocked_count = 0
    for rice in rice_population:
        if rice.locked:
            rice.lock_timer -= 1
            if rice.lock_timer <= 0:
                rice.locked = False
                unlocked_count += 1
    return unlocked_count


def update_molts(crab_population: list[CrabAgent]) -> int:
    """
    Update molting timers for all crab agents.
    
    Args:
        crab_population: List of all Crab agents
        
    Returns:
        Number of crabs that finished molting
    """
    finished_count = 0
    for crab in crab_population:
        if crab.update_molt():
            finished_count += 1
    return finished_count
