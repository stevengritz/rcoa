"""
Parameter Sensitivity Analysis

Tools for analyzing RCOA parameter sensitivity and recommending defaults.
"""

import numpy as np
from typing import Callable, Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from rcoa.core import RCOA, RCOAConfig


@dataclass
class SensitivityResult:
    """Result from parameter sensitivity analysis."""
    parameter: str
    values_tested: List[float]
    mean_fitness: List[float]
    std_fitness: List[float]
    best_value: float
    sensitivity_score: float  # Higher = more sensitive


# Recommended default values based on paper Section 6
RECOMMENDED_DEFAULTS = {
    'n_rice': {'value': 30, 'range': (10, 100), 'sensitivity': 'medium'},
    'n_crabs': {'value': 20, 'range': (5, 50), 'sensitivity': 'medium'},
    'gamma': {'value': 0.5, 'range': (0.0, 2.0), 'sensitivity': 'high'},
    'eta': {'value': 0.25, 'range': (0.01, 1.0), 'sensitivity': 'high'},
    'alpha': {'value': 0.2, 'range': (0.01, 1.0), 'sensitivity': 'medium'},
    'epsilon': {'value': 0.15, 'range': (0.01, 0.5), 'sensitivity': 'high'},
    'tau_stag': {'value': 6, 'range': (2, 20), 'sensitivity': 'low'},
    'tau_molt': {'value': 15, 'range': (5, 50), 'sensitivity': 'low'},
    'r_int_ratio': {'value': 0.1, 'range': (0.05, 0.2), 'sensitivity': 'high'},
    'lambda_deg': {'value': 0.01, 'range': (0.0, 0.1), 'sensitivity': 'problem-dependent'},
    'omega': {'value': 0.65, 'range': (0.4, 0.9), 'sensitivity': 'medium'},
    'c1': {'value': 1.5, 'range': (1.0, 2.5), 'sensitivity': 'low'},
    'c2': {'value': 2.0, 'range': (1.5, 3.0), 'sensitivity': 'low'},
}


def recommended_defaults() -> RCOAConfig:
    """
    Get recommended default RCOA configuration.
    
    Based on parameter sensitivity analysis from RCOA paper Section 6.
    These defaults are robust across a wide range of problems.
    
    Returns:
        RCOAConfig with recommended defaults
    """
    return RCOAConfig(
        n_rice=30,
        n_crabs=20,
        gamma=0.5,
        eta=0.25,
        alpha=0.2,
        epsilon=0.15,
        tau_stag=6,
        tau_molt=15,
        r_int_ratio=0.1,
        lambda_deg=0.01,
        omega=0.65,
        c1=1.5,
        c2=2.0,
    )


def parameter_sensitivity_analysis(
    objective_fn: Callable[[np.ndarray], float],
    bounds: Tuple[np.ndarray, np.ndarray],
    parameters: Optional[List[str]] = None,
    n_values: int = 5,
    n_runs: int = 11,
    max_iterations: int = 100,
    verbose: bool = True,
) -> Dict[str, SensitivityResult]:
    """
    Analyze sensitivity of RCOA parameters.
    
    For each parameter, tests multiple values and measures
    the impact on optimization performance.
    
    Args:
        objective_fn: Objective function (to maximize)
        bounds: Search bounds
        parameters: List of parameters to analyze (None = critical ones)
        n_values: Number of values to test per parameter
        n_runs: Number of runs per configuration
        max_iterations: Max iterations per run
        verbose: Print progress
        
    Returns:
        Dictionary mapping parameter names to SensitivityResult
    """
    if parameters is None:
        # Test critical parameters by default
        parameters = ['gamma', 'eta', 'epsilon', 'alpha']
    
    results = {}
    
    for param_name in parameters:
        if param_name not in RECOMMENDED_DEFAULTS:
            continue
        
        param_info = RECOMMENDED_DEFAULTS[param_name]
        low, high = param_info['range']
        
        # Generate test values
        if isinstance(low, int) and isinstance(high, int):
            test_values = np.linspace(low, high, n_values).astype(int).tolist()
        else:
            test_values = np.linspace(low, high, n_values).tolist()
        
        if verbose:
            print(f"\nAnalyzing {param_name}: {test_values}")
        
        mean_fitness = []
        std_fitness = []
        
        for value in test_values:
            run_results = []
            
            for run in range(n_runs):
                # Create config with modified parameter
                config = recommended_defaults()
                setattr(config, param_name, value)
                
                # Run optimization
                np.random.seed(42 + run)
                optimizer = RCOA(config)
                result = optimizer.optimize(
                    objective_fn=objective_fn,
                    bounds=bounds,
                    max_iterations=max_iterations,
                )
                run_results.append(result.best_fitness)
            
            mean_fitness.append(np.mean(run_results))
            std_fitness.append(np.std(run_results))
            
            if verbose:
                print(f"  {param_name}={value:.3f}: {mean_fitness[-1]:.4e} ± {std_fitness[-1]:.4e}")
        
        # Find best value
        best_idx = np.argmax(mean_fitness)
        best_value = test_values[best_idx]
        
        # Compute sensitivity score (coefficient of variation of means)
        sensitivity_score = np.std(mean_fitness) / (np.abs(np.mean(mean_fitness)) + 1e-10)
        
        results[param_name] = SensitivityResult(
            parameter=param_name,
            values_tested=test_values,
            mean_fitness=mean_fitness,
            std_fitness=std_fitness,
            best_value=best_value,
            sensitivity_score=sensitivity_score,
        )
    
    return results


def grid_search(
    objective_fn: Callable[[np.ndarray], float],
    bounds: Tuple[np.ndarray, np.ndarray],
    param_grid: Dict[str, List[Any]],
    n_runs: int = 5,
    max_iterations: int = 100,
    verbose: bool = True,
) -> Tuple[RCOAConfig, float]:
    """
    Grid search for optimal RCOA parameters.
    
    Args:
        objective_fn: Objective function (to maximize)
        bounds: Search bounds
        param_grid: Dictionary of {param_name: [values_to_try]}
        n_runs: Number of runs per configuration
        max_iterations: Max iterations per run
        verbose: Print progress
        
    Returns:
        Tuple of (best_config, best_mean_fitness)
    """
    import itertools
    
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))
    
    if verbose:
        print(f"Grid search: {len(combinations)} configurations")
    
    best_config = None
    best_fitness = float('-inf')
    
    for i, combo in enumerate(combinations):
        # Create config
        config = recommended_defaults()
        for name, value in zip(param_names, combo):
            setattr(config, name, value)
        
        # Run multiple times
        run_results = []
        for run in range(n_runs):
            np.random.seed(42 + run)
            optimizer = RCOA(config)
            result = optimizer.optimize(
                objective_fn=objective_fn,
                bounds=bounds,
                max_iterations=max_iterations,
            )
            run_results.append(result.best_fitness)
        
        mean_fitness = np.mean(run_results)
        
        if verbose:
            param_str = ", ".join(f"{n}={v}" for n, v in zip(param_names, combo))
            print(f"  [{i+1}/{len(combinations)}] {param_str}: {mean_fitness:.4e}")
        
        if mean_fitness > best_fitness:
            best_fitness = mean_fitness
            best_config = config
    
    return best_config, best_fitness


def get_parameter_recommendations(
    problem_type: str = 'general',
) -> RCOAConfig:
    """
    Get parameter recommendations based on problem type.
    
    Args:
        problem_type: One of 'general', 'maintenance', 'feature_selection', 'neural_pruning'
        
    Returns:
        RCOAConfig tuned for the problem type
    """
    config = recommended_defaults()
    
    if problem_type == 'maintenance':
        # For SMP-like problems
        config.lambda_deg = 0.02  # Higher degradation
        config.gamma = 0.5        # Balance density
        config.enable_weeding = False  # No feature selection needed
        
    elif problem_type == 'feature_selection':
        # For noisy feature selection
        config.epsilon = 0.15     # Moderate weeding threshold
        config.enable_weeding = True
        config.stochastic_weeding_dims = None  # Test all dims
        
    elif problem_type == 'neural_pruning':
        # For network pruning
        config.epsilon = 0.10     # Conservative pruning
        config.enable_weeding = True
        config.alpha = 0.1        # Smaller perturbations
        
    elif problem_type == 'multimodal':
        # For highly multimodal functions
        config.gamma = 0.7        # More exploration
        config.alpha = 0.3        # Stronger perturbations
        config.tau_stag = 4       # Faster stagnation detection
    
    return config
