"""
Experiment Runner

Utilities for running reproducible benchmark experiments.
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json
import time
from pathlib import Path

from rcoa.core import RCOA, RCOAConfig
from rcoa.comparisons.pso import PSO
from rcoa.comparisons.de import DifferentialEvolution
from rcoa.comparisons.ga import GeneticAlgorithm
from rcoa.stats import compare_algorithms, summary_statistics, wilcoxon_signed_rank


@dataclass
class ExperimentConfig:
    """Configuration for benchmark experiments."""
    n_runs: int = 51              # Paper standard: 51 independent runs
    max_fe_factor: int = 10000    # Max FEs = dimension * factor
    algorithms: List[str] = field(default_factory=lambda: ['RCOA', 'PSO', 'DE', 'GA'])
    dimensions: List[int] = field(default_factory=lambda: [10, 30, 50])
    seed_base: int = 42           # Base seed for reproducibility
    verbose: bool = True
    save_convergence: bool = False


@dataclass
class RunResult:
    """Result from a single optimization run."""
    algorithm: str
    problem: str
    dimension: int
    run_id: int
    best_fitness: float
    function_evaluations: int
    iterations: int
    wall_time: float
    convergence: Optional[List[float]] = None


@dataclass 
class ExperimentResult:
    """Result from a complete experiment."""
    config: ExperimentConfig
    results: List[RunResult]
    comparisons: Dict[str, Any]
    summary: Dict[str, Any]


def get_algorithm(name: str, pop_size: int = 50):
    """Get an algorithm instance by name."""
    if name == 'RCOA':
        config = RCOAConfig(n_rice=30, n_crabs=20)
        return RCOA(config)
    elif name == 'PSO':
        return PSO(n_particles=pop_size)
    elif name == 'DE':
        return DifferentialEvolution(pop_size=pop_size)
    elif name == 'GA':
        return GeneticAlgorithm(pop_size=pop_size)
    else:
        raise ValueError(f"Unknown algorithm: {name}")


def run_single(
    algorithm_name: str,
    objective_fn: Callable,
    bounds: Tuple[np.ndarray, np.ndarray],
    max_fe: int,
    problem_name: str = "unknown",
    run_id: int = 0,
    seed: Optional[int] = None,
    save_convergence: bool = False,
) -> RunResult:
    """
    Run a single optimization trial.
    
    Args:
        algorithm_name: Name of algorithm to use
        objective_fn: Objective function (to minimize)
        bounds: Search bounds
        max_fe: Maximum function evaluations
        problem_name: Name for logging
        run_id: Run identifier
        seed: Random seed
        save_convergence: Whether to save convergence history
        
    Returns:
        RunResult with trial outcomes
    """
    if seed is not None:
        np.random.seed(seed)
    
    D = len(bounds[0])
    algorithm = get_algorithm(algorithm_name)
    
    # Negate for maximization (all our algorithms maximize)
    def negated(x):
        return -objective_fn(x)
    
    start_time = time.time()
    
    result = algorithm.optimize(
        objective_fn=negated,
        bounds=bounds,
        max_fe=max_fe,
    )
    
    wall_time = time.time() - start_time
    
    return RunResult(
        algorithm=algorithm_name,
        problem=problem_name,
        dimension=D,
        run_id=run_id,
        best_fitness=-result.best_fitness,  # Convert back to minimization
        function_evaluations=result.function_evaluations,
        iterations=result.iterations,
        wall_time=wall_time,
        convergence=[-f for f in result.convergence_history] if save_convergence else None,
    )


def run_experiment(
    problems: Dict[str, Tuple[Callable, Tuple[np.ndarray, np.ndarray]]],
    config: Optional[ExperimentConfig] = None,
) -> ExperimentResult:
    """
    Run a complete benchmark experiment.
    
    Args:
        problems: Dictionary of {name: (objective_fn, bounds)}
        config: Experiment configuration
        
    Returns:
        ExperimentResult with all trials and comparisons
    """
    if config is None:
        config = ExperimentConfig()
    
    results: List[RunResult] = []
    
    total_runs = len(problems) * len(config.algorithms) * config.n_runs
    current_run = 0
    
    for problem_name, (objective_fn, bounds) in problems.items():
        D = len(bounds[0])
        max_fe = D * config.max_fe_factor
        
        if config.verbose:
            print(f"\n{'='*60}")
            print(f"Problem: {problem_name} (D={D})")
            print(f"{'='*60}")
        
        for alg_name in config.algorithms:
            alg_results = []
            
            for run_id in range(config.n_runs):
                current_run += 1
                seed = config.seed_base + run_id
                
                if config.verbose:
                    print(f"\r  {alg_name}: run {run_id+1}/{config.n_runs}", end="", flush=True)
                
                result = run_single(
                    algorithm_name=alg_name,
                    objective_fn=objective_fn,
                    bounds=bounds,
                    max_fe=max_fe,
                    problem_name=problem_name,
                    run_id=run_id,
                    seed=seed,
                    save_convergence=config.save_convergence,
                )
                
                results.append(result)
                alg_results.append(result.best_fitness)
            
            if config.verbose:
                stats = summary_statistics(np.array(alg_results))
                print(f"\r  {alg_name}: {stats['mean']:.4e} ± {stats['std']:.4e}")
    
    # Compute comparisons
    comparisons = compute_comparisons(results, config)
    summary = compute_summary(results, config)
    
    return ExperimentResult(
        config=config,
        results=results,
        comparisons=comparisons,
        summary=summary,
    )


def compute_comparisons(
    results: List[RunResult],
    config: ExperimentConfig,
) -> Dict[str, Any]:
    """Compute pairwise algorithm comparisons."""
    comparisons = {}
    
    # Group results by problem and algorithm
    grouped = {}
    for r in results:
        key = (r.problem, r.algorithm)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r.best_fitness)
    
    # Get unique problems
    problems = list(set(r.problem for r in results))
    
    # Compare RCOA against each other algorithm
    if 'RCOA' in config.algorithms:
        for other_alg in config.algorithms:
            if other_alg == 'RCOA':
                continue
            
            wins, losses, ties = 0, 0, 0
            all_rcoa = []
            all_other = []
            
            for problem in problems:
                rcoa_results = grouped.get((problem, 'RCOA'), [])
                other_results = grouped.get((problem, other_alg), [])
                
                if rcoa_results and other_results:
                    all_rcoa.extend(rcoa_results)
                    all_other.extend(other_results)
                    
                    # Per-problem comparison
                    rcoa_mean = np.mean(rcoa_results)
                    other_mean = np.mean(other_results)
                    
                    wilcoxon = wilcoxon_signed_rank(
                        np.array(rcoa_results),
                        np.array(other_results)
                    )
                    
                    # For small samples (n<10), Wilcoxon can't reach p<0.05
                    # Use practical significance: >10% relative difference
                    n_samples = len(rcoa_results)
                    if n_samples < 10:
                        # Use relative difference for small samples
                        rel_diff = abs(rcoa_mean - other_mean) / max(abs(other_mean), 1e-10)
                        if rel_diff > 0.1:  # >10% difference is practically significant
                            if rcoa_mean < other_mean:
                                wins += 1
                            else:
                                losses += 1
                        else:
                            ties += 1
                    elif wilcoxon.significant:
                        if rcoa_mean < other_mean:  # Lower is better (minimization)
                            wins += 1
                        else:
                            losses += 1
                    else:
                        ties += 1
            
            comparisons[f"RCOA_vs_{other_alg}"] = {
                'wins': wins,
                'ties': ties,
                'losses': losses,
                'win_rate': wins / len(problems) if problems else 0,
            }
    
    return comparisons


def compute_summary(
    results: List[RunResult],
    config: ExperimentConfig,
) -> Dict[str, Any]:
    """Compute summary statistics."""
    summary = {
        'total_runs': len(results),
        'algorithms': {},
    }
    
    for alg in config.algorithms:
        alg_results = [r for r in results if r.algorithm == alg]
        if alg_results:
            fitness_values = [r.best_fitness for r in alg_results]
            summary['algorithms'][alg] = {
                'mean_fitness': float(np.mean(fitness_values)),
                'std_fitness': float(np.std(fitness_values)),
                'mean_fe': float(np.mean([r.function_evaluations for r in alg_results])),
                'mean_time': float(np.mean([r.wall_time for r in alg_results])),
            }
    
    return summary


def save_results(
    experiment: ExperimentResult,
    output_dir: str = "results",
    prefix: str = "experiment",
) -> None:
    """Save experiment results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save summary as JSON
    summary_data = {
        'config': {
            'n_runs': experiment.config.n_runs,
            'algorithms': experiment.config.algorithms,
            'dimensions': experiment.config.dimensions,
        },
        'comparisons': experiment.comparisons,
        'summary': experiment.summary,
    }
    
    with open(output_path / f"{prefix}_summary.json", 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # Save detailed results as CSV
    csv_lines = ["algorithm,problem,dimension,run_id,best_fitness,fe,iterations,wall_time"]
    for r in experiment.results:
        csv_lines.append(
            f"{r.algorithm},{r.problem},{r.dimension},{r.run_id},"
            f"{r.best_fitness},{r.function_evaluations},{r.iterations},{r.wall_time:.4f}"
        )
    
    with open(output_path / f"{prefix}_results.csv", 'w') as f:
        f.write("\n".join(csv_lines))
    
    print(f"\nResults saved to {output_path}/")


def print_comparison_table(experiment: ExperimentResult) -> None:
    """Print formatted comparison table."""
    print("\n" + "="*70)
    print("COMPARISON SUMMARY (RCOA vs Others)")
    print("="*70)
    print(f"{'Comparison':<25} {'Win':<8} {'Tie':<8} {'Loss':<8} {'Win Rate':<10}")
    print("-"*70)
    
    for name, data in experiment.comparisons.items():
        print(f"{name:<25} {data['wins']:<8} {data['ties']:<8} {data['losses']:<8} {data['win_rate']:.2%}")
    
    print("="*70)
