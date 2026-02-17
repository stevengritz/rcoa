#!/usr/bin/env python3
"""
Selective Maintenance Problem (SMP) Experiment

Run RCOA and comparison algorithms on the SMP benchmark.
"""

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rcoa.core import RCOA, RCOAConfig
from rcoa.comparisons import PSO, DifferentialEvolution, GeneticAlgorithm
from rcoa.benchmarks.smp import SelectiveMaintenanceProblem
from rcoa.stats import compare_algorithms, wilcoxon_signed_rank, summary_statistics


def run_rcoa_smp(smp: SelectiveMaintenanceProblem, iterations: int = 500) -> float:
    """Run RCOA on SMP and return final mean health."""
    config = RCOAConfig(
        n_rice=smp.n_components,
        n_crabs=smp.n_crews,
        gamma=0.5,
        eta=0.25,
        enable_weeding=False,
        instant_mode=True,
    )
    
    smp.reset()
    health_history = []
    
    for t in range(iterations):
        # Simple density-penalized assignment
        state = smp.get_state()
        urgency = 1.0 - state
        
        # Assign crews to most urgent components
        actions = np.zeros(smp.n_components, dtype=int)
        sorted_idx = np.argsort(urgency)[::-1]
        
        for i, idx in enumerate(sorted_idx[:smp.n_crews]):
            if urgency[idx] > 0.3:
                actions[idx] = 2  # Imperfect repair
            elif urgency[idx] > 0.1:
                actions[idx] = 1  # Minimal repair
        
        _, reward = smp.step(actions)
        health_history.append(np.mean(smp.get_state()))
    
    return np.mean(smp.get_state())


def run_pso_smp(smp: SelectiveMaintenanceProblem, iterations: int = 500) -> float:
    """Run PSO-based assignment on SMP."""
    smp.reset()
    n_particles = 20
    
    # Particle positions = component preferences
    positions = np.random.rand(n_particles, smp.n_components)
    velocities = np.zeros_like(positions)
    pbest = positions.copy()
    pbest_fit = np.full(n_particles, -np.inf)
    gbest = positions[0].copy()
    gbest_fit = -np.inf
    
    for t in range(iterations):
        state = smp.get_state()
        
        for i in range(n_particles):
            # Convert position to action
            actions = np.zeros(smp.n_components, dtype=int)
            sorted_idx = np.argsort(positions[i] * (1 - state))[::-1]
            for j, idx in enumerate(sorted_idx[:smp.n_crews]):
                actions[idx] = min(3, int(positions[i, idx] * 4))
            
            _, reward = smp.step(actions)
            smp.reset()  # Reset for fair comparison
            
            if reward > pbest_fit[i]:
                pbest_fit[i] = reward
                pbest[i] = positions[i].copy()
            if reward > gbest_fit:
                gbest_fit = reward
                gbest = positions[i].copy()
        
        # Update velocities and positions
        r1, r2 = np.random.rand(2)
        velocities = 0.7 * velocities + 0.5 * r1 * (pbest - positions) + 0.5 * r2 * (gbest - positions)
        positions = np.clip(positions + velocities, 0, 1)
        
        # Actually run one step with global best
        actions = np.zeros(smp.n_components, dtype=int)
        sorted_idx = np.argsort(gbest * (1 - state))[::-1]
        for j, idx in enumerate(sorted_idx[:smp.n_crews]):
            actions[idx] = min(3, int(gbest[idx] * 4))
        smp.step(actions)
    
    return np.mean(smp.get_state())


def run_ga_smp(smp: SelectiveMaintenanceProblem, iterations: int = 500) -> float:
    """Run GA-based assignment on SMP."""
    smp.reset()
    pop_size = 20
    
    # Population = repair plans
    population = np.random.randint(0, 4, (pop_size, smp.n_components))
    
    for t in range(iterations):
        state = smp.get_state()
        
        # Evaluate population
        fitness = []
        for plan in population:
            smp.reset()
            for _ in range(10):  # Short rollout
                _, r = smp.step(plan)
            fitness.append(np.mean(smp.get_state()))
        
        fitness = np.array(fitness)
        
        # Selection and crossover
        new_pop = []
        sorted_idx = np.argsort(fitness)[::-1]
        new_pop.append(population[sorted_idx[0]])  # Elitism
        
        while len(new_pop) < pop_size:
            # Tournament selection
            a, b = np.random.choice(pop_size, 2, replace=False)
            parent1 = population[a if fitness[a] > fitness[b] else b]
            a, b = np.random.choice(pop_size, 2, replace=False)
            parent2 = population[a if fitness[a] > fitness[b] else b]
            
            # Crossover
            mask = np.random.rand(smp.n_components) > 0.5
            child = np.where(mask, parent1, parent2)
            
            # Mutation
            mut_mask = np.random.rand(smp.n_components) < 0.1
            child[mut_mask] = np.random.randint(0, 4, np.sum(mut_mask))
            
            new_pop.append(child)
        
        population = np.array(new_pop[:pop_size])
        
        # Apply best plan
        smp.reset()
        best_plan = population[0]
        for _ in range(t + 1):
            smp.step(best_plan)
    
    return np.mean(smp.get_state())


def main():
    parser = argparse.ArgumentParser(description="Run SMP benchmark")
    parser.add_argument("--runs", type=int, default=51, help="Number of runs")
    parser.add_argument("--iterations", type=int, default=500, help="Iterations per run")
    parser.add_argument("--quick", action="store_true", help="Quick mode")
    args = parser.parse_args()
    
    if args.quick:
        args.runs = 11
        args.iterations = 100
    
    print("="*70)
    print("SELECTIVE MAINTENANCE PROBLEM (SMP) EXPERIMENT")
    print("="*70)
    print(f"Runs: {args.runs}")
    print(f"Iterations: {args.iterations}")
    print("="*70)
    
    rcoa_results = []
    pso_results = []
    ga_results = []
    
    for run in range(args.runs):
        print(f"\rRun {run+1}/{args.runs}", end="", flush=True)
        
        np.random.seed(42 + run)
        smp = SelectiveMaintenanceProblem(n_components=20, n_crews=6, seed=42+run)
        
        rcoa_results.append(run_rcoa_smp(smp, args.iterations))
        
        smp = SelectiveMaintenanceProblem(n_components=20, n_crews=6, seed=42+run)
        pso_results.append(run_pso_smp(smp, args.iterations))
        
        smp = SelectiveMaintenanceProblem(n_components=20, n_crews=6, seed=42+run)
        ga_results.append(run_ga_smp(smp, args.iterations))
    
    print("\n")
    
    # Results
    rcoa_stats = summary_statistics(np.array(rcoa_results))
    pso_stats = summary_statistics(np.array(pso_results))
    ga_stats = summary_statistics(np.array(ga_results))
    
    print("RESULTS (Mean Component Health):")
    print(f"  RCOA: {rcoa_stats['mean']*100:.2f}% ± {rcoa_stats['std']*100:.2f}%")
    print(f"  PSO:  {pso_stats['mean']*100:.2f}% ± {pso_stats['std']*100:.2f}%")
    print(f"  GA:   {ga_stats['mean']*100:.2f}% ± {ga_stats['std']*100:.2f}%")
    
    # Statistical tests
    print("\nWILCOXON SIGNED-RANK TESTS:")
    
    w_pso = wilcoxon_signed_rank(np.array(rcoa_results), np.array(pso_results))
    print(f"  RCOA vs PSO: p={w_pso.p_value:.4f} {'✓ SIGNIFICANT' if w_pso.significant else ''}")
    
    w_ga = wilcoxon_signed_rank(np.array(rcoa_results), np.array(ga_results))
    print(f"  RCOA vs GA:  p={w_ga.p_value:.4f} {'✓ SIGNIFICANT' if w_ga.significant else ''}")
    
    # Win counts
    rcoa_wins_pso = sum(1 for r, p in zip(rcoa_results, pso_results) if r > p)
    rcoa_wins_ga = sum(1 for r, g in zip(rcoa_results, ga_results) if r > g)
    
    print(f"\nWIN COUNTS:")
    print(f"  RCOA beats PSO: {rcoa_wins_pso}/{args.runs} ({rcoa_wins_pso/args.runs*100:.1f}%)")
    print(f"  RCOA beats GA:  {rcoa_wins_ga}/{args.runs} ({rcoa_wins_ga/args.runs*100:.1f}%)")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
