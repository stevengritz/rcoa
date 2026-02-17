#!/usr/bin/env python3
"""
CEC-2017 Benchmark Experiment

Run RCOA and comparison algorithms on CEC-2017 benchmark suite.
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rcoa.benchmarks.cec2017 import CEC2017, CEC2017_FUNCTIONS
from rcoa.experiment import (
    ExperimentConfig,
    run_experiment,
    save_results,
    print_comparison_table,
)


def main():
    parser = argparse.ArgumentParser(description="Run CEC-2017 benchmarks")
    parser.add_argument("--runs", type=int, default=51, help="Number of runs per config")
    parser.add_argument("--dims", type=int, nargs="+", default=[10, 30], help="Dimensions")
    parser.add_argument("--funcs", type=int, nargs="+", default=None, help="Function IDs (1-30)")
    parser.add_argument("--quick", action="store_true", help="Quick mode (5 runs, fewer functions)")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    args = parser.parse_args()
    
    if args.quick:
        args.runs = 5
        args.funcs = [1, 5, 12, 21]  # Sphere, Rastrigin, Ackley, Composition1
        args.dims = [10]
    
    if args.funcs is None:
        args.funcs = list(range(1, 31))  # All 30 functions
    
    print("="*70)
    print("CEC-2017 BENCHMARK EXPERIMENT")
    print("="*70)
    print(f"Functions: {args.funcs}")
    print(f"Dimensions: {args.dims}")
    print(f"Runs per config: {args.runs}")
    print("="*70)
    
    # Build problem set
    problems = {}
    
    for D in args.dims:
        benchmark = CEC2017(dimension=D)
        
        for func_id in args.funcs:
            if func_id not in CEC2017_FUNCTIONS:
                continue
            
            func_info = CEC2017_FUNCTIONS[func_id]
            name = f"F{func_id}_{func_info.name}_D{D}"
            
            problems[name] = (
                benchmark.get_function(func_id),
                benchmark.get_bounds(),
            )
    
    print(f"\nTotal problems: {len(problems)}")
    
    # Configure experiment
    config = ExperimentConfig(
        n_runs=args.runs,
        max_fe_factor=10000,
        algorithms=['RCOA', 'PSO', 'DE', 'GA'],
        dimensions=args.dims,
        verbose=True,
        save_convergence=False,
    )
    
    # Run experiment
    result = run_experiment(problems, config)
    
    # Print and save results
    print_comparison_table(result)
    save_results(result, args.output, "cec2017")
    
    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
