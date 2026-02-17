# RCOA - Rice-Crab Optimization Algorithm

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A heterogeneous multi-agent metaheuristic for regenerative optimization problems.

## Overview

The Rice-Crab Optimization Algorithm (RCOA) is a novel bio-inspired metaheuristic based on the symbiotic co-culture of rice (*Oryza sativa*) and Chinese mitten crab (*Eriocheir sinensis*). Unlike conventional swarm intelligence methods, RCOA features:

- **Heterogeneous dual-population architecture**: Stationary Rice agents (candidate solutions) interact with mobile Crab agents (localized optimizers)
- **Four biologically-motivated operators**:
  - **Weeding**: Embedded dimensionality reduction via sensitivity analysis
  - **Fertilization**: Density-penalized gradient injection
  - **Bioturbation**: Lévy-flight perturbation for stagnation escape
  - **Molting**: Periodic elitism archiving

## Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Using pip
pip install -e .

# With development dependencies
uv pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from rcoa import RCOA, RCOAConfig

# Define objective function (RCOA maximizes by default)
def sphere(x):
    return -np.sum(x**2)  # Negate for minimization

# Configure and run
config = RCOAConfig(n_rice=30, n_crabs=20)
optimizer = RCOA(config)

bounds = (np.full(10, -5.0), np.full(10, 5.0))
result = optimizer.optimize(sphere, bounds, max_iterations=100)

print(f"Best fitness: {result.best_fitness}")
print(f"Best position: {result.best_position}")
```

## Benchmarks

Run the CEC-2017 benchmark suite:

```bash
# Quick test (5 runs, 4 functions)
python experiments/run_cec2017.py --quick

# Full benchmark (51 runs, all 30 functions)
python experiments/run_cec2017.py --runs 51 --dims 10 30 50

# Selective Maintenance Problem
python experiments/run_smp.py --runs 51
```

## Package Structure

```
rcoa/
├── src/rcoa/
│   ├── core.py           # Main RCOA algorithm
│   ├── agents.py         # Rice and Crab agent definitions
│   ├── operators.py      # Weeding, Fertilization, Bioturbation, Molting
│   ├── benchmarks/       # CEC-2017, SMP, Feature Selection
│   ├── comparisons/      # PSO, DE, GA implementations
│   ├── stats.py          # Statistical analysis (Wilcoxon, etc.)
│   └── tuning/           # Parameter sensitivity analysis
├── experiments/          # Benchmark experiment scripts
├── paper/                # LaTeX article
└── tests/                # Unit tests
```

## Algorithm Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Rice population | N | 30 | Number of stationary agents |
| Crab population | M | 20 | Number of mobile agents |
| Cannibalism coeff. | γ | 0.5 | Density penalty strength |
| Fertilization rate | η | 0.25 | Learning rate for solution improvement |
| Bioturbation scale | α | 0.2 | Lévy flight perturbation magnitude |
| Weeding threshold | ε | 0.15 | Sensitivity threshold for pruning |
| Stagnation threshold | τ_stag | 6 | Iterations before bioturbation |
| Molting period | τ_molt | 15 | Iterations between molting phases |

## When to Use RCOA

RCOA is designed for **regenerative optimization problems** where:

- Solutions degrade over time if unattended
- Mobile agents actively improve solutions through local intervention
- Resource constraints limit simultaneous attention
- The objective is to maintain system-wide quality over time

**Good applications**: Fleet maintenance scheduling, server cluster management, neural network pruning, noisy feature selection.

**Not recommended for**: Low-dimensional unimodal functions, tight evaluation budgets (<1000 FEs).

## Citation

```bibtex
@article{ritz2024rcoa,
  title={The Rice-Crab Optimization Algorithm: A Heterogeneous Multi-Agent 
         Metaheuristic for Regenerative Optimization Problems},
  author={Ritz, Steven and Claude and Gemini},
  journal={arXiv preprint},
  year={2026}
}
```

## Authors

- **Steven Ritz** - Primary author
- **Claude** (Anthropic) - Co-author
- **Gemini** (Google) - Co-author

## License

MIT License - see [LICENSE](LICENSE) for details.

## References

- Kennedy, J., & Eberhart, R. (1995). Particle Swarm Optimization. ICNN'95.
- Storn, R., & Price, K. (1997). Differential Evolution. J. Global Optimization.
- Cassady, C.R., et al. (2001). Selective Maintenance Modeling. J. Quality Maintenance Eng.
- Awad, N.H., et al. (2016). CEC 2017 Benchmark Suite. Technical Report.
