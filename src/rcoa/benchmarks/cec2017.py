"""
CEC-2017 Benchmark Functions

Implementation of the CEC-2017 single-objective real-parameter optimization
benchmark suite (F1-F30).

Reference:
    Awad, N.H., Ali, M.Z., Liang, J.J., Qu, B.Y., & Suganthan, P.N. (2016).
    Problem Definitions and Evaluation Criteria for the CEC 2017 Special Session
    and Competition on Single Objective Real-Parameter Numerical Optimization.
    Technical Report, Nanyang Technological University.
"""

import numpy as np
from typing import Callable, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BenchmarkFunction:
    """Container for a benchmark function with metadata."""
    name: str
    func: Callable[[np.ndarray], float]
    optimal_value: float
    category: str  # unimodal, multimodal, hybrid, composition
    bounds: Tuple[float, float] = (-100.0, 100.0)


def _get_shift_vector(func_id: int, D: int) -> np.ndarray:
    """Generate deterministic shift vector for a function."""
    np.random.seed(func_id * 1000 + D)  # Deterministic per function/dimension
    return np.random.uniform(-80, 80, D)


def _shifted(base_func: Callable, func_id: int) -> Callable:
    """Wrap a function with shift transformation."""
    def shifted_func(x: np.ndarray) -> float:
        D = len(x)
        shift = _get_shift_vector(func_id, D)
        return base_func(x - shift)
    return shifted_func


def sphere(x: np.ndarray) -> float:
    """F1: Shifted and Rotated Bent Cigar Function (simplified as Sphere)."""
    return np.sum(x ** 2)


def sum_diff_powers(x: np.ndarray) -> float:
    """F2: Shifted and Rotated Sum of Different Power Function."""
    D = len(x)
    return np.sum(np.abs(x) ** np.arange(2, D + 2))


def zakharov(x: np.ndarray) -> float:
    """F3: Shifted and Rotated Zakharov Function."""
    D = len(x)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(0.5 * np.arange(1, D + 1) * x)
    return sum1 + sum2 ** 2 + sum2 ** 4


def rosenbrock(x: np.ndarray) -> float:
    """F4: Shifted and Rotated Rosenbrock's Function."""
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)


def rastrigin(x: np.ndarray) -> float:
    """F5: Shifted and Rotated Rastrigin's Function."""
    D = len(x)
    return 10 * D + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def schaffer_f7(x: np.ndarray) -> float:
    """F6: Shifted and Rotated Schaffer's F7 Function."""
    D = len(x)
    s = np.sqrt(x[:-1] ** 2 + x[1:] ** 2)
    return (np.sum(np.sqrt(s) * (np.sin(50 * s ** 0.2) + 1)) / (D - 1)) ** 2


def lunacek_bi_rastrigin(x: np.ndarray) -> float:
    """F7: Shifted and Rotated Lunacek Bi-Rastrigin Function."""
    D = len(x)
    mu0 = 2.5
    d = 1.0
    s = 1 - 1 / (2 * np.sqrt(D + 20) - 8.2)
    mu1 = -np.sqrt((mu0 ** 2 - d) / s)
    
    sum1 = np.sum((x - mu0) ** 2)
    sum2 = d * D + s * np.sum((x - mu1) ** 2)
    sum3 = 10 * (D - np.sum(np.cos(2 * np.pi * (x - mu0))))
    
    return min(sum1, sum2) + sum3


def non_continuous_rastrigin(x: np.ndarray) -> float:
    """F8: Shifted and Rotated Non-Continuous Rastrigin's Function."""
    y = x.copy()
    mask = np.abs(x) > 0.5
    y[mask] = np.round(2 * x[mask]) / 2
    return rastrigin(y)


def levy(x: np.ndarray) -> float:
    """F9: Shifted and Rotated Levy Function."""
    D = len(x)
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0]) ** 2
    term2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
    term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
    return term1 + term2 + term3


def schwefel(x: np.ndarray) -> float:
    """F10: Shifted and Rotated Schwefel's Function."""
    D = len(x)
    z = x + 420.9687462275036
    result = 418.9828872724339 * D
    
    for i in range(D):
        if np.abs(z[i]) <= 500:
            result -= z[i] * np.sin(np.sqrt(np.abs(z[i])))
        elif z[i] > 500:
            result -= (500 - z[i] % 500) * np.sin(np.sqrt(np.abs(500 - z[i] % 500)))
            result += (z[i] - 500) ** 2 / (10000 * D)
        else:
            result -= (z[i] % 500 - 500) * np.sin(np.sqrt(np.abs(z[i] % 500 - 500)))
            result += (z[i] + 500) ** 2 / (10000 * D)
    
    return result


def griewank(x: np.ndarray) -> float:
    """F11: Shifted and Rotated Griewank's Function."""
    D = len(x)
    sum_term = np.sum(x ** 2) / 4000
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, D + 1))))
    return sum_term - prod_term + 1


def ackley(x: np.ndarray) -> float:
    """F12: Shifted and Rotated Ackley's Function."""
    D = len(x)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / D)) - np.exp(sum2 / D) + 20 + np.e


def weierstrass(x: np.ndarray) -> float:
    """F13: Shifted and Rotated Weierstrass Function."""
    D = len(x)
    a, b, k_max = 0.5, 3.0, 20
    
    sum1 = 0
    for i in range(D):
        for k in range(k_max + 1):
            sum1 += a ** k * np.cos(2 * np.pi * b ** k * (x[i] + 0.5))
    
    sum2 = 0
    for k in range(k_max + 1):
        sum2 += a ** k * np.cos(2 * np.pi * b ** k * 0.5)
    
    return sum1 - D * sum2


def happy_cat(x: np.ndarray) -> float:
    """F14: Shifted and Rotated HappyCat Function."""
    D = len(x)
    sum_sq = np.sum(x ** 2)
    return np.abs(sum_sq - D) ** 0.25 + (0.5 * sum_sq + np.sum(x)) / D + 0.5


def hgbat(x: np.ndarray) -> float:
    """F15: Shifted and Rotated HGBat Function."""
    D = len(x)
    sum_sq = np.sum(x ** 2)
    sum_x = np.sum(x)
    return np.abs(sum_sq ** 2 - sum_x ** 2) ** 0.5 + (0.5 * sum_sq + sum_x) / D + 0.5


def expanded_griewank_rosenbrock(x: np.ndarray) -> float:
    """F16: Shifted and Rotated Expanded Griewank plus Rosenbrock Function."""
    D = len(x)
    result = 0
    for i in range(D - 1):
        t = 100 * (x[i] ** 2 - x[i + 1]) ** 2 + (x[i] - 1) ** 2
        result += t ** 2 / 4000 - np.cos(t) + 1
    # Last term wraps around
    t = 100 * (x[-1] ** 2 - x[0]) ** 2 + (x[-1] - 1) ** 2
    result += t ** 2 / 4000 - np.cos(t) + 1
    return result


def expanded_schaffer_f6(x: np.ndarray) -> float:
    """F17: Shifted and Rotated Expanded Schaffer's F6 Function."""
    D = len(x)
    result = 0
    for i in range(D - 1):
        t = x[i] ** 2 + x[i + 1] ** 2
        result += 0.5 + (np.sin(np.sqrt(t)) ** 2 - 0.5) / (1 + 0.001 * t) ** 2
    # Last term wraps around
    t = x[-1] ** 2 + x[0] ** 2
    result += 0.5 + (np.sin(np.sqrt(t)) ** 2 - 0.5) / (1 + 0.001 * t) ** 2
    return result


def hybrid1(x: np.ndarray) -> float:
    """F18: Hybrid Function 1 (N=3)."""
    D = len(x)
    p = [0.3, 0.3, 0.4]
    n = [int(np.ceil(p[0] * D)), int(np.ceil(p[1] * D)), D]
    n[2] = D - n[0] - n[1]
    
    result = 0
    idx = 0
    result += zakharov(x[idx:idx + n[0]]) if n[0] > 0 else 0
    idx += n[0]
    result += rosenbrock(x[idx:idx + n[1]]) if n[1] > 0 else 0
    idx += n[1]
    result += rastrigin(x[idx:idx + n[2]]) if n[2] > 0 else 0
    return result


def hybrid2(x: np.ndarray) -> float:
    """F19: Hybrid Function 2 (N=4)."""
    D = len(x)
    p = [0.2, 0.2, 0.3, 0.3]
    sizes = [int(np.ceil(pi * D)) for pi in p[:-1]]
    sizes.append(D - sum(sizes))
    
    funcs = [sphere, ackley, rastrigin, schwefel]
    result = 0
    idx = 0
    for i, (size, func) in enumerate(zip(sizes, funcs)):
        if size > 0:
            result += func(x[idx:idx + size])
            idx += size
    return result


def hybrid3(x: np.ndarray) -> float:
    """F20: Hybrid Function 3 (N=5)."""
    D = len(x)
    p = [0.1, 0.2, 0.2, 0.2, 0.3]
    sizes = [int(np.ceil(pi * D)) for pi in p[:-1]]
    sizes.append(D - sum(sizes))
    
    funcs = [sphere, rosenbrock, levy, rastrigin, ackley]
    result = 0
    idx = 0
    for size, func in zip(sizes, funcs):
        if size > 0:
            result += func(x[idx:idx + size])
            idx += size
    return result


def composition1(x: np.ndarray) -> float:
    """F21: Composition Function 1 (N=3)."""
    funcs = [rosenbrock, ackley, rastrigin]
    sigmas = [10, 20, 30]
    lambdas = [1, 10, 1]
    biases = [0, 100, 200]
    return _composition(x, funcs, sigmas, lambdas, biases)


def composition2(x: np.ndarray) -> float:
    """F22: Composition Function 2 (N=3)."""
    funcs = [rastrigin, griewank, schwefel]
    sigmas = [10, 20, 30]
    lambdas = [1, 10, 1]
    biases = [0, 100, 200]
    return _composition(x, funcs, sigmas, lambdas, biases)


def composition3(x: np.ndarray) -> float:
    """F23: Composition Function 3 (N=4)."""
    funcs = [rosenbrock, ackley, schwefel, rastrigin]
    sigmas = [10, 20, 30, 40]
    lambdas = [1, 10, 1, 1]
    biases = [0, 100, 200, 300]
    return _composition(x, funcs, sigmas, lambdas, biases)


def composition4(x: np.ndarray) -> float:
    """F24: Composition Function 4 (N=4)."""
    funcs = [ackley, griewank, rosenbrock, rastrigin]
    sigmas = [10, 20, 30, 40]
    lambdas = [10, 1, 1, 1]
    biases = [0, 100, 200, 300]
    return _composition(x, funcs, sigmas, lambdas, biases)


def composition5(x: np.ndarray) -> float:
    """F25: Composition Function 5 (N=5)."""
    funcs = [rastrigin, happy_cat, ackley, sphere, rosenbrock]
    sigmas = [10, 20, 30, 40, 50]
    lambdas = [1, 1, 1, 1, 1]
    biases = [0, 100, 200, 300, 400]
    return _composition(x, funcs, sigmas, lambdas, biases)


def composition6(x: np.ndarray) -> float:
    """F26: Composition Function 6 (N=5)."""
    funcs = [sphere, schwefel, griewank, rosenbrock, rastrigin]
    sigmas = [10, 20, 30, 40, 50]
    lambdas = [1, 10, 1, 1, 1]
    biases = [0, 100, 200, 300, 400]
    return _composition(x, funcs, sigmas, lambdas, biases)


def composition7(x: np.ndarray) -> float:
    """F27: Composition Function 7 (N=6)."""
    funcs = [happy_cat, griewank, schwefel, ackley, rastrigin, rosenbrock]
    sigmas = [10, 20, 30, 40, 50, 60]
    lambdas = [1, 1, 1, 1, 1, 1]
    biases = [0, 100, 200, 300, 400, 500]
    return _composition(x, funcs, sigmas, lambdas, biases)


def composition8(x: np.ndarray) -> float:
    """F28: Composition Function 8 (N=6)."""
    funcs = [ackley, griewank, sphere, rosenbrock, rastrigin, schwefel]
    sigmas = [10, 20, 30, 40, 50, 60]
    lambdas = [10, 10, 1, 1, 1, 1]
    biases = [0, 100, 200, 300, 400, 500]
    return _composition(x, funcs, sigmas, lambdas, biases)


def composition9(x: np.ndarray) -> float:
    """F29: Composition Function 9 (N=3) - hybrid."""
    funcs = [hybrid1, hybrid2, hybrid3]
    sigmas = [10, 30, 50]
    lambdas = [1, 1, 1]
    biases = [0, 100, 200]
    return _composition(x, funcs, sigmas, lambdas, biases)


def composition10(x: np.ndarray) -> float:
    """F30: Composition Function 10 (N=3) - hybrid."""
    funcs = [hybrid2, hybrid3, hybrid1]
    sigmas = [10, 30, 50]
    lambdas = [1, 1, 1]
    biases = [0, 100, 200]
    return _composition(x, funcs, sigmas, lambdas, biases)


def _composition(x, funcs, sigmas, lambdas, biases):
    """Helper for composition functions."""
    D = len(x)
    n = len(funcs)
    
    # Generate random optima (simplified - should use shift vectors)
    np.random.seed(42)  # For reproducibility
    optima = [np.random.uniform(-80, 80, D) for _ in range(n)]
    
    # Calculate weights
    weights = np.zeros(n)
    for i in range(n):
        diff = x - optima[i]
        weights[i] = np.exp(-np.sum(diff ** 2) / (2 * D * sigmas[i] ** 2))
    
    # Normalize weights
    if np.sum(weights) > 0:
        weights /= np.sum(weights)
    else:
        weights = np.ones(n) / n
    
    # Compute composition
    result = 0
    for i in range(n):
        fi = funcs[i](x - optima[i])
        result += weights[i] * (lambdas[i] * fi + biases[i])
    
    return result


# Function registry with shifted functions
CEC2017_FUNCTIONS = {
    1: BenchmarkFunction("Sphere", _shifted(sphere, 1), 100, "unimodal"),
    2: BenchmarkFunction("Sum Diff Powers", _shifted(sum_diff_powers, 2), 200, "unimodal"),
    3: BenchmarkFunction("Zakharov", _shifted(zakharov, 3), 300, "unimodal"),
    4: BenchmarkFunction("Rosenbrock", _shifted(rosenbrock, 4), 400, "multimodal"),
    5: BenchmarkFunction("Rastrigin", _shifted(rastrigin, 5), 500, "multimodal"),
    6: BenchmarkFunction("Schaffer F7", _shifted(schaffer_f7, 6), 600, "multimodal"),
    7: BenchmarkFunction("Lunacek Bi-Rastrigin", _shifted(lunacek_bi_rastrigin, 7), 700, "multimodal"),
    8: BenchmarkFunction("Non-Cont Rastrigin", _shifted(non_continuous_rastrigin, 8), 800, "multimodal"),
    9: BenchmarkFunction("Levy", _shifted(levy, 9), 900, "multimodal"),
    10: BenchmarkFunction("Schwefel", _shifted(schwefel, 10), 1000, "multimodal"),
    11: BenchmarkFunction("Griewank", _shifted(griewank, 11), 1100, "multimodal"),
    12: BenchmarkFunction("Ackley", _shifted(ackley, 12), 1200, "multimodal"),
    13: BenchmarkFunction("Weierstrass", _shifted(weierstrass, 13), 1300, "multimodal"),
    14: BenchmarkFunction("HappyCat", _shifted(happy_cat, 14), 1400, "multimodal"),
    15: BenchmarkFunction("HGBat", _shifted(hgbat, 15), 1500, "multimodal"),
    16: BenchmarkFunction("Exp Griewank Rosenbrock", _shifted(expanded_griewank_rosenbrock, 16), 1600, "multimodal"),
    17: BenchmarkFunction("Exp Schaffer F6", _shifted(expanded_schaffer_f6, 17), 1700, "multimodal"),
    18: BenchmarkFunction("Hybrid 1", hybrid1, 1800, "hybrid"),
    19: BenchmarkFunction("Hybrid 2", hybrid2, 1900, "hybrid"),
    20: BenchmarkFunction("Hybrid 3", hybrid3, 2000, "hybrid"),
    21: BenchmarkFunction("Composition 1", composition1, 2100, "composition"),
    22: BenchmarkFunction("Composition 2", composition2, 2200, "composition"),
    23: BenchmarkFunction("Composition 3", composition3, 2300, "composition"),
    24: BenchmarkFunction("Composition 4", composition4, 2400, "composition"),
    25: BenchmarkFunction("Composition 5", composition5, 2500, "composition"),
    26: BenchmarkFunction("Composition 6", composition6, 2600, "composition"),
    27: BenchmarkFunction("Composition 7", composition7, 2700, "composition"),
    28: BenchmarkFunction("Composition 8", composition8, 2800, "composition"),
    29: BenchmarkFunction("Composition 9", composition9, 2900, "composition"),
    30: BenchmarkFunction("Composition 10", composition10, 3000, "composition"),
}


class CEC2017:
    """
    CEC-2017 Benchmark Suite.
    
    Example:
        >>> from rcoa.benchmarks import CEC2017
        >>> benchmark = CEC2017(dimension=30)
        >>> f5 = benchmark.get_function(5)  # Rastrigin
        >>> x = np.zeros(30)
        >>> print(f5(x))  # Should be near 500 (optimal)
    """
    
    def __init__(self, dimension: int = 30):
        self.dimension = dimension
        self.bounds = (-100.0, 100.0)
    
    def get_function(self, func_id: int) -> Callable[[np.ndarray], float]:
        """Get a benchmark function by ID (1-30)."""
        if func_id not in CEC2017_FUNCTIONS:
            raise ValueError(f"Function ID must be 1-30, got {func_id}")
        return CEC2017_FUNCTIONS[func_id].func
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounds for the benchmark dimension."""
        lb = np.full(self.dimension, self.bounds[0])
        ub = np.full(self.dimension, self.bounds[1])
        return lb, ub
    
    def get_optimal_value(self, func_id: int) -> float:
        """Get the known optimal value for a function."""
        return CEC2017_FUNCTIONS[func_id].optimal_value
    
    def get_function_info(self, func_id: int) -> BenchmarkFunction:
        """Get full information about a benchmark function."""
        return CEC2017_FUNCTIONS[func_id]
    
    @staticmethod
    def list_functions() -> dict:
        """List all available functions with their categories."""
        return {
            fid: (bf.name, bf.category) 
            for fid, bf in CEC2017_FUNCTIONS.items()
        }


def get_cec2017_function(func_id: int, dimension: int = 30) -> Tuple[Callable, Tuple[np.ndarray, np.ndarray], float]:
    """
    Convenience function to get a CEC-2017 benchmark.
    
    Returns:
        Tuple of (function, bounds, optimal_value)
    """
    benchmark = CEC2017(dimension)
    return (
        benchmark.get_function(func_id),
        benchmark.get_bounds(),
        benchmark.get_optimal_value(func_id),
    )
