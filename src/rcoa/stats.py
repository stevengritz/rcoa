"""
Statistical Analysis Tools

Statistical tests and utilities for metaheuristic comparison.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy import stats


@dataclass
class WilcoxonResult:
    """Result from Wilcoxon signed-rank test."""
    statistic: float
    p_value: float
    significant: bool
    effect_size: float  # r = Z / sqrt(N)
    winner: str  # 'A', 'B', or 'tie'


@dataclass
class ComparisonResult:
    """Result from algorithm comparison."""
    algorithm_a: str
    algorithm_b: str
    wins_a: int
    wins_b: int
    ties: int
    wilcoxon: WilcoxonResult
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float


def wilcoxon_signed_rank(
    a: np.ndarray,
    b: np.ndarray,
    alpha: float = 0.05,
) -> WilcoxonResult:
    """
    Perform Wilcoxon signed-rank test.
    
    Non-parametric test for paired samples. Tests whether the
    median difference between pairs is zero.
    
    Args:
        a: First sample (e.g., RCOA results)
        b: Second sample (e.g., PSO results)
        alpha: Significance level
        
    Returns:
        WilcoxonResult with test statistics
        
    Reference:
        Wilcoxon, F. (1945). Individual Comparisons by Ranking Methods.
        Biometrics Bulletin, 1(6), 80-83.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    
    # Remove ties (pairs with zero difference)
    diff = a - b
    nonzero_mask = diff != 0
    
    if np.sum(nonzero_mask) < 5:
        # Not enough data for Wilcoxon test (minimum 5 pairs)
        # Fall back to simple mean comparison for small samples
        mean_diff = np.mean(a) - np.mean(b)
        winner = 'A' if mean_diff < 0 else ('B' if mean_diff > 0 else 'tie')
        return WilcoxonResult(
            statistic=np.nan,
            p_value=1.0,
            significant=False,
            effect_size=0.0,
            winner=winner,
        )
    
    try:
        stat, p_value = stats.wilcoxon(a, b, alternative='two-sided')
    except Exception:
        return WilcoxonResult(
            statistic=np.nan,
            p_value=1.0,
            significant=False,
            effect_size=0.0,
            winner='tie',
        )
    
    # Effect size: r = Z / sqrt(N)
    n = len(a)
    z = stats.norm.ppf(1 - p_value / 2) if p_value < 1 else 0
    effect_size = abs(z) / np.sqrt(n)
    
    # Determine winner
    significant = p_value < alpha
    if significant:
        winner = 'A' if np.mean(a) > np.mean(b) else 'B'
    else:
        winner = 'tie'
    
    return WilcoxonResult(
        statistic=stat,
        p_value=p_value,
        significant=significant,
        effect_size=effect_size,
        winner=winner,
    )


def friedman_test(
    *samples: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[float, float, bool]:
    """
    Perform Friedman test for multiple algorithms.
    
    Non-parametric test for comparing more than two related samples.
    
    Args:
        *samples: Variable number of sample arrays
        alpha: Significance level
        
    Returns:
        Tuple of (statistic, p_value, significant)
        
    Reference:
        Friedman, M. (1937). The Use of Ranks to Avoid the Assumption of
        Normality Implicit in the Analysis of Variance.
    """
    try:
        stat, p_value = stats.friedmanchisquare(*samples)
        return stat, p_value, p_value < alpha
    except Exception:
        return np.nan, 1.0, False


def holm_bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[Tuple[int, float, bool]]:
    """
    Apply Holm-Bonferroni correction for multiple comparisons.
    
    Controls family-wise error rate while being less conservative
    than Bonferroni.
    
    Args:
        p_values: List of p-values from pairwise tests
        alpha: Family-wise significance level
        
    Returns:
        List of (original_index, adjusted_p, significant) tuples
        
    Reference:
        Holm, S. (1979). A Simple Sequentially Rejective Multiple Test
        Procedure. Scandinavian Journal of Statistics, 6(2), 65-70.
    """
    n = len(p_values)
    
    # Sort p-values with original indices
    indexed = [(i, p) for i, p in enumerate(p_values)]
    indexed.sort(key=lambda x: x[1])
    
    results = []
    for rank, (orig_idx, p) in enumerate(indexed):
        # Holm correction: alpha / (n - rank)
        adjusted_alpha = alpha / (n - rank)
        significant = p < adjusted_alpha
        
        # Adjusted p-value
        adjusted_p = min(p * (n - rank), 1.0)
        
        results.append((orig_idx, adjusted_p, significant))
    
    # Sort back to original order
    results.sort(key=lambda x: x[0])
    
    return results


def compare_algorithms(
    results_a: np.ndarray,
    results_b: np.ndarray,
    name_a: str = "Algorithm A",
    name_b: str = "Algorithm B",
    alpha: float = 0.05,
) -> ComparisonResult:
    """
    Compare two algorithms with statistical analysis.
    
    Args:
        results_a: Results from algorithm A (array of fitness values)
        results_b: Results from algorithm B
        name_a: Name of algorithm A
        name_b: Name of algorithm B
        alpha: Significance level
        
    Returns:
        ComparisonResult with all statistics
    """
    results_a = np.asarray(results_a)
    results_b = np.asarray(results_b)
    
    # Win/Tie/Loss counts (lower is better for minimization)
    wins_a = np.sum(results_a < results_b)
    wins_b = np.sum(results_b < results_a)
    ties = np.sum(results_a == results_b)
    
    # Wilcoxon test
    wilcoxon = wilcoxon_signed_rank(results_a, results_b, alpha)
    
    # Basic statistics
    mean_a = np.mean(results_a)
    mean_b = np.mean(results_b)
    std_a = np.std(results_a)
    std_b = np.std(results_b)
    
    return ComparisonResult(
        algorithm_a=name_a,
        algorithm_b=name_b,
        wins_a=int(wins_a),
        wins_b=int(wins_b),
        ties=int(ties),
        wilcoxon=wilcoxon,
        mean_a=mean_a,
        mean_b=mean_b,
        std_a=std_a,
        std_b=std_b,
    )


def aggregate_comparisons(
    comparisons: List[ComparisonResult],
) -> Dict[str, Dict]:
    """
    Aggregate comparison results across multiple problems.
    
    Args:
        comparisons: List of ComparisonResult from multiple problems
        
    Returns:
        Dictionary with aggregate statistics
    """
    if not comparisons:
        return {}
    
    name_a = comparisons[0].algorithm_a
    name_b = comparisons[0].algorithm_b
    
    total_wins_a = sum(c.wins_a for c in comparisons)
    total_wins_b = sum(c.wins_b for c in comparisons)
    total_ties = sum(c.ties for c in comparisons)
    
    sig_wins_a = sum(1 for c in comparisons 
                    if c.wilcoxon.significant and c.wilcoxon.winner == 'A')
    sig_wins_b = sum(1 for c in comparisons 
                    if c.wilcoxon.significant and c.wilcoxon.winner == 'B')
    sig_ties = len(comparisons) - sig_wins_a - sig_wins_b
    
    return {
        name_a: {
            'total_wins': total_wins_a,
            'significant_wins': sig_wins_a,
        },
        name_b: {
            'total_wins': total_wins_b,
            'significant_wins': sig_wins_b,
        },
        'ties': {
            'total': total_ties,
            'significant': sig_ties,
        },
        'win_tie_loss': f"{sig_wins_a}/{sig_ties}/{sig_wins_b}",
    }


def summary_statistics(results: np.ndarray) -> Dict[str, float]:
    """
    Compute summary statistics for a set of results.
    
    Args:
        results: Array of fitness values from multiple runs
        
    Returns:
        Dictionary with mean, std, min, max, median
    """
    results = np.asarray(results)
    return {
        'mean': float(np.mean(results)),
        'std': float(np.std(results)),
        'min': float(np.min(results)),
        'max': float(np.max(results)),
        'median': float(np.median(results)),
        'n_runs': len(results),
    }
