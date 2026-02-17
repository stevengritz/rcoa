"""
Feature Selection Benchmark

Synthetic noisy feature selection problem for evaluating
RCOA's weeding operator.
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class DataPoint:
    """A data point with features and label."""
    features: np.ndarray
    is_signal: bool


class FeatureSelectionProblem:
    """
    Noisy Feature Selection Problem.
    
    Generates synthetic data with:
    - Signal dimensions: Gaussian clusters
    - Noise dimensions: Uniform random
    
    The goal is to identify and weight signal dimensions
    while suppressing noise dimensions.
    
    This problem showcases RCOA's weeding operator for
    embedded dimensionality reduction.
    
    Example:
        >>> from rcoa.benchmarks import FeatureSelectionProblem
        >>> problem = FeatureSelectionProblem(d_total=12, d_noise=5)
        >>> objective, bounds = problem.as_optimization_problem()
        >>> # Optimize feature mask and centroids
    """
    
    def __init__(
        self,
        d_total: int = 12,
        d_noise: int = 5,
        n_signal: int = 80,
        n_noise: int = 120,
        n_clusters: int = 3,
        cluster_spread: float = 0.5,
        seed: Optional[int] = None,
    ):
        """
        Initialize feature selection problem.
        
        Args:
            d_total: Total number of dimensions
            d_noise: Number of noise dimensions
            n_signal: Number of signal points
            n_noise: Number of noise points
            n_clusters: Number of Gaussian clusters
            cluster_spread: Standard deviation of clusters
            seed: Random seed for reproducibility
        """
        self.d_total = d_total
        self.d_noise = d_noise
        self.d_signal = d_total - d_noise
        self.n_signal = n_signal
        self.n_noise = n_noise
        self.n_clusters = n_clusters
        self.cluster_spread = cluster_spread
        
        if seed is not None:
            np.random.seed(seed)
        
        self.data: List[DataPoint] = []
        self.cluster_centers: List[np.ndarray] = []
        self.is_signal_dim: np.ndarray = np.zeros(d_total, dtype=bool)
        
        self._generate_data()
    
    def _generate_data(self):
        """Generate synthetic dataset."""
        # Mark signal dimensions
        self.is_signal_dim = np.array(
            [i < self.d_signal for i in range(self.d_total)]
        )
        
        # Generate cluster centers (only in signal dimensions)
        self.cluster_centers = []
        for _ in range(self.n_clusters):
            center = np.zeros(self.d_total)
            center[:self.d_signal] = np.random.uniform(-2, 2, self.d_signal)
            self.cluster_centers.append(center)
        
        # Generate signal points (clustered)
        self.data = []
        points_per_cluster = self.n_signal // self.n_clusters
        
        for c, center in enumerate(self.cluster_centers):
            for _ in range(points_per_cluster):
                features = np.zeros(self.d_total)
                # Signal dimensions: Gaussian around center
                features[:self.d_signal] = (
                    center[:self.d_signal] + 
                    np.random.normal(0, self.cluster_spread, self.d_signal)
                )
                # Noise dimensions: uniform random
                features[self.d_signal:] = np.random.uniform(-3, 3, self.d_noise)
                
                self.data.append(DataPoint(features=features, is_signal=True))
        
        # Generate noise points (uniform)
        for _ in range(self.n_noise):
            features = np.random.uniform(-4, 4, self.d_total)
            self.data.append(DataPoint(features=features, is_signal=False))
    
    def evaluate_mask(
        self,
        feature_mask: np.ndarray,
        centroid: np.ndarray,
        threshold: float = 2.0,
    ) -> Tuple[float, int, int]:
        """
        Evaluate a feature mask and centroid.
        
        Args:
            feature_mask: Weight for each dimension (0-1)
            centroid: Cluster centroid for classification
            threshold: Distance threshold for classification
            
        Returns:
            Tuple of (accuracy, true_positives, false_positives)
        """
        correct = 0
        true_pos = 0
        false_pos = 0
        
        for point in self.data:
            # Weighted distance
            diff = (point.features - centroid) * feature_mask
            distance = np.sqrt(np.sum(diff ** 2))
            
            # Classify based on distance
            classified_as_signal = distance < threshold
            
            if classified_as_signal == point.is_signal:
                correct += 1
            
            if classified_as_signal and point.is_signal:
                true_pos += 1
            if classified_as_signal and not point.is_signal:
                false_pos += 1
        
        accuracy = correct / len(self.data)
        return accuracy, true_pos, false_pos
    
    def evaluate_weeding(self, feature_weights: np.ndarray) -> dict:
        """
        Evaluate how well weeding identified noise dimensions.
        
        Args:
            feature_weights: Final weights from optimization
            
        Returns:
            Dictionary with weeding metrics
        """
        weeded_threshold = 0.3
        
        noise_dims_weeded = sum(
            1 for i in range(self.d_total)
            if not self.is_signal_dim[i] and feature_weights[i] < weeded_threshold
        )
        
        signal_dims_kept = sum(
            1 for i in range(self.d_total)
            if self.is_signal_dim[i] and feature_weights[i] >= weeded_threshold
        )
        
        false_positives = sum(
            1 for i in range(self.d_total)
            if self.is_signal_dim[i] and feature_weights[i] < weeded_threshold
        )
        
        return {
            'noise_dims_weeded': noise_dims_weeded,
            'total_noise_dims': self.d_noise,
            'signal_dims_kept': signal_dims_kept,
            'total_signal_dims': self.d_signal,
            'false_positives': false_positives,
            'weeding_precision': noise_dims_weeded / self.d_noise if self.d_noise > 0 else 0,
        }
    
    def fitness(
        self,
        x: np.ndarray,
        noise_penalty: float = 0.05,
    ) -> float:
        """
        Fitness function for optimization.
        
        Args:
            x: Solution vector [mask (d_total) | centroid (d_total)]
            noise_penalty: Penalty per active noise dimension
            
        Returns:
            Fitness value (higher is better)
        """
        mask = x[:self.d_total]
        centroid = x[self.d_total:]
        
        accuracy, _, _ = self.evaluate_mask(mask, centroid)
        
        # Penalty for keeping noise dimensions active
        penalty = 0
        for i in range(self.d_signal, self.d_total):
            penalty += mask[i] * noise_penalty
        
        return accuracy - penalty
    
    def as_optimization_problem(self) -> Tuple[callable, Tuple[np.ndarray, np.ndarray]]:
        """
        Convert to standard optimization problem format.
        
        Solution vector: [feature_mask (d_total) | centroid (d_total)]
        
        Returns:
            Tuple of (objective_function, bounds)
        """
        # Bounds: mask in [0,1], centroid in [-4,4]
        lb = np.concatenate([
            np.zeros(self.d_total),      # mask lower
            np.full(self.d_total, -4.0), # centroid lower
        ])
        ub = np.concatenate([
            np.ones(self.d_total),       # mask upper
            np.full(self.d_total, 4.0),  # centroid upper
        ])
        
        return self.fitness, (lb, ub)
    
    def get_optimal_mask(self) -> np.ndarray:
        """Get the ground-truth optimal feature mask."""
        return self.is_signal_dim.astype(float)
