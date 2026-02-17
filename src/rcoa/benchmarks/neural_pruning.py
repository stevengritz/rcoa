"""
Neural Network Pruning Benchmark

Simulated neural network pruning problem for evaluating
sensitivity-based pruning via RCOA's weeding operator.
"""

import numpy as np
from typing import Tuple, List, Optional, Callable
from dataclasses import dataclass


@dataclass
class Neuron:
    """A neuron in the network."""
    layer: int
    index: int
    active: bool = True
    sensitivity: float = 1.0


class NeuralPruningProblem:
    """
    Neural Network Pruning Problem.
    
    Simulates a feedforward neural network where:
    - Neurons have varying sensitivities (importance)
    - Goal is to prune low-sensitivity neurons
    - Maintain accuracy while reducing network size
    
    This problem showcases RCOA's weeding operator for
    sensitivity-based pruning decisions.
    
    Example:
        >>> from rcoa.benchmarks import NeuralPruningProblem
        >>> problem = NeuralPruningProblem(
        ...     input_size=8, hidden_size=12, output_size=4
        ... )
        >>> objective, bounds = problem.as_optimization_problem()
    """
    
    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 12,
        output_size: int = 4,
        target_sparsity: float = 0.4,
        n_samples: int = 50,
        seed: Optional[int] = None,
    ):
        """
        Initialize neural pruning problem.
        
        Args:
            input_size: Number of input neurons
            hidden_size: Number of hidden neurons
            output_size: Number of output neurons
            target_sparsity: Target fraction of neurons to prune
            n_samples: Number of samples for accuracy evaluation
            seed: Random seed for reproducibility
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.target_sparsity = target_sparsity
        self.n_samples = n_samples
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize network
        self.neurons: List[Neuron] = []
        self.weights_ih: np.ndarray = np.array([])
        self.weights_ho: np.ndarray = np.array([])
        self.baseline_accuracy: float = 0.0
        
        self._init_network()
    
    def _init_network(self):
        """Initialize network architecture and weights."""
        # Create neurons
        self.neurons = []
        
        # Input layer
        for i in range(self.input_size):
            self.neurons.append(Neuron(layer=0, index=i, sensitivity=1.0))
        
        # Hidden layer (varying sensitivities)
        for i in range(self.hidden_size):
            # Some neurons are more important than others
            sensitivity = np.random.uniform(0.1, 1.0)
            self.neurons.append(Neuron(layer=1, index=i, sensitivity=sensitivity))
        
        # Output layer
        for i in range(self.output_size):
            self.neurons.append(Neuron(layer=2, index=i, sensitivity=1.0))
        
        # Initialize weights (Xavier initialization)
        self.weights_ih = np.random.randn(
            self.input_size, self.hidden_size
        ) * np.sqrt(2.0 / self.input_size)
        
        self.weights_ho = np.random.randn(
            self.hidden_size, self.output_size
        ) * np.sqrt(2.0 / self.hidden_size)
        
        # Compute baseline accuracy
        self.baseline_accuracy = self._compute_accuracy()
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, x)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def _forward(
        self,
        x: np.ndarray,
        hidden_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Forward pass through network.
        
        Args:
            x: Input vector
            hidden_mask: Mask for hidden neurons (1=active, 0=pruned)
            
        Returns:
            Output vector
        """
        if hidden_mask is None:
            hidden_mask = np.ones(self.hidden_size)
        
        # Input to hidden
        hidden = self._relu(x @ self.weights_ih) * hidden_mask
        
        # Hidden to output
        output = self._sigmoid(hidden @ self.weights_ho)
        
        return output
    
    def _compute_accuracy(
        self,
        hidden_mask: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute accuracy on synthetic task.
        
        Uses a simple XOR-like classification task.
        """
        if hidden_mask is None:
            hidden_mask = np.ones(self.hidden_size)
        
        correct = 0
        
        for _ in range(self.n_samples):
            # Generate random input
            x = np.random.uniform(0, 1, self.input_size)
            
            # Target: XOR-like pattern based on input sums
            target = int(np.sum(x) * 2) % self.output_size
            
            # Forward pass
            output = self._forward(x, hidden_mask)
            predicted = np.argmax(output)
            
            if predicted == target:
                correct += 1
        
        return correct / self.n_samples
    
    def compute_sensitivity(
        self,
        neuron_idx: int,
        n_trials: int = 20,
    ) -> float:
        """
        Compute sensitivity of a hidden neuron.
        
        Sensitivity = accuracy_with - accuracy_without
        
        Args:
            neuron_idx: Index of hidden neuron to test
            n_trials: Number of trials for accuracy estimation
            
        Returns:
            Sensitivity score
        """
        # Accuracy with neuron
        mask_with = np.ones(self.hidden_size)
        acc_with = self._compute_accuracy(mask_with)
        
        # Accuracy without neuron
        mask_without = np.ones(self.hidden_size)
        mask_without[neuron_idx] = 0
        acc_without = self._compute_accuracy(mask_without)
        
        return acc_with - acc_without
    
    def get_sparsity(self, hidden_mask: np.ndarray) -> float:
        """Get current sparsity (fraction of pruned neurons)."""
        return 1.0 - np.mean(hidden_mask > 0.5)
    
    def get_compression(self, hidden_mask: np.ndarray) -> float:
        """Get compression ratio."""
        active = np.sum(hidden_mask > 0.5)
        original_params = (
            self.input_size * self.hidden_size +
            self.hidden_size * self.output_size
        )
        pruned_params = (
            self.input_size * active +
            active * self.output_size
        )
        return 1.0 - (pruned_params / original_params) if original_params > 0 else 0
    
    def fitness(
        self,
        hidden_mask: np.ndarray,
        sparsity_weight: float = 0.3,
    ) -> float:
        """
        Fitness function for optimization.
        
        Balances accuracy and sparsity.
        
        Args:
            hidden_mask: Mask for hidden neurons (values in [0,1])
            sparsity_weight: Weight for sparsity term
            
        Returns:
            Fitness value (higher is better)
        """
        # Binarize mask
        binary_mask = (hidden_mask > 0.5).astype(float)
        
        # Compute accuracy
        accuracy = self._compute_accuracy(binary_mask)
        
        # Compute sparsity bonus
        sparsity = self.get_sparsity(binary_mask)
        sparsity_bonus = sparsity_weight * min(sparsity / self.target_sparsity, 1.0)
        
        # Penalize if too sparse (accuracy drops too much)
        accuracy_drop = self.baseline_accuracy - accuracy
        if accuracy_drop > 0.2:
            sparsity_bonus = 0
        
        return accuracy + sparsity_bonus
    
    def as_optimization_problem(self) -> Tuple[Callable, Tuple[np.ndarray, np.ndarray]]:
        """
        Convert to standard optimization problem format.
        
        Solution vector: hidden neuron mask
        
        Returns:
            Tuple of (objective_function, bounds)
        """
        lb = np.zeros(self.hidden_size)
        ub = np.ones(self.hidden_size)
        
        return self.fitness, (lb, ub)
    
    def evaluate_pruning(self, hidden_mask: np.ndarray) -> dict:
        """
        Evaluate pruning results.
        
        Args:
            hidden_mask: Final pruning mask
            
        Returns:
            Dictionary with pruning metrics
        """
        binary_mask = (hidden_mask > 0.5).astype(float)
        accuracy = self._compute_accuracy(binary_mask)
        sparsity = self.get_sparsity(binary_mask)
        compression = self.get_compression(binary_mask)
        
        # Compare with ground truth sensitivities
        hidden_neurons = [n for n in self.neurons if n.layer == 1]
        true_sensitivities = np.array([n.sensitivity for n in hidden_neurons])
        
        # Neurons that should be pruned (low sensitivity)
        should_prune = true_sensitivities < 0.3
        did_prune = binary_mask < 0.5
        
        correct_prunes = np.sum(should_prune & did_prune)
        false_prunes = np.sum((~should_prune) & did_prune)
        
        return {
            'accuracy': accuracy,
            'baseline_accuracy': self.baseline_accuracy,
            'accuracy_retention': accuracy / self.baseline_accuracy if self.baseline_accuracy > 0 else 0,
            'sparsity': sparsity,
            'compression': compression,
            'neurons_pruned': int(np.sum(did_prune)),
            'total_hidden': self.hidden_size,
            'correct_prunes': correct_prunes,
            'false_prunes': false_prunes,
        }
