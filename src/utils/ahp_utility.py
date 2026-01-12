"""
Analytic Hierarchy Process (AHP) Calculator
For multi-criteria decision making in fitness evaluation
"""

import numpy as np
from typing import Dict, List


class AHPCalculator:
    """
    AHP Calculator for weight determination
    
    Implements the Analytic Hierarchy Process for calculating
    priority weights from pairwise comparison matrices
    """
    
    def __init__(self, criteria_weights: Dict[str, float] = None):
        """
        Initialize AHP calculator
        
        Args:
            criteria_weights: Initial criteria weights (optional)
        """
        self.criteria_weights = criteria_weights or {
            'iou': 0.4,
            'precision': 0.3,
            'recall': 0.2,
            'fps': 0.1
        }
        
        # Saaty's scale for pairwise comparisons
        self.saaty_scale = {
            1: 'Equal importance',
            3: 'Moderate importance',
            5: 'Strong importance',
            7: 'Very strong importance',
            9: 'Extreme importance'
        }
        
        # Consistency index for random matrices
        self.random_consistency = {
            1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
            6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
        }
    
    def create_pairwise_matrix(self, criteria: List[str] = None) -> np.ndarray:
        """
        Create pairwise comparison matrix from criteria weights
        
        Args:
            criteria: List of criteria names
            
        Returns:
            Pairwise comparison matrix
        """
        if criteria is None:
            criteria = list(self.criteria_weights.keys())
        
        n = len(criteria)
        matrix = np.ones((n, n))
        
        # Fill matrix with pairwise comparisons
        for i in range(n):
            for j in range(i + 1, n):
                weight_i = self.criteria_weights[criteria[i]]
                weight_j = self.criteria_weights[criteria[j]]
                
                # Calculate comparison value
                ratio = weight_i / weight_j if weight_j > 0 else 1
                matrix[i, j] = ratio
                matrix[j, i] = 1 / ratio
        
        return matrix
    
    def calculate_priority_vector(self, matrix: np.ndarray) -> np.ndarray:
        """
        Calculate priority vector using eigenvector method
        
        Args:
            matrix: Pairwise comparison matrix
            
        Returns:
            Priority vector (weights)
        """
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        
        # Find principal eigenvector (corresponding to max eigenvalue)
        max_eigenvalue_idx = np.argmax(eigenvalues.real)
        principal_eigenvector = eigenvectors[:, max_eigenvalue_idx].real
        
        # Normalize to get priority vector
        priority_vector = principal_eigenvector / np.sum(principal_eigenvector)
        
        return priority_vector
    
    def calculate_consistency_ratio(self, matrix: np.ndarray) -> float:
        """
        Calculate consistency ratio to check matrix validity
        
        Args:
            matrix: Pairwise comparison matrix
            
        Returns:
            Consistency ratio (should be < 0.1 for valid matrix)
        """
        n = matrix.shape[0]
        
        # Calculate maximum eigenvalue
        eigenvalues = np.linalg.eigvals(matrix)
        lambda_max = np.max(eigenvalues.real)
        
        # Calculate consistency index
        ci = (lambda_max - n) / (n - 1) if n > 1 else 0
        
        # Calculate consistency ratio
        ri = self.random_consistency.get(n, 1.49)
        cr = ci / ri if ri > 0 else 0
        
        return cr
    
    def calculate_weights(self, criteria: List[str] = None) -> Dict[str, float]:
        """
        Calculate final weights using AHP
        
        Args:
            criteria: List of criteria names
            
        Returns:
            Dictionary of criteria weights
        """
        if criteria is None:
            criteria = list(self.criteria_weights.keys())
        
        # Create pairwise comparison matrix
        matrix = self.create_pairwise_matrix(criteria)
        
        # Calculate priority vector
        priority_vector = self.calculate_priority_vector(matrix)
        
        # Check consistency
        cr = self.calculate_consistency_ratio(matrix)
        
        if cr > 0.1:
            print(f"Warning: Consistency ratio {cr:.3f} exceeds threshold 0.1")
        
        # Create weights dictionary
        weights = {
            criterion: weight
            for criterion, weight in zip(criteria, priority_vector)
        }
        
        return weights
    
    def update_weights_from_performance(
        self,
        performance_metrics: Dict[str, float],
        adaptation_rate: float = 0.1
    ) -> Dict[str, float]:
        """
        Adaptively update weights based on performance feedback
        
        Args:
            performance_metrics: Current performance on each criterion
            adaptation_rate: Rate of weight adaptation (0-1)
            
        Returns:
            Updated weights
        """
        # Calculate performance-based importance
        total_performance = sum(performance_metrics.values())
        
        if total_performance > 0:
            performance_weights = {
                k: (1.0 - v) / total_performance  # Inverse: lower performance = higher weight
                for k, v in performance_metrics.items()
            }
            
            # Blend current weights with performance-based weights
            updated_weights = {}
            for criterion in self.criteria_weights:
                if criterion in performance_weights:
                    current = self.criteria_weights[criterion]
                    target = performance_weights[criterion]
                    updated_weights[criterion] = (
                        (1 - adaptation_rate) * current +
                        adaptation_rate * target
                    )
                else:
                    updated_weights[criterion] = self.criteria_weights[criterion]
            
            # Normalize
            total = sum(updated_weights.values())
            updated_weights = {k: v / total for k, v in updated_weights.items()}
            
            self.criteria_weights = updated_weights
        
        return self.criteria_weights
    
    def get_weighted_score(
        self,
        metrics: Dict[str, float],
        weights: Dict[str, float] = None
    ) -> float:
        """
        Calculate weighted score from metrics
        
        Args:
            metrics: Dictionary of metric values
            weights: Dictionary of weights (uses self.criteria_weights if None)
            
        Returns:
            Weighted score
        """
        if weights is None:
            weights = self.criteria_weights
        
        score = 0.0
        for criterion, weight in weights.items():
            if criterion in metrics:
                score += weight * metrics[criterion]
        
        return score
