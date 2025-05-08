"""
MABAC (Multi-Attributive Border Approximation area Comparison) implementation.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from .base import BaseMCDMMethod
from ..data.decision_matrix import DecisionMatrix
from ..utils.normalization import normalize_matrix

class MABAC(BaseMCDMMethod):
    """
    MABAC (Multi-Attributive Border Approximation area Comparison) implementation.
    
    MABAC is based on the concept of border approximation area and uses a specific
    normalization procedure to calculate the distance of each alternative from the
    border approximation area.
    """
    
    def __init__(self, 
                 decision_matrix: DecisionMatrix,
                 weights: Optional[np.ndarray] = None,
                 normalization_method: str = 'vector'):
        """
        Initialize MABAC method.
        
        Args:
            decision_matrix (DecisionMatrix): The decision matrix
            weights (Optional[np.ndarray]): Weights for criteria. If None, equal weights are used
            normalization_method (str): Method for normalizing the decision matrix
                                      ('vector', 'minmax', or 'sum')
                                      
        Raises:
            ValueError: If decision matrix is empty or weights are invalid
        """
        super().__init__(decision_matrix)
        
        # Validate decision matrix
        if decision_matrix.matrix.size == 0:
            raise ValueError("Decision matrix cannot be empty")
            
        # Validate weights
        if weights is not None:
            if np.any(weights < 0):
                raise ValueError("Weights cannot be negative")
            if len(weights) != len(self.decision_matrix.criteria):
                raise ValueError("Number of weights must match number of criteria")
        
        # Validate normalization method
        valid_methods = ('vector', 'minmax', 'sum')
        if normalization_method not in valid_methods:
            raise ValueError(f"Normalization method must be one of {valid_methods}")
        
        self.weights = weights
        self.normalization_method = normalization_method
        self.normalized_matrix = None
        self.weighted_matrix = None
        self.border_matrix = None
        self.distance_matrix = None
        self.scores = None
        
    def calculate_weights(self) -> np.ndarray:
        """
        Calculate or validate weights for criteria.
        
        Returns:
            np.ndarray: Array of weights for each criterion
        """
        if self.weights is None:
            # Use equal weights if none provided
            n_criteria = len(self.decision_matrix.criteria)
            self.weights = np.ones(n_criteria) / n_criteria
        elif len(self.weights) != len(self.decision_matrix.criteria):
            raise ValueError("Number of weights must match number of criteria")
        
        # Normalize weights to sum to 1
        self.weights = self.weights / np.sum(self.weights)
        return self.weights
    
    def normalize_matrix(self) -> np.ndarray:
        """
        Normalize the decision matrix using MABAC-specific normalization.
        
        Returns:
            np.ndarray: Normalized decision matrix
        """
        matrix = self.decision_matrix.matrix
        n_alternatives, n_criteria = matrix.shape
        normalized = np.zeros_like(matrix, dtype=float)
        
        for j in range(n_criteria):
            col = matrix[:, j]
            min_val = np.min(col)
            max_val = np.max(col)
            
            if max_val == min_val:
                normalized[:, j] = 1
            else:
                if self.decision_matrix.criteria_types[j].lower() == 'benefit':
                    normalized[:, j] = (col - min_val) / (max_val - min_val)
                else:  # cost criterion
                    normalized[:, j] = (max_val - col) / (max_val - min_val)
        
        self.normalized_matrix = normalized
        return normalized
    
    def calculate_weighted_matrix(self) -> np.ndarray:
        """
        Calculate the weighted normalized matrix.
        
        Returns:
            np.ndarray: Weighted normalized matrix
        """
        if self.normalized_matrix is None:
            self.normalize_matrix()
            
        if self.weights is None:
            self.calculate_weights()
            
        self.weighted_matrix = self.normalized_matrix * self.weights
        return self.weighted_matrix
    
    def calculate_border_matrix(self) -> np.ndarray:
        """
        Calculate the border approximation area matrix.
        
        Returns:
            np.ndarray: Border approximation area matrix
        """
        if self.weighted_matrix is None:
            self.calculate_weighted_matrix()
            
        n_alternatives, n_criteria = self.weighted_matrix.shape
        border = np.zeros(n_criteria)
        
        for j in range(n_criteria):
            border[j] = np.prod(self.weighted_matrix[:, j]) ** (1/n_alternatives)
        
        self.border_matrix = border
        return border
    
    def calculate_distance_matrix(self) -> np.ndarray:
        """
        Calculate the distance matrix from the border approximation area.
        
        Returns:
            np.ndarray: Distance matrix
        """
        if self.border_matrix is None:
            self.calculate_border_matrix()
            
        self.distance_matrix = self.weighted_matrix - self.border_matrix
        return self.distance_matrix
    
    def calculate_scores(self) -> np.ndarray:
        """
        Calculate MABAC scores for each alternative.
        
        Returns:
            np.ndarray: Array of MABAC scores
        """
        if self.distance_matrix is None:
            self.calculate_distance_matrix()
            
        self.scores = np.sum(self.distance_matrix, axis=1)
        return self.scores
    
    def rank(self) -> Dict[str, Any]:
        """
        Rank alternatives based on MABAC scores.
        
        Returns:
            Dict[str, Any]: Dictionary containing rankings and scores
        """
        if self.scores is None:
            self.calculate_scores()
            
        # Create ranking
        ranking = np.argsort(-self.scores)  # Sort in descending order
        
        # Prepare results
        results = {
            'rankings': [
                {
                    'rank': i + 1,
                    'alternative': self.decision_matrix.alternatives[idx],
                    'score': self.scores[idx]
                }
                for i, idx in enumerate(ranking)
            ],
            'scores': dict(zip(self.decision_matrix.alternatives, self.scores)),
            'border_matrix': dict(zip(self.decision_matrix.criteria, self.border_matrix)),
            'distance_matrix': self.distance_matrix.tolist()
        }
        
        self.rankings = results
        return results 