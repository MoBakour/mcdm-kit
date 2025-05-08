"""
CIMAS (Criterion Impact MeAsurement System) implementation.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from .base import BaseMCDMMethod
from ..data.decision_matrix import DecisionMatrix
from ..utils.normalization import normalize_matrix

class CIMAS(BaseMCDMMethod):
    """
    CIMAS (Criterion Impact MeAsurement System) implementation.
    
    CIMAS is based on the concept of measuring the impact of each criterion on the
    overall decision and uses a specific normalization procedure to calculate the
    impact scores.
    """
    
    def __init__(self, 
                 decision_matrix: DecisionMatrix,
                 weights: Optional[np.ndarray] = None,
                 normalization_method: str = 'vector'):
        """
        Initialize CIMAS method.
        
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
        self.impact_matrix = None
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
        Normalize the decision matrix using CIMAS-specific normalization.
        
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
    
    def calculate_impact_matrix(self) -> np.ndarray:
        """
        Calculate the impact matrix.
        
        Returns:
            np.ndarray: Impact matrix
        """
        if self.weighted_matrix is None:
            self.calculate_weighted_matrix()
            
        n_alternatives, n_criteria = self.weighted_matrix.shape
        impact = np.zeros_like(self.weighted_matrix)
        
        for j in range(n_criteria):
            col = self.weighted_matrix[:, j]
            mean_val = np.mean(col)
            impact[:, j] = col - mean_val
        
        self.impact_matrix = impact
        return impact
    
    def calculate_scores(self) -> np.ndarray:
        """
        Calculate CIMAS scores for each alternative.
        
        Returns:
            np.ndarray: Array of CIMAS scores
        """
        if self.impact_matrix is None:
            self.calculate_impact_matrix()
            
        self.scores = np.sum(self.impact_matrix, axis=1)
        return self.scores
    
    def rank(self) -> Dict[str, Any]:
        """
        Rank alternatives based on CIMAS scores.
        
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
            'impact_matrix': self.impact_matrix.tolist(),
            'weighted_matrix': self.weighted_matrix.tolist()
        }
        
        self.rankings = results
        return results 