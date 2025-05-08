"""
ARLON (Aggregated Ranking of Level-based Ordinal Normalization) implementation.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from .base import BaseMCDMMethod
from ..data.decision_matrix import DecisionMatrix
from ..utils.normalization import normalize_matrix

class ARLON(BaseMCDMMethod):
    """
    ARLON (Aggregated Ranking of Level-based Ordinal Normalization) implementation.
    
    ARLON is based on the concept of level-based ordinal normalization and uses a specific
    aggregation procedure to calculate the final rankings of alternatives.
    """
    
    def __init__(self, 
                 decision_matrix: DecisionMatrix,
                 weights: Optional[np.ndarray] = None,
                 normalization_method: str = 'vector',
                 levels: Optional[int] = None):
        """
        Initialize ARLON method.
        
        Args:
            decision_matrix (DecisionMatrix): The decision matrix
            weights (Optional[np.ndarray]): Weights for criteria. If None, equal weights are used
            normalization_method (str): Method for normalizing the decision matrix
                                      ('vector', 'minmax', or 'sum')
            levels (Optional[int]): Number of levels for ordinal normalization. If None, 5 levels are used.
            
        Raises:
            ValueError: If weights contain negative values, levels is less than 1,
                      or normalization_method is not one of ('vector', 'minmax', 'sum')
        """
        super().__init__(decision_matrix)
        
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
        
        # Validate levels
        if levels is not None and levels < 1:
            raise ValueError("Number of levels must be at least 1")
        
        self.weights = weights
        self.normalization_method = normalization_method
        self.levels = levels if levels is not None else 5
        self.normalized_matrix = None
        self.ordinal_matrix = None
        self.weighted_matrix = None
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
        Normalize the decision matrix using ARLON-specific normalization.
        
        Returns:
            np.ndarray: Normalized decision matrix
        """
        matrix = self.decision_matrix.matrix
        n_alternatives, n_criteria = matrix.shape
        normalized = np.zeros_like(matrix, dtype=float)
        
        for j in range(n_criteria):
            col = matrix[:, j]
            if self.decision_matrix.criteria_types[j].lower() == 'benefit':
                min_val = np.min(col)
                max_val = np.max(col)
                if max_val == min_val:
                    normalized[:, j] = 1
                else:
                    normalized[:, j] = (col - min_val) / (max_val - min_val)
            else:  # cost criterion
                min_val = np.min(col)
                max_val = np.max(col)
                if max_val == min_val:
                    normalized[:, j] = 1
                else:
                    normalized[:, j] = (max_val - col) / (max_val - min_val)
        
        self.normalized_matrix = normalized
        return normalized
    
    def calculate_ordinal_matrix(self) -> np.ndarray:
        """
        Calculate the ordinal matrix based on normalized values.
        
        Returns:
            np.ndarray: Ordinal matrix with values in range [0, levels-1]
        """
        if self.normalized_matrix is None:
            self.normalize_matrix()
            
        n_alternatives, n_criteria = self.normalized_matrix.shape
        ordinal = np.zeros_like(self.normalized_matrix, dtype=int)
        
        for j in range(n_criteria):
            col = self.normalized_matrix[:, j]
            # Create bins for ordinal levels
            bins = np.linspace(0, 1, self.levels + 1)
            # Assign ordinal values based on bins (subtract 1 to get values in [0, levels-1])
            ordinal[:, j] = np.minimum(np.digitize(col, bins[:-1]) - 1, self.levels - 1)
        
        self.ordinal_matrix = ordinal
        return ordinal
    
    def calculate_weighted_matrix(self) -> np.ndarray:
        """
        Calculate the weighted ordinal matrix.
        
        Returns:
            np.ndarray: Weighted ordinal matrix
        """
        if self.ordinal_matrix is None:
            self.calculate_ordinal_matrix()
            
        if self.weights is None:
            self.calculate_weights()
            
        self.weighted_matrix = self.ordinal_matrix * self.weights
        return self.weighted_matrix
    
    def calculate_scores(self) -> np.ndarray:
        """
        Calculate ARLON scores for each alternative.
        
        Returns:
            np.ndarray: Array of ARLON scores
        """
        if self.weighted_matrix is None:
            self.calculate_weighted_matrix()
            
        self.scores = np.sum(self.weighted_matrix, axis=1)
        return self.scores
    
    def rank(self) -> Dict[str, Any]:
        """
        Rank alternatives based on ARLON scores.
        
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
            'ordinal_matrix': self.ordinal_matrix.tolist(),
            'weighted_matrix': self.weighted_matrix.tolist()
        }
        
        self.rankings = results
        return results 