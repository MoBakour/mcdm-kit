"""
WISP (Weighted Integrated Score Performance) implementation.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from .base import BaseMCDMMethod
from ..data.decision_matrix import DecisionMatrix
from ..utils.normalization import normalize_matrix

class WISP(BaseMCDMMethod):
    """
    WISP (Weighted Integrated Score Performance) implementation.
    
    WISP is based on the concept of integrated performance scores and uses a specific
    normalization procedure to calculate the weighted performance of alternatives.
    """
    
    def __init__(self, 
                 decision_matrix: DecisionMatrix,
                 weights: Optional[np.ndarray] = None,
                 normalization_method: str = 'vector',
                 performance_thresholds: Optional[np.ndarray] = None):
        """
        Initialize WISP method.
        
        Args:
            decision_matrix (DecisionMatrix): The decision matrix
            weights (Optional[np.ndarray]): Weights for criteria. If None, equal weights are used
            normalization_method (str): Method for normalizing the decision matrix
                                      ('vector', 'minmax', or 'sum')
            performance_thresholds (Optional[np.ndarray]): Performance thresholds for each criterion.
                                                         If None, mean values are used.
                                                         
        Raises:
            ValueError: If weights contain negative values, normalization_method is invalid,
                      or performance thresholds contain negative values
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
            
        # Validate performance thresholds
        if performance_thresholds is not None:
            if np.any(performance_thresholds < 0):
                raise ValueError("Performance thresholds cannot be negative")
            if len(performance_thresholds) != len(self.decision_matrix.criteria):
                raise ValueError("Number of performance thresholds must match number of criteria")
        
        self.weights = weights
        self.normalization_method = normalization_method
        self.performance_thresholds = performance_thresholds
        self.normalized_matrix = None
        self.weighted_matrix = None
        self.performance_matrix = None
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
    
    def calculate_performance_thresholds(self) -> np.ndarray:
        """
        Calculate performance thresholds for each criterion.
        
        Returns:
            np.ndarray: Array of performance thresholds
        """
        if self.performance_thresholds is not None:
            if len(self.performance_thresholds) != len(self.decision_matrix.criteria):
                raise ValueError("Number of performance thresholds must match number of criteria")
            return self.performance_thresholds
        
        matrix = self.decision_matrix.matrix
        n_criteria = len(self.decision_matrix.criteria)
        thresholds = np.zeros(n_criteria)
        
        for j in range(n_criteria):
            col = matrix[:, j]
            if self.decision_matrix.criteria_types[j].lower() == 'benefit':
                thresholds[j] = np.max(col)
            else:  # cost criterion
                thresholds[j] = np.min(col)
        
        self.performance_thresholds = thresholds
        return thresholds
    
    def normalize_matrix(self) -> np.ndarray:
        """
        Normalize the decision matrix using WISP-specific normalization.
        
        Returns:
            np.ndarray: Normalized decision matrix
        """
        matrix = self.decision_matrix.matrix
        n_alternatives, n_criteria = matrix.shape
        normalized = np.zeros_like(matrix, dtype=float)
        
        if self.performance_thresholds is None:
            self.calculate_performance_thresholds()
        
        for j in range(n_criteria):
            col = matrix[:, j]
            threshold = self.performance_thresholds[j]
            
            if threshold == 0:
                normalized[:, j] = 1
            else:
                if self.decision_matrix.criteria_types[j].lower() == 'benefit':
                    normalized[:, j] = col / threshold
                else:  # cost criterion
                    normalized[:, j] = threshold / col
        
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
    
    def calculate_performance_matrix(self) -> np.ndarray:
        """
        Calculate the performance matrix.
        
        Returns:
            np.ndarray: Performance matrix
        """
        if self.weighted_matrix is None:
            self.calculate_weighted_matrix()
            
        n_alternatives, n_criteria = self.weighted_matrix.shape
        performance = np.zeros_like(self.weighted_matrix)
        
        for j in range(n_criteria):
            col = self.weighted_matrix[:, j]
            mean_val = np.mean(col)
            performance[:, j] = col - mean_val
        
        self.performance_matrix = performance
        return performance
    
    def calculate_scores(self) -> np.ndarray:
        """
        Calculate WISP scores for each alternative.
        
        Returns:
            np.ndarray: Array of WISP scores
        """
        if self.performance_matrix is None:
            self.calculate_performance_matrix()
            
        self.scores = np.sum(self.performance_matrix, axis=1)
        return self.scores
    
    def rank(self) -> Dict[str, Any]:
        """
        Rank alternatives based on WISP scores.
        
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
            'performance_thresholds': dict(zip(self.decision_matrix.criteria, self.performance_thresholds)),
            'performance_matrix': self.performance_matrix.tolist()
        }
        
        self.rankings = results
        return results 