"""
ARTASI (Additive Ratio Transition to Aspiration Solution Integration) implementation.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from .base import BaseMCDMMethod
from ..data.decision_matrix import DecisionMatrix
from ..utils.normalization import normalize_matrix

class ARTASI(BaseMCDMMethod):
    """
    ARTASI (Additive Ratio Transition to Aspiration Solution Integration) implementation.
    
    ARTASI is based on the concept of aspiration levels and uses a specific normalization
    procedure to calculate the distance of each alternative from the aspiration solution.
    """
    
    def __init__(self, 
                 decision_matrix: DecisionMatrix,
                 weights: Optional[np.ndarray] = None,
                 normalization_method: str = 'vector',
                 aspiration_levels: Optional[np.ndarray] = None):
        """
        Initialize ARTASI method.
        
        Args:
            decision_matrix (DecisionMatrix): The decision matrix
            weights (Optional[np.ndarray]): Weights for criteria. If None, equal weights are used
            normalization_method (str): Method for normalizing the decision matrix
                                      ('vector', 'minmax', or 'sum')
            aspiration_levels (Optional[np.ndarray]): Aspiration levels for each criterion.
                                                    If None, maximum values are used for benefit
                                                    criteria and minimum values for cost criteria.
                                                    
        Raises:
            ValueError: If decision matrix is empty, weights are invalid,
                      or aspiration levels are invalid
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
            
        # Validate aspiration levels
        if aspiration_levels is not None:
            if len(aspiration_levels) != len(self.decision_matrix.criteria):
                raise ValueError("Number of aspiration levels must match number of criteria")
            if np.any(aspiration_levels < 0):
                raise ValueError("Aspiration levels cannot be negative")
        
        self.weights = weights
        self.normalization_method = normalization_method
        self.aspiration_levels = aspiration_levels
        self.normalized_matrix = None
        self.weighted_matrix = None
        self.aspiration_matrix = None
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
    
    def calculate_aspiration_levels(self) -> np.ndarray:
        """
        Calculate aspiration levels for each criterion.
        
        Returns:
            np.ndarray: Array of aspiration levels
        """
        if self.aspiration_levels is not None:
            if len(self.aspiration_levels) != len(self.decision_matrix.criteria):
                raise ValueError("Number of aspiration levels must match number of criteria")
            return self.aspiration_levels
        
        matrix = self.decision_matrix.matrix
        n_criteria = len(self.decision_matrix.criteria)
        aspiration = np.zeros(n_criteria)
        
        for j in range(n_criteria):
            col = matrix[:, j]
            if self.decision_matrix.criteria_types[j].lower() == 'benefit':
                aspiration[j] = np.max(col)
            else:  # cost criterion
                aspiration[j] = np.min(col)
        
        self.aspiration_levels = aspiration
        return aspiration
    
    def normalize_matrix(self) -> np.ndarray:
        """
        Normalize the decision matrix using ARTASI-specific normalization.
        
        Returns:
            np.ndarray: Normalized decision matrix
        """
        matrix = self.decision_matrix.matrix
        n_alternatives, n_criteria = matrix.shape
        normalized = np.zeros_like(matrix, dtype=float)
        
        if self.aspiration_levels is None:
            self.calculate_aspiration_levels()
        
        for j in range(n_criteria):
            col = matrix[:, j]
            aspiration = self.aspiration_levels[j]
            
            if aspiration == 0:
                normalized[:, j] = 1
            else:
                if self.decision_matrix.criteria_types[j].lower() == 'benefit':
                    normalized[:, j] = col / aspiration
                else:  # cost criterion
                    normalized[:, j] = aspiration / col
        
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
    
    def calculate_aspiration_matrix(self) -> np.ndarray:
        """
        Calculate the aspiration matrix.
        
        Returns:
            np.ndarray: Aspiration matrix
        """
        if self.weighted_matrix is None:
            self.calculate_weighted_matrix()
            
        n_alternatives, n_criteria = self.weighted_matrix.shape
        aspiration = np.zeros_like(self.weighted_matrix)
        
        for j in range(n_criteria):
            aspiration[:, j] = self.weights[j]
        
        self.aspiration_matrix = aspiration
        return aspiration
    
    def calculate_distance_matrix(self) -> np.ndarray:
        """
        Calculate the distance matrix from the aspiration solution.
        
        Returns:
            np.ndarray: Distance matrix
        """
        if self.aspiration_matrix is None:
            self.calculate_aspiration_matrix()
            
        self.distance_matrix = np.abs(self.weighted_matrix - self.aspiration_matrix)
        return self.distance_matrix
    
    def calculate_scores(self) -> np.ndarray:
        """
        Calculate ARTASI scores for each alternative.
        
        Returns:
            np.ndarray: Array of ARTASI scores
        """
        if self.distance_matrix is None:
            self.calculate_distance_matrix()
            
        self.scores = 1 / (1 + np.sum(self.distance_matrix, axis=1))
        return self.scores
    
    def rank(self) -> Dict[str, Any]:
        """
        Rank alternatives based on ARTASI scores.
        
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
            'aspiration_levels': dict(zip(self.decision_matrix.criteria, self.aspiration_levels)),
            'distance_matrix': self.distance_matrix.tolist()
        }
        
        self.rankings = results
        return results 