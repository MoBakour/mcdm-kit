"""
TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution) implementation.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from .base import BaseMCDMMethod
from ..data.decision_matrix import DecisionMatrix
from ..utils.normalization import normalize_matrix
from ..utils.distance import euclidean_distance

class TOPSIS(BaseMCDMMethod):
    """
    TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution) implementation.
    
    TOPSIS is based on the concept that the chosen alternative should have the shortest
    geometric distance from the positive ideal solution and the longest geometric distance
    from the negative ideal solution.
    """
    
    def __init__(self, 
                 decision_matrix: DecisionMatrix,
                 weights: Optional[np.ndarray] = None,
                 normalization_method: str = 'vector'):
        """
        Initialize TOPSIS method.
        
        Args:
            decision_matrix (DecisionMatrix): The decision matrix
            weights (Optional[np.ndarray]): Weights for criteria. If None, equal weights are used
            normalization_method (str): Method for normalizing the decision matrix
                                      ('vector', 'minmax', or 'sum')
                                      
        Raises:
            ValueError: If weights contain negative values or normalization_method is invalid
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
        
        self.weights = weights
        self.normalization_method = normalization_method
        self.normalized_matrix = None
        self.ideal_solution = None
        self.anti_ideal_solution = None
        self.distances = None
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
        Normalize the decision matrix.
        
        Returns:
            np.ndarray: Normalized decision matrix
        """
        self.normalized_matrix = normalize_matrix(
            self.decision_matrix.matrix,
            self.decision_matrix.criteria_types,
            method=self.normalization_method
        )
        return self.normalized_matrix
    
    def calculate_ideal_solutions(self) -> tuple:
        """
        Calculate ideal and anti-ideal solutions.
        
        Returns:
            tuple: (ideal_solution, anti_ideal_solution)
        """
        if self.normalized_matrix is None:
            self.normalize_matrix()
            
        # Initialize ideal and anti-ideal solutions
        n_criteria = len(self.decision_matrix.criteria)
        ideal = np.zeros(n_criteria)
        anti_ideal = np.zeros(n_criteria)
        
        # Calculate ideal and anti-ideal solutions
        for j in range(n_criteria):
            if self.decision_matrix.criteria_types[j].lower() == 'benefit':
                ideal[j] = np.max(self.normalized_matrix[:, j])
                anti_ideal[j] = np.min(self.normalized_matrix[:, j])
            else:  # cost criterion
                ideal[j] = np.min(self.normalized_matrix[:, j])
                anti_ideal[j] = np.max(self.normalized_matrix[:, j])
        
        self.ideal_solution = ideal
        self.anti_ideal_solution = anti_ideal
        return ideal, anti_ideal
    
    def calculate_distances(self) -> tuple:
        """
        Calculate distances to ideal and anti-ideal solutions.
        
        Returns:
            tuple: (distances_to_ideal, distances_to_anti_ideal)
        """
        if self.ideal_solution is None or self.anti_ideal_solution is None:
            self.calculate_ideal_solutions()
            
        if self.weights is None:
            self.calculate_weights()
            
        # Calculate weighted distances
        n_alternatives = len(self.decision_matrix.alternatives)
        d_ideal = np.zeros(n_alternatives)
        d_anti_ideal = np.zeros(n_alternatives)
        
        for i in range(n_alternatives):
            d_ideal[i] = euclidean_distance(
                self.normalized_matrix[i] * self.weights,
                self.ideal_solution * self.weights
            )
            d_anti_ideal[i] = euclidean_distance(
                self.normalized_matrix[i] * self.weights,
                self.anti_ideal_solution * self.weights
            )
        
        self.distances = (d_ideal, d_anti_ideal)
        return d_ideal, d_anti_ideal
    
    def calculate_scores(self) -> np.ndarray:
        """
        Calculate TOPSIS scores for each alternative.
        
        Returns:
            np.ndarray: Array of TOPSIS scores
        """
        if self.distances is None:
            self.calculate_distances()
            
        d_ideal, d_anti_ideal = self.distances
        
        # Handle edge cases
        if np.all(d_ideal == 0) and np.all(d_anti_ideal == 0):
            # If all distances are zero, return equal scores
            n_alternatives = len(self.decision_matrix.alternatives)
            self.scores = np.ones(n_alternatives) / n_alternatives
        else:
            # Normal TOPSIS score calculation
            denominator = d_ideal + d_anti_ideal
            # Handle potential division by zero
            denominator[denominator == 0] = np.finfo(float).eps
            self.scores = d_anti_ideal / denominator
            
        return self.scores
    
    def rank(self) -> Dict[str, Any]:
        """
        Rank alternatives based on TOPSIS scores.
        
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
            'ideal_solution': dict(zip(self.decision_matrix.criteria, self.ideal_solution)),
            'anti_ideal_solution': dict(zip(self.decision_matrix.criteria, self.anti_ideal_solution))
        }
        
        self.rankings = results
        return results 