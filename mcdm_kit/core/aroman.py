"""
AROMAN (Additive Ratio Assessment with Multiple Criteria) implementation.
"""

import numpy as np
from typing import List, Dict, Union, Optional
from ..data.decision_matrix import DecisionMatrix
from ..utils.normalization import normalize_matrix
from ..utils.distance import calculate_euclidean_distance

class AROMAN:
    """AROMAN (Additive Ratio Assessment with Multiple Criteria) method implementation."""
    
    def __init__(self, 
                 decision_matrix: Union[DecisionMatrix, np.ndarray],
                 weights: Optional[List[float]] = None,
                 criteria_types: Optional[List[str]] = None):
        """
        Initialize AROMAN method.
        
        Args:
            decision_matrix: Decision matrix or DecisionMatrix object
            weights: List of criteria weights (optional)
            criteria_types: List of criterion types ('benefit' or 'cost') (optional)
        """
        if isinstance(decision_matrix, DecisionMatrix):
            self.matrix = decision_matrix.matrix
            self.alternatives = decision_matrix.alternatives
            self.criteria = decision_matrix.criteria
            self.criteria_types = decision_matrix.criteria_types
            # Use provided weights or equal weights if not provided
            self.weights = weights or [1/len(self.criteria)] * len(self.criteria)
        else:
            if not isinstance(decision_matrix, (np.ndarray, list)):
                raise ValueError("Decision matrix must be a numpy array or list")
            self.matrix = np.array(decision_matrix)
            if self.matrix.size == 0:
                raise ValueError("Decision matrix cannot be empty")
            if len(self.matrix.shape) != 2:
                raise ValueError("Decision matrix must be 2-dimensional")
                
            self.alternatives = [f"Alt_{i+1}" for i in range(len(self.matrix))]
            self.criteria = [f"Criterion_{i+1}" for i in range(self.matrix.shape[1])]
            self.criteria_types = criteria_types or ['benefit'] * self.matrix.shape[1]
            self.weights = weights or [1/self.matrix.shape[1]] * self.matrix.shape[1]
            
        if len(self.weights) != len(self.criteria):
            raise ValueError("Number of weights must match number of criteria")
        if len(self.criteria_types) != len(self.criteria):
            raise ValueError("Number of criterion types must match number of criteria")
            
        self.weights = np.array(self.weights)
        self.normalized_matrix = None
        self.weighted_matrix = None
        self.ideal_solution = None
        self.anti_ideal_solution = None
        self.scores = None
        self.rankings = None
    
    def normalize_matrix(self) -> np.ndarray:
        """
        Normalize the decision matrix using vector normalization.
        
        Returns:
            np.ndarray: Normalized decision matrix
        """
        self.normalized_matrix = _vector_normalization(self.matrix)
        return self.normalized_matrix
    
    def calculate_weighted_matrix(self) -> np.ndarray:
        """
        Calculate the weighted normalized decision matrix.
        
        Returns:
            np.ndarray: Weighted normalized decision matrix
        """
        if self.normalized_matrix is None:
            self.normalize_matrix()
            
        self.weighted_matrix = self.normalized_matrix * self.weights
        return self.weighted_matrix
    
    def calculate_ideal_solutions(self) -> tuple:
        """
        Calculate ideal and anti-ideal solutions.
        
        Returns:
            tuple: (ideal_solution, anti_ideal_solution)
        """
        if self.weighted_matrix is None:
            self.calculate_weighted_matrix()
            
        # Initialize ideal and anti-ideal solutions
        self.ideal_solution = np.zeros(len(self.criteria))
        self.anti_ideal_solution = np.zeros(len(self.criteria))
        
        # Calculate ideal and anti-ideal solutions for each criterion
        for j, criterion_type in enumerate(self.criteria_types):
            if criterion_type.lower() == 'benefit':
                self.ideal_solution[j] = np.max(self.weighted_matrix[:, j])
                self.anti_ideal_solution[j] = np.min(self.weighted_matrix[:, j])
            else:  # cost criterion
                self.ideal_solution[j] = np.min(self.weighted_matrix[:, j])
                self.anti_ideal_solution[j] = np.max(self.weighted_matrix[:, j])
        
        return self.ideal_solution, self.anti_ideal_solution
    
    def calculate_scores(self) -> Dict[str, float]:
        """
        Calculate AROMAN scores for each alternative.
        
        Returns:
            Dict[str, float]: Dictionary of alternative scores
        """
        if self.ideal_solution is None or self.anti_ideal_solution is None:
            self.calculate_ideal_solutions()
            
        # Calculate distances to ideal and anti-ideal solutions
        ideal_distances = np.array([
            calculate_euclidean_distance(row, self.ideal_solution)
            for row in self.weighted_matrix
        ])
        anti_ideal_distances = np.array([
            calculate_euclidean_distance(row, self.anti_ideal_solution)
            for row in self.weighted_matrix
        ])
        
        # Calculate AROMAN scores
        self.scores = anti_ideal_distances / (ideal_distances + anti_ideal_distances)
        
        # Create dictionary of alternative scores
        return dict(zip(self.alternatives, self.scores))
    
    def rank(self) -> Dict[str, int]:
        """
        Rank alternatives based on their AROMAN scores.
        
        Returns:
            Dict[str, int]: Dictionary of alternative rankings (1 is best)
        """
        if self.scores is None:
            self.calculate_scores()
            
        # Sort alternatives by score in descending order
        sorted_indices = np.argsort(-self.scores)
        rankings = {self.alternatives[i]: rank + 1 
                   for rank, i in enumerate(sorted_indices)}
        
        self.rankings = rankings
        return rankings

def _vector_normalization(matrix: np.ndarray) -> np.ndarray:
    """
    Normalize matrix using vector normalization.
    
    Args:
        matrix (np.ndarray): Decision matrix to normalize
        
    Returns:
        np.ndarray: Normalized matrix
    """
    # Calculate sum of squares for each column
    sum_squares = np.sqrt(np.sum(matrix ** 2, axis=0))
    
    # Avoid division by zero
    sum_squares[sum_squares == 0] = 1
    
    # Normalize
    return matrix / sum_squares 