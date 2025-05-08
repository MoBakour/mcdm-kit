"""
DEMATEL (DEcision MAking Trial and Evaluation Laboratory) implementation.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from .base import BaseMCDMMethod
from ..data.decision_matrix import DecisionMatrix
from ..utils.normalization import normalize_matrix

class DEMATEL(BaseMCDMMethod):
    """
    DEMATEL (DEcision MAking Trial and Evaluation Laboratory) implementation.
    
    DEMATEL is a structural modeling method that can identify the cause-effect relationships
    between different criteria and visualize them in a structural model.
    """
    
    def __init__(self, 
                 decision_matrix: DecisionMatrix,
                 threshold: Optional[float] = None,
                 alpha: float = 0.1):
        """
        Initialize DEMATEL method.
        
        Args:
            decision_matrix (DecisionMatrix): The decision matrix
            threshold (Optional[float]): Threshold for influence relationships. If None, mean + std is used.
            alpha (float): Alpha parameter for normalization (default: 0.1)
            
        Raises:
            ValueError: If threshold is negative or alpha is not in (0, 1]
        """
        super().__init__(decision_matrix)
        
        # Validate threshold
        if threshold is not None and threshold < 0:
            raise ValueError("Threshold cannot be negative")
            
        # Validate alpha
        if not 0 < alpha <= 1:
            raise ValueError("Alpha must be in range (0, 1]")
        
        self.threshold = threshold
        self.alpha = alpha
        self.normalized_matrix = None
        self.total_relation_matrix = None
        self.cause_effect_matrix = None
        self.influence_relationships = None
        self.scores = None
        self.weights = None
        
    def normalize_matrix(self) -> np.ndarray:
        """
        Normalize the decision matrix using DEMATEL-specific normalization.
        
        Returns:
            np.ndarray: Normalized decision matrix
        """
        matrix = self.decision_matrix.matrix
        n_criteria = len(self.decision_matrix.criteria)
        
        # Calculate the maximum row sum
        row_sums = np.sum(matrix, axis=1)
        max_row_sum = np.max(row_sums)
        
        # Normalize the matrix
        normalized = matrix / max_row_sum
        
        self.normalized_matrix = normalized
        return normalized
    
    def calculate_total_relation_matrix(self) -> np.ndarray:
        """
        Calculate the total relation matrix.
        
        Returns:
            np.ndarray: Total relation matrix
        """
        if self.normalized_matrix is None:
            self.normalize_matrix()
            
        n_criteria = len(self.decision_matrix.criteria)
        identity = np.eye(n_criteria)
        
        # Calculate (I - alpha * X)^(-1)
        inverse = np.linalg.inv(identity - self.alpha * self.normalized_matrix)
        
        # Calculate total relation matrix
        self.total_relation_matrix = self.alpha * self.normalized_matrix @ inverse
        return self.total_relation_matrix
    
    def calculate_cause_effect_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the cause-effect matrix and influence relationships.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Cause-effect matrix and influence relationships
        """
        if self.total_relation_matrix is None:
            self.calculate_total_relation_matrix()
            
        n_criteria = len(self.decision_matrix.criteria)
        
        # Calculate row and column sums
        row_sums = np.sum(self.total_relation_matrix, axis=1)
        col_sums = np.sum(self.total_relation_matrix, axis=0)
        
        # Calculate cause-effect matrix
        cause_effect = np.zeros((n_criteria, 2))
        cause_effect[:, 0] = row_sums + col_sums  # Prominence
        cause_effect[:, 1] = row_sums - col_sums  # Relation
        
        self.cause_effect_matrix = cause_effect
        
        # Calculate influence relationships
        if self.threshold is None:
            # Use mean + std as threshold
            self.threshold = np.mean(self.total_relation_matrix) + np.std(self.total_relation_matrix)
        
        relationships = []
        for i in range(n_criteria):
            for j in range(n_criteria):
                if i != j and self.total_relation_matrix[i, j] > self.threshold:
                    relationships.append({
                        'from': self.decision_matrix.criteria[i],
                        'to': self.decision_matrix.criteria[j],
                        'influence': self.total_relation_matrix[i, j]
                    })
        
        self.influence_relationships = relationships
        return cause_effect, relationships
    
    def calculate_scores(self) -> np.ndarray:
        """
        Calculate DEMATEL scores for each criterion.
        In DEMATEL, scores are based on prominence (D + R) values.
        
        Returns:
            np.ndarray: Array of DEMATEL scores
        """
        if self.cause_effect_matrix is None:
            self.calculate_cause_effect_matrix()
            
        # Use prominence (D + R) as scores
        self.scores = self.cause_effect_matrix[:, 0]
        return self.scores
        
    def calculate_weights(self) -> np.ndarray:
        """
        Calculate weights for criteria based on DEMATEL analysis.
        Weights are normalized prominence values.
        
        Returns:
            np.ndarray: Array of weights for each criterion
        """
        if self.scores is None:
            self.calculate_scores()
            
        # Normalize scores to get weights
        weights = self.scores / np.sum(self.scores)
        self.weights = weights
        return weights
    
    def rank(self) -> Dict[str, Any]:
        """
        Analyze criteria relationships and provide rankings.
        
        Returns:
            Dict[str, Any]: Dictionary containing analysis results
        """
        if self.cause_effect_matrix is None:
            self.calculate_cause_effect_matrix()
            
        # Sort criteria by prominence (row + column sum)
        prominence_ranking = np.argsort(-self.cause_effect_matrix[:, 0])
        
        # Sort criteria by relation (row - column sum)
        relation_ranking = np.argsort(-self.cause_effect_matrix[:, 1])
        
        # Prepare results
        results = {
            'criteria_analysis': [
                {
                    'criterion': self.decision_matrix.criteria[i],
                    'prominence': self.cause_effect_matrix[i, 0],
                    'relation': self.cause_effect_matrix[i, 1],
                    'prominence_rank': np.where(prominence_ranking == i)[0][0] + 1,
                    'relation_rank': np.where(relation_ranking == i)[0][0] + 1
                }
                for i in range(len(self.decision_matrix.criteria))
            ],
            'influence_relationships': self.influence_relationships,
            'total_relation_matrix': self.total_relation_matrix.tolist(),
            'threshold': self.threshold
        }
        
        self.rankings = results
        return results 