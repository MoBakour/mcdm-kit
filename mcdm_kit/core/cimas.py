"""
CIMAS (Criterion Impact MeAsurement System) implementation.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from .base import BaseMCDMMethod
from ..data.decision_matrix import DecisionMatrix


class CIMAS(BaseMCDMMethod):
    """
    CIMAS (Criterion Impact MeAsurement System) implementation.

    CIMAS ranks alternatives by comparing their distances to ideal and anti-ideal
    solutions based on a weighted normalized decision matrix.
    """

    def __init__(self,
                 decision_matrix: DecisionMatrix,
                 weights: Optional[np.ndarray] = None,
                 normalization_method: str = 'minmax'):
        """
        Initialize CIMAS method.

        Args:
            decision_matrix (DecisionMatrix): The decision matrix
            weights (Optional[np.ndarray]): Weights for criteria. If None, equal weights are used
            normalization_method (str): Method for normalizing the decision matrix
                                        ('vector', 'minmax', or 'sum')
        """
        super().__init__(decision_matrix)

        if decision_matrix.matrix.size == 0:
            raise ValueError("Decision matrix cannot be empty")

        if weights is not None:
            if np.any(weights < 0):
                raise ValueError("Weights cannot be negative")
            if len(weights) != len(self.decision_matrix.criteria):
                raise ValueError("Number of weights must match number of criteria")

        valid_methods = ('vector', 'minmax', 'sum')
        if normalization_method not in valid_methods:
            raise ValueError(f"Normalization method must be one of {valid_methods}")

        self.weights = weights
        self.normalization_method = normalization_method
        self.normalized_matrix = None
        self.weighted_matrix = None
        self.scores = None

    def calculate_weights(self) -> np.ndarray:
        if self.weights is None:
            n_criteria = len(self.decision_matrix.criteria)
            self.weights = np.ones(n_criteria) / n_criteria
        self.weights = self.weights / np.sum(self.weights)
        return self.weights

    def normalize_matrix(self) -> np.ndarray:
        matrix = self.decision_matrix.matrix
        n_alternatives, n_criteria = matrix.shape
        normalized = np.zeros_like(matrix, dtype=float)

        for j in range(n_criteria):
            col = matrix[:, j]
            min_val = np.min(col)
            max_val = np.max(col)

            if max_val == min_val:
                normalized[:, j] = 1.0
            else:
                if self.decision_matrix.criteria_types[j].lower() == 'benefit':
                    normalized[:, j] = (col - min_val) / (max_val - min_val)
                else:  # cost
                    normalized[:, j] = (max_val - col) / (max_val - min_val)

        self.normalized_matrix = normalized
        return normalized

    def calculate_weighted_matrix(self) -> np.ndarray:
        if self.normalized_matrix is None:
            self.normalize_matrix()
        if self.weights is None:
            self.calculate_weights()
        self.weighted_matrix = self.normalized_matrix * self.weights
        return self.weighted_matrix

    def calculate_distances(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate distances to ideal and anti-ideal solutions using Manhattan distance.
        """
        ideal = np.max(self.weighted_matrix, axis=0)
        anti_ideal = np.min(self.weighted_matrix, axis=0)

        D_plus = np.sum(np.abs(self.weighted_matrix - ideal), axis=1)
        D_minus = np.sum(np.abs(self.weighted_matrix - anti_ideal), axis=1)

        return D_plus, D_minus

    def calculate_scores(self) -> np.ndarray:
        """
        Calculate CIMAS scores for each alternative.
        """
        if self.weighted_matrix is None:
            self.calculate_weighted_matrix()

        D_plus, D_minus = self.calculate_distances()
        self.scores = D_minus / (D_plus + D_minus + 1e-10)  # epsilon to avoid division by zero
        return self.scores

    def rank(self) -> Dict[str, Any]:
        """
        Rank alternatives based on CIMAS scores.

        Returns:
            Dict[str, Any]: Rankings, scores, and matrices
        """
        if self.scores is None:
            self.calculate_scores()

        ranking = np.argsort(-self.scores)  # descending order

        results = {
            'rankings': [
                {
                    'rank': i + 1,
                    'alternative': self.decision_matrix.alternatives[idx],
                    'score': float(self.scores[idx])
                }
                for i, idx in enumerate(ranking)
            ],
            'scores': dict(zip(self.decision_matrix.alternatives, self.scores)),
            'weighted_matrix': self.weighted_matrix.tolist(),
            'normalized_matrix': self.normalized_matrix.tolist()
        }

        self.rankings = results
        return results
