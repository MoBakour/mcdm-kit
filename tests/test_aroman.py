"""
Tests for AROMAN (Additive Ratio Assessment with Multiple Criteria) method.
"""

import unittest
import numpy as np
from mcdm_kit.core.aroman import AROMAN
from mcdm_kit.data.decision_matrix import DecisionMatrix

class TestAROMAN(unittest.TestCase):
    """Test cases for AROMAN method."""
    
    def setUp(self):
        """Set up test data."""
        # Example decision matrix
        self.matrix = np.array([
            [4, 3, 5, 2],
            [3, 4, 3, 3],
            [5, 2, 4, 4],
            [2, 5, 2, 5]
        ])
        
        # Example weights
        self.weights = [0.3, 0.2, 0.3, 0.2]
        
        # Example criteria types
        self.criteria_types = ['benefit', 'benefit', 'cost', 'benefit']
        
        # Example alternatives and criteria
        self.alternatives = ['A1', 'A2', 'A3', 'A4']
        self.criteria = ['C1', 'C2', 'C3', 'C4']
        
        # Create DecisionMatrix object
        self.decision_matrix = DecisionMatrix(
            decision_matrix=self.matrix,
            alternatives=self.alternatives,
            criteria=self.criteria,
            criteria_types=self.criteria_types
        )
    
    def test_aroman_initialization(self):
        """Test AROMAN initialization."""
        # Test with numpy array
        aroman = AROMAN(self.matrix, self.weights, self.criteria_types)
        self.assertEqual(aroman.matrix.shape, self.matrix.shape)
        self.assertEqual(len(aroman.weights), len(self.weights))
        self.assertEqual(len(aroman.criteria_types), len(self.criteria_types))
        
        # Test with DecisionMatrix object
        aroman = AROMAN(self.decision_matrix)
        self.assertEqual(aroman.matrix.shape, self.matrix.shape)
        self.assertEqual(len(aroman.alternatives), len(self.alternatives))
        self.assertEqual(len(aroman.criteria), len(self.criteria))
    
    def test_aroman_normalize_matrix(self):
        """Test matrix normalization."""
        aroman = AROMAN(self.matrix, self.weights, self.criteria_types)
        normalized = aroman.normalize_matrix()
        
        # Check shape
        self.assertEqual(normalized.shape, self.matrix.shape)
        
        # Check normalization properties
        for j in range(normalized.shape[1]):
            column_sum = np.sum(normalized[:, j] ** 2)
            self.assertAlmostEqual(column_sum, 1.0, places=6)
    
    def test_aroman_calculate_weighted_matrix(self):
        """Test weighted matrix calculation."""
        aroman = AROMAN(self.matrix, self.weights, self.criteria_types)
        weighted = aroman.calculate_weighted_matrix()
        
        # Check shape
        self.assertEqual(weighted.shape, self.matrix.shape)
        
        # Check if weights are applied correctly
        normalized = aroman.normalize_matrix()
        for i in range(weighted.shape[0]):
            for j in range(weighted.shape[1]):
                self.assertAlmostEqual(
                    weighted[i, j],
                    normalized[i, j] * self.weights[j],
                    places=6
                )
    
    def test_aroman_calculate_ideal_solutions(self):
        """Test ideal solutions calculation."""
        aroman = AROMAN(self.matrix, self.weights, self.criteria_types)
        ideal, anti_ideal = aroman.calculate_ideal_solutions()
        
        # Check shapes
        self.assertEqual(len(ideal), len(self.criteria))
        self.assertEqual(len(anti_ideal), len(self.criteria))
        
        # Check if ideal and anti-ideal solutions are correctly identified
        weighted = aroman.calculate_weighted_matrix()
        for j, criterion_type in enumerate(self.criteria_types):
            if criterion_type.lower() == 'benefit':
                self.assertEqual(ideal[j], np.max(weighted[:, j]))
                self.assertEqual(anti_ideal[j], np.min(weighted[:, j]))
            else:  # cost criterion
                self.assertEqual(ideal[j], np.min(weighted[:, j]))
                self.assertEqual(anti_ideal[j], np.max(weighted[:, j]))
    
    def test_aroman_calculate_scores(self):
        """Test score calculation."""
        aroman = AROMAN(self.matrix, self.weights, self.criteria_types)
        scores = aroman.calculate_scores()
        
        # Check if scores are calculated for all alternatives
        self.assertEqual(len(scores), len(self.alternatives))
        
        # Check if scores are between 0 and 1
        for score in scores.values():
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)
    
    def test_aroman_rank(self):
        """Test ranking calculation."""
        aroman = AROMAN(self.matrix, self.weights, self.criteria_types)
        rankings = aroman.rank()
        
        # Check if rankings are calculated for all alternatives
        self.assertEqual(len(rankings), len(self.alternatives))
        
        # Check if rankings are unique and sequential
        rank_values = list(rankings.values())
        self.assertEqual(sorted(rank_values), list(range(1, len(self.alternatives) + 1)))
    
    def test_aroman_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Test with invalid weights
        with self.assertRaises(ValueError):
            AROMAN(self.matrix, weights=[0.3, 0.2], criteria_types=self.criteria_types)
        
        # Test with invalid criteria types
        with self.assertRaises(ValueError):
            AROMAN(self.matrix, weights=self.weights, criteria_types=['benefit', 'cost'])
        
        # Test with invalid matrix
        with self.assertRaises(ValueError):
            AROMAN(np.array([]), self.weights, self.criteria_types)

if __name__ == '__main__':
    unittest.main() 