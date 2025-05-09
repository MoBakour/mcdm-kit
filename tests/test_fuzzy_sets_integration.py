"""
Integration tests for fuzzy sets in decision matrices.
"""

import unittest
import numpy as np
from mcdm_kit.data import DecisionMatrix
from mcdm_kit.fuzz import (
    PictureFuzzySet,
    IntervalFuzzySet,
    Type2FuzzySet,
    IntuitionisticFuzzySet,
    SphericalFuzzySet,
    NeutrosophicSet,
    PythagoreanFuzzySet,
    FermateanFuzzySet,
    HesitantFuzzySet
)

# Map fuzzy type strings to their corresponding classes
FUZZY_CLASS_MAP = {
    'PFS': PictureFuzzySet,
    'IFS': IntervalFuzzySet,
    'T2FS': Type2FuzzySet,
    'INFS': IntuitionisticFuzzySet,
    'SFS': SphericalFuzzySet,
    'NFS': NeutrosophicSet,
    'PYFS': PythagoreanFuzzySet,
    'FFS': FermateanFuzzySet,
    'HFS': HesitantFuzzySet
}

class TestFuzzySetsIntegration(unittest.TestCase):
    """Test integration of fuzzy sets with decision matrices."""
    
    def setUp(self):
        """Set up test data."""
        self.test_data = {
            'PFS': {
                'matrix': [
                    [PictureFuzzySet(0.8, 0.1, 0.1), PictureFuzzySet(0.6, 0.2, 0.2)],
                    [PictureFuzzySet(0.5, 0.3, 0.1), PictureFuzzySet(0.9, 0.05, 0.03)]
                ],
                'tuples': [
                    [(0.8, 0.1, 0.1), (0.6, 0.2, 0.2)],
                    [(0.5, 0.3, 0.1), (0.9, 0.05, 0.03)]
                ]
            },
            'IFS': {
                'matrix': [
                    [IntervalFuzzySet((0.7, 0.8), (0.1, 0.2)), IntervalFuzzySet((0.6, 0.7), (0.2, 0.3))],
                    [IntervalFuzzySet((0.5, 0.6), (0.3, 0.4)), IntervalFuzzySet((0.8, 0.9), (0.05, 0.1))]
                ],
                'tuples': [
                    [((0.7, 0.8), (0.1, 0.2)), ((0.6, 0.7), (0.2, 0.3))],
                    [((0.5, 0.6), (0.3, 0.4)), ((0.8, 0.9), (0.05, 0.1))]
                ]
            }
        }
    
    def test_fuzzy_matrix_creation(self):
        """Test creation of decision matrix with fuzzy sets."""
        for fuzzy, data in self.test_data.items():
            with self.subTest(fuzzy=fuzzy):
                # Test with pre-constructed fuzzy sets
                matrix = DecisionMatrix(
                    decision_matrix=data['matrix'],
                    alternatives=['A1', 'A2'],
                    criteria=['C1', 'C2'],
                    criteria_types=['benefit', 'benefit'],
                    fuzzy=fuzzy
                )
                
                self.assertEqual(matrix.fuzzy, fuzzy)
                self.assertIsNotNone(matrix.fuzzy_matrix)
                self.assertEqual(matrix.matrix.shape, (2, 2))
    
    def test_fuzzy_matrix_from_tuples(self):
        """Test creation of decision matrix from fuzzy set tuples."""
        for fuzzy, data in self.test_data.items():
            with self.subTest(fuzzy=fuzzy):
                # Test with tuples
                matrix = DecisionMatrix(
                    decision_matrix=data['tuples'],
                    alternatives=['A1', 'A2'],
                    criteria=['C1', 'C2'],
                    criteria_types=['benefit', 'benefit'],
                    fuzzy=FUZZY_CLASS_MAP[fuzzy]  # Pass the constructor directly
                )
                
                self.assertEqual(matrix.fuzzy, fuzzy)
                self.assertIsNotNone(matrix.fuzzy_matrix)
                self.assertEqual(matrix.matrix.shape, (2, 2))
    
    def test_fuzzy_matrix_from_constructor(self):
        """Test creation of decision matrix using fuzzy set constructor."""
        for fuzzy, data in self.test_data.items():
            with self.subTest(fuzzy=fuzzy):
                # Get the constructor class from our map
                constructor = FUZZY_CLASS_MAP[fuzzy]
                
                # Test with tuples
                matrix = DecisionMatrix(
                    decision_matrix=data['tuples'],
                    alternatives=['A1', 'A2'],
                    criteria=['C1', 'C2'],
                    criteria_types=['benefit', 'benefit'],
                    fuzzy=constructor
                )
                
                self.assertEqual(matrix.fuzzy, fuzzy)
                self.assertIsNotNone(matrix.fuzzy_matrix)
                self.assertEqual(matrix.matrix.shape, (2, 2))
    
    def test_fuzzy_matrix_validation(self):
        """Test validation of fuzzy matrices."""
        for fuzzy, data in self.test_data.items():
            with self.subTest(fuzzy=fuzzy):
                # Test with invalid fuzzy set
                invalid_matrix = [
                    [data['matrix'][0][0], "invalid"],
                    data['matrix'][1]
                ]
                
                with self.assertRaises(ValueError):
                    DecisionMatrix(
                        decision_matrix=invalid_matrix,
                        alternatives=['A1', 'A2'],
                        criteria=['C1', 'C2'],
                        criteria_types=['benefit', 'benefit'],
                        fuzzy=fuzzy
                    )
    
    def test_fuzzy_matrix_operations(self):
        """Test operations on fuzzy matrices."""
        for fuzzy, data in self.test_data.items():
            with self.subTest(fuzzy=fuzzy):
                matrix = DecisionMatrix(
                    decision_matrix=data['matrix'],
                    alternatives=['A1', 'A2'],
                    criteria=['C1', 'C2'],
                    criteria_types=['benefit', 'benefit'],
                    fuzzy=fuzzy
                )
                
                # Test fuzzy details
                details = matrix.get_fuzzy_details()
                self.assertIsNotNone(details)
                self.assertEqual(len(details), 2)  # 2 alternatives
                
                # Test fuzzy distances
                distances = matrix.get_fuzzy_distances()
                self.assertIsNotNone(distances)
                self.assertEqual(len(distances), 1)  # 1 pair of alternatives
    
    def test_invalid_fuzzy_type(self):
        """Test handling of invalid fuzzy type."""
        with self.assertRaises(ValueError):
            DecisionMatrix(
                decision_matrix=[[1, 2], [3, 4]],
                alternatives=['A1', 'A2'],
                criteria=['C1', 'C2'],
                criteria_types=['benefit', 'benefit'],
                fuzzy='INVALID'
            )
    
    def test_mixed_fuzzy_types(self):
        """Test handling of mixed fuzzy types."""
        mixed_matrix = [
            [PictureFuzzySet(0.8, 0.1, 0.1), IntervalFuzzySet((0.6, 0.7), (0.2, 0.3))],
            [PictureFuzzySet(0.5, 0.3, 0.1), PictureFuzzySet(0.9, 0.05, 0.03)]
        ]
        
        with self.assertRaises(ValueError):
            DecisionMatrix(
                decision_matrix=mixed_matrix,
                alternatives=['A1', 'A2'],
                criteria=['C1', 'C2'],
                criteria_types=['benefit', 'benefit'],
                fuzzy='PFS'
            )

if __name__ == '__main__':
    unittest.main() 