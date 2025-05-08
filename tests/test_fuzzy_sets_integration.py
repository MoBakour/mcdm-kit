"""
Integration tests for fuzzy sets with MCDM methods.
"""

import unittest
import numpy as np
from typing import List
from mcdm_kit.data import DecisionMatrix
from mcdm_kit.core import CIMAS, ARTASI
from mcdm_kit.fuzz.picture import PictureFuzzySet
from mcdm_kit.fuzz.interval import IntervalFuzzySet
from mcdm_kit.fuzz.type2 import Type2FuzzySet
from mcdm_kit.fuzz.intuitionistic import IntuitionisticFuzzySet
from mcdm_kit.fuzz.spherical import SphericalFuzzySet
from mcdm_kit.fuzz.neutrosophic import NeutrosophicSet
from mcdm_kit.fuzz.pythagorean import PythagoreanFuzzySet
from mcdm_kit.fuzz.fermatean import FermateanFuzzySet
from mcdm_kit.fuzz.hesitant import HesitantFuzzySet

class TestFuzzySetsIntegration(unittest.TestCase):
    """Test integration of fuzzy sets with MCDM methods."""
    
    def setUp(self):
        """Set up test data."""
        # Helper functions for Type-2 Fuzzy Sets
        def make_t2fs(center: float, spread: float = 0.1):
            def primary_membership(x: float) -> List[float]:
                return [max(0, min(1, center + spread * i)) for i in [-1, 0, 1]]
            def secondary_membership(x: float, grade: float) -> float:
                return 1.0 if abs(grade - center) <= spread else 0.0
            return Type2FuzzySet(primary_membership, secondary_membership)
        
        # Test data for each fuzzy set type
        self.test_data = {
            'PFS': [
                [PictureFuzzySet(0.8, 0.1, 0.1), PictureFuzzySet(0.6, 0.2, 0.2)],
                [PictureFuzzySet(0.5, 0.3, 0.1), PictureFuzzySet(0.9, 0.05, 0.03)]
            ],
            'IFS': [
                [IntervalFuzzySet((0.6, 0.8), (0.1, 0.3)), IntervalFuzzySet((0.5, 0.7), (0.2, 0.4))],
                [IntervalFuzzySet((0.4, 0.6), (0.3, 0.5)), IntervalFuzzySet((0.7, 0.9), (0.0, 0.2))]
            ],
            'T2FS': [
                [make_t2fs(0.8, 0.1), make_t2fs(0.6, 0.2)],
                [make_t2fs(0.5, 0.3), make_t2fs(0.9, 0.05)]
            ],
            'INFS': [
                [IntuitionisticFuzzySet(0.7, 0.2), IntuitionisticFuzzySet(0.6, 0.3)],
                [IntuitionisticFuzzySet(0.5, 0.4), IntuitionisticFuzzySet(0.8, 0.1)]
            ],
            'SFS': [
                [SphericalFuzzySet(0.8, 0.1, 0.1), SphericalFuzzySet(0.6, 0.2, 0.2)],
                [SphericalFuzzySet(0.5, 0.3, 0.1), SphericalFuzzySet(0.9, 0.05, 0.03)]
            ],
            'NFS': [
                [NeutrosophicSet(0.8, 0.1, 0.1), NeutrosophicSet(0.6, 0.2, 0.2)],
                [NeutrosophicSet(0.5, 0.3, 0.1), NeutrosophicSet(0.9, 0.05, 0.03)]
            ],
            'PYFS': [
                [PythagoreanFuzzySet(0.8, 0.1), PythagoreanFuzzySet(0.6, 0.2)],
                [PythagoreanFuzzySet(0.5, 0.3), PythagoreanFuzzySet(0.9, 0.05)]
            ],
            'FFS': [
                [FermateanFuzzySet(0.8, 0.1), FermateanFuzzySet(0.6, 0.2)],
                [FermateanFuzzySet(0.5, 0.3), FermateanFuzzySet(0.9, 0.05)]
            ],
            'HFS': [
                [HesitantFuzzySet([0.8, 0.7]), HesitantFuzzySet([0.6, 0.5])],
                [HesitantFuzzySet([0.5, 0.4]), HesitantFuzzySet([0.9, 0.8])]
            ]
        }
        
        # Common test parameters
        self.alternatives = ["Alt1", "Alt2"]
        self.criteria = ["Criterion1", "Criterion2"]
        self.criteria_types = ["benefit", "benefit"]
    
    def test_matrix_creation(self):
        """Test creation of DecisionMatrix with different fuzzy set types."""
        for fuzzy_type, data in self.test_data.items():
            with self.subTest(fuzzy_type=fuzzy_type):
                matrix = DecisionMatrix(
                    data,
                    alternatives=self.alternatives,
                    criteria=self.criteria,
                    criteria_types=self.criteria_types,
                    fuzzy_type=fuzzy_type
                )
                
                # Check matrix dimensions
                self.assertEqual(matrix.matrix.shape, (2, 2))
                
                # Check metadata
                self.assertEqual(matrix.alternatives, self.alternatives)
                self.assertEqual(matrix.criteria, self.criteria)
                self.assertEqual(matrix.criteria_types, self.criteria_types)
                
                # Check fuzzy type
                self.assertEqual(matrix.fuzzy_type, fuzzy_type)
                
                # Check numerical conversion
                self.assertTrue(np.all(matrix.matrix >= 0))
                self.assertTrue(np.all(matrix.matrix <= 1))
    
    def test_fuzzy_details(self):
        """Test retrieval of fuzzy details."""
        for fuzzy_type, data in self.test_data.items():
            with self.subTest(fuzzy_type=fuzzy_type):
                matrix = DecisionMatrix(
                    data,
                    alternatives=self.alternatives,
                    criteria=self.criteria,
                    criteria_types=self.criteria_types,
                    fuzzy_type=fuzzy_type
                )
                
                details = matrix.get_fuzzy_details()
                
                # Check structure
                self.assertIsNotNone(details)
                self.assertIn(self.alternatives[0], details)
                self.assertIn(self.criteria[0], details[self.alternatives[0]])
                
                # Check values
                for alt in self.alternatives:
                    for crit in self.criteria:
                        self.assertIsInstance(details[alt][crit], dict)
    
    def test_fuzzy_distances(self):
        """Test calculation of fuzzy distances."""
        for fuzzy_type, data in self.test_data.items():
            with self.subTest(fuzzy_type=fuzzy_type):
                matrix = DecisionMatrix(
                    data,
                    alternatives=self.alternatives,
                    criteria=self.criteria,
                    criteria_types=self.criteria_types,
                    fuzzy_type=fuzzy_type
                )
                
                distances = matrix.get_fuzzy_distances()
                
                # Check structure
                self.assertIsNotNone(distances)
                self.assertIn(f"{self.alternatives[0]} vs {self.alternatives[1]}", distances)
                
                # Check values
                self.assertTrue(all(0 <= d <= 1 for d in distances.values()))
    
    def test_cimas_integration(self):
        """Test integration with CIMAS method."""
        for fuzzy_type, data in self.test_data.items():
            with self.subTest(fuzzy_type=fuzzy_type):
                matrix = DecisionMatrix(
                    data,
                    alternatives=self.alternatives,
                    criteria=self.criteria,
                    criteria_types=self.criteria_types,
                    fuzzy_type=fuzzy_type
                )
                
                cimas = CIMAS(decision_matrix=matrix)
                weights = cimas.calculate_weights()
                
                # Check weights
                self.assertEqual(len(weights), len(self.criteria))
                self.assertAlmostEqual(sum(weights), 1.0)
                self.assertTrue(all(0 <= w <= 1 for w in weights))
    
    def test_artasi_integration(self):
        """Test integration with ARTASI method."""
        for fuzzy_type, data in self.test_data.items():
            with self.subTest(fuzzy_type=fuzzy_type):
                matrix = DecisionMatrix(
                    data,
                    alternatives=self.alternatives,
                    criteria=self.criteria,
                    criteria_types=self.criteria_types,
                    fuzzy_type=fuzzy_type
                )
                
                # Calculate weights using CIMAS
                cimas = CIMAS(decision_matrix=matrix)
                weights = cimas.calculate_weights()
                
                # Use weights with ARTASI
                artasi = ARTASI(decision_matrix=matrix, weights=weights)
                rankings = artasi.rank()
                
                # Check rankings
                self.assertIsNotNone(rankings)
                self.assertEqual(len(rankings['rankings']), len(self.alternatives))
                self.assertTrue(all('rank' in r for r in rankings['rankings']))
                self.assertTrue(all('score' in r for r in rankings['rankings']))
                self.assertTrue(all('alternative' in r for r in rankings['rankings']))
                self.assertEqual(len(rankings['scores']), len(self.alternatives))
                self.assertEqual(len(rankings['aspiration_levels']), len(self.criteria))
                self.assertEqual(len(rankings['distance_matrix']), len(self.alternatives))
    
    def test_invalid_fuzzy_type(self):
        """Test handling of invalid fuzzy type."""
        with self.assertRaises(ValueError):
            DecisionMatrix(
                self.test_data['PFS'],
                alternatives=self.alternatives,
                criteria=self.criteria,
                criteria_types=self.criteria_types,
                fuzzy_type='INVALID'
            )
    
    def test_mixed_fuzzy_types(self):
        """Test handling of mixed fuzzy types."""
        mixed_data = [
            [PictureFuzzySet(0.8, 0.1, 0.1), IntuitionisticFuzzySet(0.6, 0.3)],
            [SphericalFuzzySet(0.5, 0.3, 0.1), PythagoreanFuzzySet(0.9, 0.05)]
        ]
        
        with self.assertRaises(ValueError):
            DecisionMatrix(
                mixed_data,
                alternatives=self.alternatives,
                criteria=self.criteria,
                criteria_types=self.criteria_types,
                fuzzy_type='PFS'
            )

if __name__ == '__main__':
    unittest.main() 