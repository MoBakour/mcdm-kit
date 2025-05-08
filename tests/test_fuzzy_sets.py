"""
Test suite for fuzzy set implementations.
"""

import unittest
import numpy as np
from mcdm_kit.fuzz import (
    BaseFuzzySet,
    PictureFuzzySet,
    IntuitionisticFuzzySet,
    SphericalFuzzySet,
    FermateanFuzzySet,
    NeutrosophicSet,
    IntervalFuzzySet,
    HesitantFuzzySet,
    PythagoreanFuzzySet,
    Type2FuzzySet
)

class TestFuzzySets(unittest.TestCase):
    def setUp(self):
        # Test values
        self.mu = 0.7
        self.eta = 0.2
        self.nu = 0.1
        
        # Create test instances
        self.picture = PictureFuzzySet(self.mu, self.eta, self.nu)
        self.intuitionistic = IntuitionisticFuzzySet(self.mu, self.nu)
        self.spherical = SphericalFuzzySet(self.mu, self.eta, self.nu)
        self.fermatean = FermateanFuzzySet(self.mu, self.nu)
        self.neutrosophic = NeutrosophicSet(self.mu, self.eta, self.nu)
        self.interval = IntervalFuzzySet([self.mu, 0.8], [self.nu, 0.2])
        self.hesitant = HesitantFuzzySet([0.6, 0.7, 0.8])
        self.pythagorean = PythagoreanFuzzySet(self.mu, self.nu)
        
        # Type-2 Fuzzy Set with simple functions
        def primary(x):
            return [0.7, 0.8]
        def secondary(x, grade):
            return 0.9
        self.type2 = Type2FuzzySet(primary, secondary)

    def test_picture_fuzzy_set(self):
        self.assertTrue(self.picture.validate())
        self.assertAlmostEqual(self.picture.score(), 0.7)
        self.assertAlmostEqual(self.picture.accuracy(), 0.2)
        
        # Test complement
        comp = self.picture.complement()
        self.assertAlmostEqual(comp.membership, 0.1)
        self.assertAlmostEqual(comp.neutrality, 0.2)
        self.assertAlmostEqual(comp.non_membership, 0.7)

    def test_intuitionistic_fuzzy_set(self):
        self.assertTrue(self.intuitionistic.validate())
        self.assertAlmostEqual(self.intuitionistic.score(), 0.6)
        self.assertAlmostEqual(self.intuitionistic.accuracy(), 0.2)
        
        # Test complement
        comp = self.intuitionistic.complement()
        self.assertAlmostEqual(comp.membership, 0.1)
        self.assertAlmostEqual(comp.non_membership, 0.7)

    def test_spherical_fuzzy_set(self):
        self.assertTrue(self.spherical.validate())
        self.assertAlmostEqual(self.spherical.score(), 0.7)
        self.assertAlmostEqual(self.spherical.accuracy(), 0.8)
        
        # Test complement
        comp = self.spherical.complement()
        self.assertAlmostEqual(comp.membership, 0.1)
        self.assertAlmostEqual(comp.neutrality, 0.2)
        self.assertAlmostEqual(comp.non_membership, 0.7)

    def test_fermatean_fuzzy_set(self):
        self.assertTrue(self.fermatean.validate())
        self.assertAlmostEqual(self.fermatean.score(), 0.6)
        self.assertAlmostEqual(self.fermatean.accuracy(), 0.8)
        
        # Test complement
        comp = self.fermatean.complement()
        self.assertAlmostEqual(comp.membership, 0.1)
        self.assertAlmostEqual(comp.non_membership, 0.7)

    def test_neutrosophic_set(self):
        self.assertTrue(self.neutrosophic.validate())
        self.assertAlmostEqual(self.neutrosophic.score(), 0.7)
        self.assertAlmostEqual(self.neutrosophic.accuracy(), 0.2)
        
        # Test complement
        comp = self.neutrosophic.complement()
        self.assertAlmostEqual(comp.truth, 0.1)
        self.assertAlmostEqual(comp.indeterminacy, 0.2)
        self.assertAlmostEqual(comp.falsity, 0.7)

    def test_interval_fuzzy_set(self):
        self.assertTrue(self.interval.validate())
        self.assertAlmostEqual(self.interval.score(), 0.75)
        self.assertAlmostEqual(self.interval.accuracy(), 0.1)
        
        # Test complement
        comp = self.interval.complement()
        self.assertAlmostEqual(comp.lower_membership[0], 0.2)  # 1 - 0.8
        self.assertAlmostEqual(comp.lower_membership[1], 0.3)  # 1 - 0.7
        self.assertAlmostEqual(comp.upper_membership[0], 0.8)  # 1 - 0.2
        self.assertAlmostEqual(comp.upper_membership[1], 0.9)  # 1 - 0.1

    def test_hesitant_fuzzy_set(self):
        self.assertTrue(self.hesitant.validate())
        self.assertAlmostEqual(self.hesitant.score(), 0.7)
        self.assertAlmostEqual(self.hesitant.accuracy(), 0.0816, places=4)
        
        # Test complement
        comp = self.hesitant.complement()
        expected = [0.2, 0.3, 0.4]
        actual = comp.membership_degrees
        self.assertEqual(len(expected), len(actual))
        for e, a in zip(expected, actual):
            self.assertAlmostEqual(e, a, places=7)

    def test_pythagorean_fuzzy_set(self):
        self.assertTrue(self.pythagorean.validate())
        self.assertAlmostEqual(self.pythagorean.score(), 0.6)
        self.assertAlmostEqual(self.pythagorean.accuracy(), 0.8)
        
        # Test complement
        comp = self.pythagorean.complement()
        self.assertAlmostEqual(comp.membership, 0.1)
        self.assertAlmostEqual(comp.non_membership, 0.7)

    def test_type2_fuzzy_set(self):
        self.assertTrue(self.type2.validate())
        self.assertAlmostEqual(self.type2.score(), 0.75)
        self.assertAlmostEqual(self.type2.accuracy(), 0.05)
        
        # Test complement
        comp = self.type2.complement()
        x = 0.5  # test point
        expected = [0.2, 0.3]
        actual = comp.primary_membership(x)
        self.assertEqual(len(expected), len(actual))
        for e, a in zip(expected, actual):
            self.assertAlmostEqual(e, a)
        self.assertAlmostEqual(comp.secondary_membership(x, 0.2), 0.9)

    def test_arithmetic_operations(self):
        # Test addition
        sum_result = self.picture + self.picture
        self.assertIsInstance(sum_result, PictureFuzzySet)
        
        # Test multiplication
        prod_result = self.picture * 2
        self.assertIsInstance(prod_result, PictureFuzzySet)
        
        # Test equality
        self.assertTrue(self.picture == self.picture)
        # Compare with a different instance of the same type
        other_picture = PictureFuzzySet(0.8, 0.1, 0.1)
        self.assertFalse(self.picture == other_picture)

    def test_serialization(self):
        # Test to_dict and from_dict
        for fuzzy_set in [self.picture, self.intuitionistic, self.spherical,
                         self.fermatean, self.neutrosophic, self.interval,
                         self.hesitant, self.pythagorean, self.type2]:
            data = fuzzy_set.to_dict()
            reconstructed = fuzzy_set.__class__.from_dict(data)
            self.assertEqual(fuzzy_set, reconstructed)

if __name__ == '__main__':
    unittest.main() 