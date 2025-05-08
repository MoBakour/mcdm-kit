"""
Tests for WENSLO implementation.
"""

import numpy as np
import pytest
from mcdm_kit.core.wenslo import WENSLO
from mcdm_kit.data.decision_matrix import DecisionMatrix

def test_wenslo_initialization():
    """Test WENSLO initialization with default and custom parameters."""
    # Create a sample decision matrix
    matrix = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    alternatives = ['A1', 'A2', 'A3']
    criteria = ['C1', 'C2', 'C3']
    criteria_types = ['benefit', 'cost', 'benefit']
    
    decision_matrix = DecisionMatrix(matrix, alternatives, criteria, criteria_types)
    
    # Test with default parameters
    wenslo = WENSLO(decision_matrix)
    assert wenslo.decision_matrix == decision_matrix
    assert wenslo.weights is None
    assert wenslo.normalization_method == 'vector'
    assert wenslo.standard_levels is None
    
    # Test with custom parameters
    weights = np.array([0.3, 0.3, 0.4])
    standard_levels = np.array([4, 5, 6])
    wenslo = WENSLO(
        decision_matrix,
        weights=weights,
        normalization_method='minmax',
        standard_levels=standard_levels
    )
    assert np.array_equal(wenslo.weights, weights)
    assert wenslo.normalization_method == 'minmax'
    assert np.array_equal(wenslo.standard_levels, standard_levels)

def test_wenslo_calculate_weights():
    """Test weight calculation functionality."""
    matrix = np.array([[1, 2], [3, 4]])
    decision_matrix = DecisionMatrix(
        matrix,
        ['A1', 'A2'],
        ['C1', 'C2'],
        ['benefit', 'cost']
    )
    
    # Test with no weights provided
    wenslo = WENSLO(decision_matrix)
    weights = wenslo.calculate_weights()
    assert len(weights) == 2
    assert np.isclose(np.sum(weights), 1.0)
    assert np.all(weights > 0)
    
    # Test with custom weights
    custom_weights = np.array([0.7, 0.3])
    wenslo = WENSLO(decision_matrix, weights=custom_weights)
    weights = wenslo.calculate_weights()
    assert np.array_equal(weights, custom_weights)
    
    # Test with invalid weights
    with pytest.raises(ValueError):
        wenslo = WENSLO(decision_matrix, weights=np.array([0.5]))
        wenslo.calculate_weights()

def test_wenslo_calculate_standard_levels():
    """Test standard level calculation."""
    matrix = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    decision_matrix = DecisionMatrix(
        matrix,
        ['A1', 'A2', 'A3'],
        ['C1', 'C2', 'C3'],
        ['benefit', 'cost', 'benefit']
    )
    
    # Test automatic calculation
    wenslo = WENSLO(decision_matrix)
    standard_levels = wenslo.calculate_standard_levels()
    assert len(standard_levels) == 3
    assert np.isclose(standard_levels[0], 4)  # mean of [1, 4, 7]
    assert np.isclose(standard_levels[1], 5)  # mean of [2, 5, 8]
    assert np.isclose(standard_levels[2], 6)  # mean of [3, 6, 9]
    
    # Test with custom standard levels
    custom_levels = np.array([4, 5, 6])
    wenslo = WENSLO(decision_matrix, standard_levels=custom_levels)
    levels = wenslo.calculate_standard_levels()
    assert np.array_equal(levels, custom_levels)
    
    # Test with invalid standard levels
    with pytest.raises(ValueError):
        wenslo = WENSLO(decision_matrix, standard_levels=np.array([1, 2]))
        wenslo.calculate_standard_levels()

def test_wenslo_normalize_matrix():
    """Test matrix normalization."""
    matrix = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    decision_matrix = DecisionMatrix(
        matrix,
        ['A1', 'A2', 'A3'],
        ['C1', 'C2', 'C3'],
        ['benefit', 'cost', 'benefit']
    )
    
    wenslo = WENSLO(decision_matrix)
    normalized = wenslo.normalize_matrix()
    
    assert normalized.shape == matrix.shape
    assert np.all(normalized >= 0)
    assert np.all(normalized <= 1)

def test_wenslo_calculate_weighted_matrix():
    """Test weighted matrix calculation."""
    matrix = np.array([[1, 2], [3, 4]])
    decision_matrix = DecisionMatrix(
        matrix,
        ['A1', 'A2'],
        ['C1', 'C2'],
        ['benefit', 'cost']
    )
    
    weights = np.array([0.6, 0.4])
    wenslo = WENSLO(decision_matrix, weights=weights)
    weighted = wenslo.calculate_weighted_matrix()
    
    assert weighted.shape == matrix.shape
    assert np.all(weighted >= 0)

def test_wenslo_calculate_standard_matrix():
    """Test standard matrix calculation."""
    matrix = np.array([[1, 2], [3, 4]])
    decision_matrix = DecisionMatrix(
        matrix,
        ['A1', 'A2'],
        ['C1', 'C2'],
        ['benefit', 'cost']
    )
    
    weights = np.array([0.6, 0.4])
    wenslo = WENSLO(decision_matrix, weights=weights)
    standard = wenslo.calculate_standard_matrix()
    
    assert standard.shape == matrix.shape
    assert np.all(standard >= 0)
    assert np.all(standard <= 1)

def test_wenslo_calculate_distance_matrix():
    """Test distance matrix calculation."""
    matrix = np.array([[1, 2], [3, 4]])
    decision_matrix = DecisionMatrix(
        matrix,
        ['A1', 'A2'],
        ['C1', 'C2'],
        ['benefit', 'cost']
    )
    
    wenslo = WENSLO(decision_matrix)
    distance = wenslo.calculate_distance_matrix()
    
    assert distance.shape == matrix.shape
    assert np.all(distance >= 0)

def test_wenslo_calculate_scores():
    """Test score calculation."""
    matrix = np.array([[1, 2], [3, 4]])
    decision_matrix = DecisionMatrix(
        matrix,
        ['A1', 'A2'],
        ['C1', 'C2'],
        ['benefit', 'cost']
    )
    
    wenslo = WENSLO(decision_matrix)
    scores = wenslo.calculate_scores()
    
    assert len(scores) == len(decision_matrix.alternatives)
    assert np.all(scores >= 0)
    assert np.all(scores <= 1)

def test_wenslo_rank():
    """Test ranking functionality."""
    matrix = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    alternatives = ['A1', 'A2', 'A3']
    criteria = ['C1', 'C2', 'C3']
    criteria_types = ['benefit', 'cost', 'benefit']
    
    decision_matrix = DecisionMatrix(matrix, alternatives, criteria, criteria_types)
    wenslo = WENSLO(decision_matrix)
    results = wenslo.rank()
    
    assert 'rankings' in results
    assert 'scores' in results
    assert 'standard_levels' in results
    assert 'distance_matrix' in results
    
    rankings = results['rankings']
    assert len(rankings) == len(alternatives)
    assert all('rank' in r for r in rankings)
    assert all('alternative' in r for r in rankings)
    assert all('score' in r for r in rankings)
    
    scores = results['scores']
    assert len(scores) == len(alternatives)
    assert all(alt in scores for alt in alternatives)
    
    standard_levels = results['standard_levels']
    assert len(standard_levels) == len(criteria)
    assert all(crit in standard_levels for crit in criteria)

def test_wenslo_invalid_inputs():
    """Test handling of invalid inputs."""
    # Test with empty matrix
    with pytest.raises(ValueError):
        matrix = np.array([])
        decision_matrix = DecisionMatrix(matrix, [], [], [])
        WENSLO(decision_matrix)
    
    # Test with invalid weights
    matrix = np.array([[1, 2], [3, 4]])
    decision_matrix = DecisionMatrix(
        matrix,
        ['A1', 'A2'],
        ['C1', 'C2'],
        ['benefit', 'cost']
    )
    
    with pytest.raises(ValueError):
        WENSLO(decision_matrix, weights=np.array([-0.5, 0.5]))
    
    # Test with invalid normalization method
    with pytest.raises(ValueError):
        WENSLO(decision_matrix, normalization_method='invalid')
    
    # Test with invalid standard levels
    with pytest.raises(ValueError):
        WENSLO(decision_matrix, standard_levels=np.array([-1, 2])) 