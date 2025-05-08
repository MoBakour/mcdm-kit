"""
Tests for ARLON implementation.
"""

import numpy as np
import pytest
from mcdm_kit.core.arlon import ARLON
from mcdm_kit.data.decision_matrix import DecisionMatrix

def test_arlon_initialization():
    """Test ARLON initialization with default and custom parameters."""
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
    arlon = ARLON(decision_matrix)
    assert arlon.decision_matrix == decision_matrix
    assert arlon.weights is None
    assert arlon.normalization_method == 'vector'
    assert arlon.levels == 5
    
    # Test with custom parameters
    weights = np.array([0.3, 0.3, 0.4])
    arlon = ARLON(
        decision_matrix,
        weights=weights,
        normalization_method='minmax',
        levels=3
    )
    assert np.array_equal(arlon.weights, weights)
    assert arlon.normalization_method == 'minmax'
    assert arlon.levels == 3

def test_arlon_calculate_weights():
    """Test weight calculation functionality."""
    matrix = np.array([[1, 2], [3, 4]])
    decision_matrix = DecisionMatrix(
        matrix,
        ['A1', 'A2'],
        ['C1', 'C2'],
        ['benefit', 'cost']
    )
    
    # Test with no weights provided
    arlon = ARLON(decision_matrix)
    weights = arlon.calculate_weights()
    assert len(weights) == 2
    assert np.isclose(np.sum(weights), 1.0)
    assert np.all(weights > 0)
    
    # Test with custom weights
    custom_weights = np.array([0.7, 0.3])
    arlon = ARLON(decision_matrix, weights=custom_weights)
    weights = arlon.calculate_weights()
    assert np.array_equal(weights, custom_weights)
    
    # Test with invalid weights
    with pytest.raises(ValueError):
        arlon = ARLON(decision_matrix, weights=np.array([0.5]))
        arlon.calculate_weights()

def test_arlon_normalize_matrix():
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
    
    arlon = ARLON(decision_matrix)
    normalized = arlon.normalize_matrix()
    
    assert normalized.shape == matrix.shape
    assert np.all(normalized >= 0)
    assert np.all(normalized <= 1)

def test_arlon_calculate_ordinal_matrix():
    """Test ordinal matrix calculation."""
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
    
    arlon = ARLON(decision_matrix, levels=3)
    ordinal = arlon.calculate_ordinal_matrix()
    
    assert ordinal.shape == matrix.shape
    assert np.all(ordinal >= 0)
    assert np.all(ordinal < arlon.levels)

def test_arlon_calculate_weighted_matrix():
    """Test weighted matrix calculation."""
    matrix = np.array([[1, 2], [3, 4]])
    decision_matrix = DecisionMatrix(
        matrix,
        ['A1', 'A2'],
        ['C1', 'C2'],
        ['benefit', 'cost']
    )
    
    weights = np.array([0.6, 0.4])
    arlon = ARLON(decision_matrix, weights=weights)
    weighted = arlon.calculate_weighted_matrix()
    
    assert weighted.shape == matrix.shape
    assert np.all(weighted >= 0)

def test_arlon_calculate_scores():
    """Test score calculation."""
    matrix = np.array([[1, 2], [3, 4]])
    decision_matrix = DecisionMatrix(
        matrix,
        ['A1', 'A2'],
        ['C1', 'C2'],
        ['benefit', 'cost']
    )
    
    arlon = ARLON(decision_matrix)
    scores = arlon.calculate_scores()
    
    assert len(scores) == len(decision_matrix.alternatives)
    assert np.all(scores >= 0)

def test_arlon_rank():
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
    arlon = ARLON(decision_matrix)
    results = arlon.rank()
    
    assert 'rankings' in results
    assert 'scores' in results
    assert 'ordinal_matrix' in results
    assert 'weighted_matrix' in results
    
    rankings = results['rankings']
    assert len(rankings) == len(alternatives)
    assert all('rank' in r for r in rankings)
    assert all('alternative' in r for r in rankings)
    assert all('score' in r for r in rankings)
    
    scores = results['scores']
    assert len(scores) == len(alternatives)
    assert all(alt in scores for alt in alternatives)

def test_arlon_invalid_inputs():
    """Test handling of invalid inputs."""
    # Test with empty matrix
    with pytest.raises(ValueError):
        matrix = np.array([])
        decision_matrix = DecisionMatrix(matrix, [], [], [])
        ARLON(decision_matrix)
    
    # Test with invalid weights
    matrix = np.array([[1, 2], [3, 4]])
    decision_matrix = DecisionMatrix(
        matrix,
        ['A1', 'A2'],
        ['C1', 'C2'],
        ['benefit', 'cost']
    )
    
    with pytest.raises(ValueError):
        ARLON(decision_matrix, weights=np.array([-0.5, 0.5]))
    
    # Test with invalid normalization method
    with pytest.raises(ValueError):
        ARLON(decision_matrix, normalization_method='invalid')
    
    # Test with invalid levels
    with pytest.raises(ValueError):
        ARLON(decision_matrix, levels=0) 