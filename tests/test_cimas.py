"""
Tests for CIMAS implementation.
"""

import numpy as np
import pytest
from mcdm_kit.core.cimas import CIMAS
from mcdm_kit.data.decision_matrix import DecisionMatrix

def test_cimas_initialization():
    """Test CIMAS initialization."""
    # Create sample decision matrix
    matrix = np.array([
        [4, 3, 5, 2],
        [3, 4, 2, 5],
        [5, 3, 4, 3]
    ])
    criteria_types = ['benefit', 'benefit', 'cost', 'cost']
    alternatives = ['A1', 'A2', 'A3']
    criteria = ['C1', 'C2', 'C3', 'C4']
    
    decision_matrix = DecisionMatrix(
        decision_matrix=matrix,
        alternatives=alternatives,
        criteria=criteria,
        criteria_types=criteria_types
    )
    
    # Test initialization with default parameters
    cimas = CIMAS(decision_matrix)
    assert cimas.decision_matrix is decision_matrix
    assert cimas.weights is None
    assert cimas.normalization_method == 'minmax'
    
    # Test initialization with custom weights
    weights = np.array([0.3, 0.2, 0.3, 0.2])
    cimas = CIMAS(decision_matrix, weights=weights)
    assert np.array_equal(cimas.weights, weights)
    
    # Test initialization with custom normalization method
    cimas = CIMAS(decision_matrix, normalization_method='vector')
    assert cimas.normalization_method == 'vector'

def test_cimas_calculate_weights():
    """Test weight calculation in CIMAS."""
    matrix = np.array([
        [4, 3, 5, 2],
        [3, 4, 2, 5],
        [5, 3, 4, 3]
    ])
    decision_matrix = DecisionMatrix(
        decision_matrix=matrix,
        criteria_types=['benefit', 'benefit', 'cost', 'cost']
    )
    
    # Test with default weights
    cimas = CIMAS(decision_matrix)
    weights = cimas.calculate_weights()
    assert len(weights) == 4
    assert np.isclose(np.sum(weights), 1.0)
    assert np.all(weights > 0)
    
    # Test with custom weights
    custom_weights = np.array([0.3, 0.2, 0.3, 0.2])
    cimas = CIMAS(decision_matrix, weights=custom_weights)
    weights = cimas.calculate_weights()
    assert np.array_equal(weights, custom_weights)

def test_cimas_normalize_matrix():
    """Test matrix normalization in CIMAS."""
    matrix = np.array([
        [4, 3, 5, 2],
        [3, 4, 2, 5],
        [5, 3, 4, 3]
    ])
    decision_matrix = DecisionMatrix(
        decision_matrix=matrix,
        criteria_types=['benefit', 'benefit', 'cost', 'cost']
    )
    
    cimas = CIMAS(decision_matrix)
    normalized = cimas.normalize_matrix()
    assert normalized.shape == matrix.shape
    assert np.all(normalized >= 0)
    assert np.all(normalized <= 1)

def test_cimas_calculate_weighted_matrix():
    """Test weighted matrix calculation in CIMAS."""
    matrix = np.array([
        [4, 3, 5, 2],
        [3, 4, 2, 5],
        [5, 3, 4, 3]
    ])
    decision_matrix = DecisionMatrix(
        decision_matrix=matrix,
        criteria_types=['benefit', 'benefit', 'cost', 'cost']
    )
    
    cimas = CIMAS(decision_matrix)
    weighted = cimas.calculate_weighted_matrix()
    assert weighted.shape == matrix.shape
    assert np.all(weighted >= 0)

def test_cimas_calculate_scores():
    """Test score calculation in CIMAS."""
    matrix = np.array([
        [4, 3, 5, 2],
        [3, 4, 2, 5],
        [5, 3, 4, 3]
    ])
    decision_matrix = DecisionMatrix(
        decision_matrix=matrix,
        criteria_types=['benefit', 'benefit', 'cost', 'cost']
    )
    
    cimas = CIMAS(decision_matrix)
    scores = cimas.calculate_scores()
    assert len(scores) == 3

def test_cimas_rank():
    """Test ranking in CIMAS."""
    matrix = np.array([
        [4, 3, 5, 2],
        [3, 4, 2, 5],
        [5, 3, 4, 3]
    ])
    alternatives = ['A1', 'A2', 'A3']
    criteria = ['C1', 'C2', 'C3', 'C4']
    decision_matrix = DecisionMatrix(
        decision_matrix=matrix,
        alternatives=alternatives,
        criteria=criteria,
        criteria_types=['benefit', 'benefit', 'cost', 'cost']
    )
    
    cimas = CIMAS(decision_matrix)
    results = cimas.rank()
    
    assert 'rankings' in results
    assert 'scores' in results
    assert 'weighted_matrix' in results
    assert 'normalized_matrix' in results
    
    rankings = results['rankings']
    assert len(rankings) == 3
    assert all('rank' in r for r in rankings)
    assert all('alternative' in r for r in rankings)
    assert all('score' in r for r in rankings)
    
    # Check that rankings are in descending order of scores
    scores = [r['score'] for r in rankings]
    assert scores == sorted(scores, reverse=True)

def test_cimas_invalid_inputs():
    """Test CIMAS with invalid inputs."""
    # Test with empty matrix
    with pytest.raises(ValueError):
        decision_matrix = DecisionMatrix(
            decision_matrix=np.array([]),
            criteria_types=['benefit']
        )
        CIMAS(decision_matrix)
    
    # Test with invalid weights
    matrix = np.array([[1, 2], [3, 4]])
    decision_matrix = DecisionMatrix(
        decision_matrix=matrix,
        criteria_types=['benefit', 'benefit']
    )
    with pytest.raises(ValueError):
        CIMAS(decision_matrix, weights=np.array([0.5, 0.5, 0.5]))
    
    # Test with invalid normalization method
    with pytest.raises(ValueError):
        CIMAS(decision_matrix, normalization_method='invalid') 