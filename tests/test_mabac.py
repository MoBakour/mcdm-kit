"""
Tests for MABAC implementation.
"""

import numpy as np
import pytest
from mcdm_kit.core.mabac import MABAC
from mcdm_kit.data.decision_matrix import DecisionMatrix

def test_mabac_initialization():
    """Test MABAC initialization."""
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
    mabac = MABAC(decision_matrix)
    assert mabac.decision_matrix is decision_matrix
    assert mabac.weights is None
    assert mabac.normalization_method == 'vector'
    
    # Test initialization with custom weights
    weights = np.array([0.3, 0.2, 0.3, 0.2])
    mabac = MABAC(decision_matrix, weights=weights)
    assert np.array_equal(mabac.weights, weights)
    
    # Test initialization with custom normalization method
    mabac = MABAC(decision_matrix, normalization_method='minmax')
    assert mabac.normalization_method == 'minmax'

def test_mabac_calculate_weights():
    """Test weight calculation in MABAC."""
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
    mabac = MABAC(decision_matrix)
    weights = mabac.calculate_weights()
    assert len(weights) == 4
    assert np.isclose(np.sum(weights), 1.0)
    assert np.all(weights > 0)
    
    # Test with custom weights
    custom_weights = np.array([0.3, 0.2, 0.3, 0.2])
    mabac = MABAC(decision_matrix, weights=custom_weights)
    weights = mabac.calculate_weights()
    assert np.array_equal(weights, custom_weights)

def test_mabac_normalize_matrix():
    """Test matrix normalization in MABAC."""
    matrix = np.array([
        [4, 3, 5, 2],
        [3, 4, 2, 5],
        [5, 3, 4, 3]
    ])
    decision_matrix = DecisionMatrix(
        decision_matrix=matrix,
        criteria_types=['benefit', 'benefit', 'cost', 'cost']
    )
    
    mabac = MABAC(decision_matrix)
    normalized = mabac.normalize_matrix()
    assert normalized.shape == matrix.shape
    assert np.all(normalized >= 0)
    assert np.all(normalized <= 1)

def test_mabac_calculate_weighted_matrix():
    """Test weighted matrix calculation in MABAC."""
    matrix = np.array([
        [4, 3, 5, 2],
        [3, 4, 2, 5],
        [5, 3, 4, 3]
    ])
    decision_matrix = DecisionMatrix(
        decision_matrix=matrix,
        criteria_types=['benefit', 'benefit', 'cost', 'cost']
    )
    
    mabac = MABAC(decision_matrix)
    weighted = mabac.calculate_weighted_matrix()
    assert weighted.shape == matrix.shape
    assert np.all(weighted >= 0)

def test_mabac_calculate_border_matrix():
    """Test border matrix calculation in MABAC."""
    matrix = np.array([
        [4, 3, 5, 2],
        [3, 4, 2, 5],
        [5, 3, 4, 3]
    ])
    decision_matrix = DecisionMatrix(
        decision_matrix=matrix,
        criteria_types=['benefit', 'benefit', 'cost', 'cost']
    )
    
    mabac = MABAC(decision_matrix)
    border = mabac.calculate_border_matrix()
    assert len(border) == 4
    assert np.all(border >= 0)

def test_mabac_calculate_distance_matrix():
    """Test distance matrix calculation in MABAC."""
    matrix = np.array([
        [4, 3, 5, 2],
        [3, 4, 2, 5],
        [5, 3, 4, 3]
    ])
    decision_matrix = DecisionMatrix(
        decision_matrix=matrix,
        criteria_types=['benefit', 'benefit', 'cost', 'cost']
    )
    
    mabac = MABAC(decision_matrix)
    distance = mabac.calculate_distance_matrix()
    assert distance.shape == matrix.shape

def test_mabac_calculate_scores():
    """Test score calculation in MABAC."""
    matrix = np.array([
        [4, 3, 5, 2],
        [3, 4, 2, 5],
        [5, 3, 4, 3]
    ])
    decision_matrix = DecisionMatrix(
        decision_matrix=matrix,
        criteria_types=['benefit', 'benefit', 'cost', 'cost']
    )
    
    mabac = MABAC(decision_matrix)
    scores = mabac.calculate_scores()
    assert len(scores) == 3

def test_mabac_rank():
    """Test ranking in MABAC."""
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
    
    mabac = MABAC(decision_matrix)
    results = mabac.rank()
    
    assert 'rankings' in results
    assert 'scores' in results
    assert 'border_matrix' in results
    assert 'distance_matrix' in results
    
    rankings = results['rankings']
    assert len(rankings) == 3
    assert all('rank' in r for r in rankings)
    assert all('alternative' in r for r in rankings)
    assert all('score' in r for r in rankings)
    
    # Check that rankings are in descending order of scores
    scores = [r['score'] for r in rankings]
    assert scores == sorted(scores, reverse=True)

def test_mabac_invalid_inputs():
    """Test MABAC with invalid inputs."""
    # Test with empty matrix
    with pytest.raises(ValueError):
        decision_matrix = DecisionMatrix(
            decision_matrix=np.array([]),
            criteria_types=['benefit']
        )
        MABAC(decision_matrix)
    
    # Test with invalid weights
    matrix = np.array([[1, 2], [3, 4]])
    decision_matrix = DecisionMatrix(
        decision_matrix=matrix,
        criteria_types=['benefit', 'benefit']
    )
    with pytest.raises(ValueError):
        MABAC(decision_matrix, weights=np.array([0.5, 0.5, 0.5]))
    
    # Test with invalid normalization method
    with pytest.raises(ValueError):
        MABAC(decision_matrix, normalization_method='invalid') 