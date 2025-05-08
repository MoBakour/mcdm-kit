"""
Tests for TOPSIS implementation.
"""

import numpy as np
import pytest
from mcdm_kit.core.topsis import TOPSIS
from mcdm_kit.data.decision_matrix import DecisionMatrix

def test_topsis_initialization():
    """Test TOPSIS initialization."""
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
    topsis = TOPSIS(decision_matrix)
    assert topsis.decision_matrix is decision_matrix
    assert topsis.weights is None
    assert topsis.normalization_method == 'vector'
    
    # Test initialization with custom weights
    weights = np.array([0.3, 0.2, 0.3, 0.2])
    topsis = TOPSIS(decision_matrix, weights=weights)
    assert np.array_equal(topsis.weights, weights)
    
    # Test initialization with custom normalization method
    topsis = TOPSIS(decision_matrix, normalization_method='minmax')
    assert topsis.normalization_method == 'minmax'

def test_topsis_calculate_weights():
    """Test weight calculation in TOPSIS."""
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
    topsis = TOPSIS(decision_matrix)
    weights = topsis.calculate_weights()
    assert len(weights) == 4
    assert np.isclose(np.sum(weights), 1.0)
    assert np.all(weights > 0)
    
    # Test with custom weights
    custom_weights = np.array([0.3, 0.2, 0.3, 0.2])
    topsis = TOPSIS(decision_matrix, weights=custom_weights)
    weights = topsis.calculate_weights()
    assert np.array_equal(weights, custom_weights)

def test_topsis_normalize_matrix():
    """Test matrix normalization in TOPSIS."""
    matrix = np.array([
        [4, 3, 5, 2],
        [3, 4, 2, 5],
        [5, 3, 4, 3]
    ])
    decision_matrix = DecisionMatrix(
        decision_matrix=matrix,
        criteria_types=['benefit', 'benefit', 'cost', 'cost']
    )
    
    # Test vector normalization
    topsis = TOPSIS(decision_matrix, normalization_method='vector')
    normalized = topsis.normalize_matrix()
    assert normalized.shape == matrix.shape
    assert np.all(normalized >= 0)
    
    # Test minmax normalization
    topsis = TOPSIS(decision_matrix, normalization_method='minmax')
    normalized = topsis.normalize_matrix()
    assert normalized.shape == matrix.shape
    assert np.all(normalized >= 0)
    assert np.all(normalized <= 1)

def test_topsis_calculate_ideal_solutions():
    """Test ideal solution calculation in TOPSIS."""
    matrix = np.array([
        [4, 3, 5, 2],
        [3, 4, 2, 5],
        [5, 3, 4, 3]
    ])
    decision_matrix = DecisionMatrix(
        decision_matrix=matrix,
        criteria_types=['benefit', 'benefit', 'cost', 'cost']
    )
    
    topsis = TOPSIS(decision_matrix)
    ideal, anti_ideal = topsis.calculate_ideal_solutions()
    
    assert len(ideal) == 4
    assert len(anti_ideal) == 4
    
    # For benefit criteria (first two), ideal should be greater than anti-ideal
    assert np.all(ideal[:2] >= anti_ideal[:2])
    # For cost criteria (last two), ideal should be less than anti-ideal
    assert np.all(ideal[2:] <= anti_ideal[2:])

def test_topsis_calculate_distances():
    """Test distance calculation in TOPSIS."""
    matrix = np.array([
        [4, 3, 5, 2],
        [3, 4, 2, 5],
        [5, 3, 4, 3]
    ])
    decision_matrix = DecisionMatrix(
        decision_matrix=matrix,
        criteria_types=['benefit', 'benefit', 'cost', 'cost']
    )
    
    topsis = TOPSIS(decision_matrix)
    d_ideal, d_anti_ideal = topsis.calculate_distances()
    
    assert len(d_ideal) == 3
    assert len(d_anti_ideal) == 3
    assert np.all(d_ideal >= 0)
    assert np.all(d_anti_ideal >= 0)
    
    # Test that distances are calculated correctly with weights
    weights = np.array([0.3, 0.2, 0.3, 0.2])
    topsis = TOPSIS(decision_matrix, weights=weights)
    d_ideal_weighted, d_anti_ideal_weighted = topsis.calculate_distances()
    
    # Distances should be different with weights
    assert not np.array_equal(d_ideal, d_ideal_weighted)
    assert not np.array_equal(d_anti_ideal, d_anti_ideal_weighted)

def test_topsis_calculate_scores():
    """Test score calculation in TOPSIS."""
    matrix = np.array([
        [4, 3, 5, 2],
        [3, 4, 2, 5],
        [5, 3, 4, 3]
    ])
    decision_matrix = DecisionMatrix(
        decision_matrix=matrix,
        criteria_types=['benefit', 'benefit', 'cost', 'cost']
    )
    
    topsis = TOPSIS(decision_matrix)
    scores = topsis.calculate_scores()
    
    assert len(scores) == 3
    assert np.all(scores >= 0)
    assert np.all(scores <= 1)
    
    # Test with specific weights
    weights = np.array([0.3, 0.2, 0.3, 0.2])
    topsis = TOPSIS(decision_matrix, weights=weights)
    scores_weighted = topsis.calculate_scores()
    
    # Scores should be different with weights
    assert not np.array_equal(scores, scores_weighted)
    
    # Test that scores sum to 1 when normalized
    normalized_scores = scores / np.sum(scores)
    assert np.isclose(np.sum(normalized_scores), 1.0)

def test_topsis_normalization_edge_cases():
    """Test TOPSIS normalization with edge cases."""
    # Test with all equal values
    matrix = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]
    ])
    decision_matrix = DecisionMatrix(
        decision_matrix=matrix,
        criteria_types=['benefit', 'benefit', 'cost', 'cost']
    )
    
    # Test vector normalization
    topsis = TOPSIS(decision_matrix, normalization_method='vector')
    normalized = topsis.normalize_matrix()
    expected = 1 / np.sqrt(3)  # For vector normalization with equal values
    assert np.allclose(normalized, expected)
    
    # Test minmax normalization
    topsis = TOPSIS(decision_matrix, normalization_method='minmax')
    normalized = topsis.normalize_matrix()
    assert np.all(normalized == 1.0)  # For minmax normalization with equal values
    
    # Test with zero values
    matrix = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
    decision_matrix = DecisionMatrix(
        decision_matrix=matrix,
        criteria_types=['benefit', 'benefit', 'cost', 'cost']
    )
    
    # Test vector normalization with zeros
    topsis = TOPSIS(decision_matrix, normalization_method='vector')
    normalized = topsis.normalize_matrix()
    assert np.all(normalized == 0.0)  # For vector normalization with zeros
    
    # Test minmax normalization with zeros
    topsis = TOPSIS(decision_matrix, normalization_method='minmax')
    normalized = topsis.normalize_matrix()
    assert np.all(normalized == 1.0)  # For minmax normalization with equal values

def test_topsis_rank():
    """Test ranking in TOPSIS."""
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
    
    topsis = TOPSIS(decision_matrix)
    results = topsis.rank()
    
    assert 'rankings' in results
    assert 'scores' in results
    assert 'ideal_solution' in results
    assert 'anti_ideal_solution' in results
    
    rankings = results['rankings']
    assert len(rankings) == 3
    assert all('rank' in r for r in rankings)
    assert all('alternative' in r for r in rankings)
    assert all('score' in r for r in rankings)
    
    # Check that rankings are in descending order of scores
    scores = [r['score'] for r in rankings]
    assert scores == sorted(scores, reverse=True)

def test_topsis_invalid_inputs():
    """Test TOPSIS with invalid inputs."""
    # Test with empty matrix
    with pytest.raises(ValueError):
        decision_matrix = DecisionMatrix(
            decision_matrix=np.array([]),
            criteria_types=['benefit']
        )
        TOPSIS(decision_matrix)
    
    # Test with invalid weights
    matrix = np.array([[1, 2], [3, 4]])
    decision_matrix = DecisionMatrix(
        decision_matrix=matrix,
        criteria_types=['benefit', 'benefit']
    )
    with pytest.raises(ValueError):
        TOPSIS(decision_matrix, weights=np.array([0.5, 0.5, 0.5]))
    
    # Test with invalid normalization method
    with pytest.raises(ValueError):
        TOPSIS(decision_matrix, normalization_method='invalid')

def test_topsis_sum_normalization():
    """Test TOPSIS with sum normalization method."""
    # Test with zero values in cost criteria
    matrix = np.array([
        [4, 3, 0, 2],
        [3, 4, 2, 5],
        [5, 3, 4, 0]
    ])
    decision_matrix = DecisionMatrix(
        decision_matrix=matrix,
        criteria_types=['benefit', 'benefit', 'cost', 'cost']
    )
    
    topsis = TOPSIS(decision_matrix, normalization_method='sum')
    normalized = topsis.normalize_matrix()
    
    assert normalized.shape == matrix.shape
    assert np.all(normalized >= 0)
    assert np.all(normalized <= 1)
    
    # Test that sum normalization handles zero values correctly
    for j in range(matrix.shape[1]):
        if decision_matrix.criteria_types[j].lower() == 'benefit':
            assert np.isclose(np.sum(normalized[:, j]), 1.0)
        else:  # cost criterion
            assert np.isclose(np.sum(normalized[:, j]), 1.0) 