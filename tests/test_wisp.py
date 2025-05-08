"""
Tests for WISP implementation.
"""

import numpy as np
import pytest
from mcdm_kit.core.wisp import WISP
from mcdm_kit.data.decision_matrix import DecisionMatrix

def test_wisp_initialization():
    """Test WISP initialization with default and custom parameters."""
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
    wisp = WISP(decision_matrix)
    assert wisp.decision_matrix == decision_matrix
    assert wisp.weights is None
    assert wisp.normalization_method == 'vector'
    assert wisp.performance_thresholds is None
    
    # Test with custom parameters
    weights = np.array([0.3, 0.3, 0.4])
    performance_thresholds = np.array([7, 2, 9])
    wisp = WISP(
        decision_matrix,
        weights=weights,
        normalization_method='minmax',
        performance_thresholds=performance_thresholds
    )
    assert np.array_equal(wisp.weights, weights)
    assert wisp.normalization_method == 'minmax'
    assert np.array_equal(wisp.performance_thresholds, performance_thresholds)

def test_wisp_calculate_weights():
    """Test weight calculation functionality."""
    matrix = np.array([[1, 2], [3, 4]])
    decision_matrix = DecisionMatrix(
        matrix,
        ['A1', 'A2'],
        ['C1', 'C2'],
        ['benefit', 'cost']
    )
    
    # Test with no weights provided
    wisp = WISP(decision_matrix)
    weights = wisp.calculate_weights()
    assert len(weights) == 2
    assert np.isclose(np.sum(weights), 1.0)
    assert np.all(weights > 0)
    
    # Test with custom weights
    custom_weights = np.array([0.7, 0.3])
    wisp = WISP(decision_matrix, weights=custom_weights)
    weights = wisp.calculate_weights()
    assert np.array_equal(weights, custom_weights)
    
    # Test with invalid weights
    with pytest.raises(ValueError):
        wisp = WISP(decision_matrix, weights=np.array([0.5]))
        wisp.calculate_weights()

def test_wisp_calculate_performance_thresholds():
    """Test performance threshold calculation."""
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
    wisp = WISP(decision_matrix)
    thresholds = wisp.calculate_performance_thresholds()
    assert len(thresholds) == 3
    assert np.isclose(thresholds[0], 7)  # max of [1, 4, 7]
    assert np.isclose(thresholds[1], 2)  # min of [2, 5, 8]
    assert np.isclose(thresholds[2], 9)  # max of [3, 6, 9]
    
    # Test with custom thresholds
    custom_thresholds = np.array([7, 2, 9])
    wisp = WISP(decision_matrix, performance_thresholds=custom_thresholds)
    thresholds = wisp.calculate_performance_thresholds()
    assert np.array_equal(thresholds, custom_thresholds)
    
    # Test with invalid thresholds
    with pytest.raises(ValueError):
        wisp = WISP(decision_matrix, performance_thresholds=np.array([1, 2]))
        wisp.calculate_performance_thresholds()

def test_wisp_normalize_matrix():
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
    
    wisp = WISP(decision_matrix)
    normalized = wisp.normalize_matrix()
    
    assert normalized.shape == matrix.shape
    assert np.all(normalized >= 0)
    assert np.all(normalized <= 1)

def test_wisp_calculate_weighted_matrix():
    """Test weighted matrix calculation."""
    matrix = np.array([[1, 2], [3, 4]])
    decision_matrix = DecisionMatrix(
        matrix,
        ['A1', 'A2'],
        ['C1', 'C2'],
        ['benefit', 'cost']
    )
    
    weights = np.array([0.6, 0.4])
    wisp = WISP(decision_matrix, weights=weights)
    weighted = wisp.calculate_weighted_matrix()
    
    assert weighted.shape == matrix.shape
    assert np.all(weighted >= 0)

def test_wisp_calculate_performance_matrix():
    """Test performance matrix calculation."""
    matrix = np.array([[1, 2], [3, 4]])
    decision_matrix = DecisionMatrix(
        matrix,
        ['A1', 'A2'],
        ['C1', 'C2'],
        ['benefit', 'cost']
    )
    
    wisp = WISP(decision_matrix)
    performance = wisp.calculate_performance_matrix()
    
    assert performance.shape == matrix.shape
    assert np.isclose(np.mean(performance), 0, atol=1e-10)

def test_wisp_calculate_scores():
    """Test score calculation."""
    matrix = np.array([[1, 2], [3, 4]])
    decision_matrix = DecisionMatrix(
        matrix,
        ['A1', 'A2'],
        ['C1', 'C2'],
        ['benefit', 'cost']
    )
    
    wisp = WISP(decision_matrix)
    scores = wisp.calculate_scores()
    
    assert len(scores) == len(decision_matrix.alternatives)
    assert np.isclose(np.sum(scores), 0, atol=1e-10)

def test_wisp_rank():
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
    wisp = WISP(decision_matrix)
    results = wisp.rank()
    
    assert 'rankings' in results
    assert 'scores' in results
    assert 'performance_thresholds' in results
    assert 'performance_matrix' in results
    
    rankings = results['rankings']
    assert len(rankings) == len(alternatives)
    assert all('rank' in r for r in rankings)
    assert all('alternative' in r for r in rankings)
    assert all('score' in r for r in rankings)
    
    scores = results['scores']
    assert len(scores) == len(alternatives)
    assert all(alt in scores for alt in alternatives)
    
    performance_thresholds = results['performance_thresholds']
    assert len(performance_thresholds) == len(criteria)
    assert all(crit in performance_thresholds for crit in criteria)

def test_wisp_invalid_inputs():
    """Test handling of invalid inputs."""
    # Test with empty matrix
    with pytest.raises(ValueError):
        matrix = np.array([])
        decision_matrix = DecisionMatrix(matrix, [], [], [])
        WISP(decision_matrix)
    
    # Test with invalid weights
    matrix = np.array([[1, 2], [3, 4]])
    decision_matrix = DecisionMatrix(
        matrix,
        ['A1', 'A2'],
        ['C1', 'C2'],
        ['benefit', 'cost']
    )
    
    with pytest.raises(ValueError):
        WISP(decision_matrix, weights=np.array([-0.5, 0.5]))
    
    # Test with invalid normalization method
    with pytest.raises(ValueError):
        WISP(decision_matrix, normalization_method='invalid')
    
    # Test with invalid performance thresholds
    with pytest.raises(ValueError):
        WISP(decision_matrix, performance_thresholds=np.array([-1, 2])) 