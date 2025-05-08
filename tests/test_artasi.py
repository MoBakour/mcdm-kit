"""
Tests for ARTASI implementation.
"""

import numpy as np
import pytest
from mcdm_kit.core.artasi import ARTASI
from mcdm_kit.data.decision_matrix import DecisionMatrix

def test_artasi_initialization():
    """Test ARTASI initialization with default and custom parameters."""
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
    artasi = ARTASI(decision_matrix)
    assert artasi.decision_matrix == decision_matrix
    assert artasi.weights is None
    assert artasi.normalization_method == 'vector'
    assert artasi.aspiration_levels is None
    
    # Test with custom parameters
    weights = np.array([0.3, 0.3, 0.4])
    aspiration_levels = np.array([8, 2, 9])
    artasi = ARTASI(
        decision_matrix,
        weights=weights,
        normalization_method='minmax',
        aspiration_levels=aspiration_levels
    )
    assert np.array_equal(artasi.weights, weights)
    assert artasi.normalization_method == 'minmax'
    assert np.array_equal(artasi.aspiration_levels, aspiration_levels)

def test_artasi_calculate_weights():
    """Test weight calculation functionality."""
    matrix = np.array([[1, 2], [3, 4]])
    decision_matrix = DecisionMatrix(
        matrix,
        ['A1', 'A2'],
        ['C1', 'C2'],
        ['benefit', 'cost']
    )
    
    # Test with no weights provided
    artasi = ARTASI(decision_matrix)
    weights = artasi.calculate_weights()
    assert len(weights) == 2
    assert np.isclose(np.sum(weights), 1.0)
    assert np.all(weights > 0)
    
    # Test with custom weights
    custom_weights = np.array([0.7, 0.3])
    artasi = ARTASI(decision_matrix, weights=custom_weights)
    weights = artasi.calculate_weights()
    assert np.array_equal(weights, custom_weights)
    
    # Test with invalid weights
    with pytest.raises(ValueError):
        artasi = ARTASI(decision_matrix, weights=np.array([0.5]))
        artasi.calculate_weights()

def test_artasi_calculate_aspiration_levels():
    """Test aspiration level calculation."""
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
    artasi = ARTASI(decision_matrix)
    aspiration_levels = artasi.calculate_aspiration_levels()
    assert len(aspiration_levels) == 3
    assert aspiration_levels[0] == 7  # max for benefit
    assert aspiration_levels[1] == 2  # min for cost
    assert aspiration_levels[2] == 9  # max for benefit
    
    # Test with custom aspiration levels
    custom_levels = np.array([8, 3, 7])
    artasi = ARTASI(decision_matrix, aspiration_levels=custom_levels)
    levels = artasi.calculate_aspiration_levels()
    assert np.array_equal(levels, custom_levels)
    
    # Test with invalid aspiration levels
    with pytest.raises(ValueError):
        artasi = ARTASI(decision_matrix, aspiration_levels=np.array([1, 2]))
        artasi.calculate_aspiration_levels()

def test_artasi_normalize_matrix():
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
    
    artasi = ARTASI(decision_matrix)
    normalized = artasi.normalize_matrix()
    
    assert normalized.shape == matrix.shape
    assert np.all(normalized >= 0)
    assert np.all(normalized <= 1)

def test_artasi_calculate_weighted_matrix():
    """Test weighted matrix calculation."""
    matrix = np.array([[1, 2], [3, 4]])
    decision_matrix = DecisionMatrix(
        matrix,
        ['A1', 'A2'],
        ['C1', 'C2'],
        ['benefit', 'cost']
    )
    
    weights = np.array([0.6, 0.4])
    artasi = ARTASI(decision_matrix, weights=weights)
    weighted = artasi.calculate_weighted_matrix()
    
    assert weighted.shape == matrix.shape
    assert np.all(weighted >= 0)

def test_artasi_calculate_aspiration_matrix():
    """Test aspiration matrix calculation."""
    matrix = np.array([[1, 2], [3, 4]])
    decision_matrix = DecisionMatrix(
        matrix,
        ['A1', 'A2'],
        ['C1', 'C2'],
        ['benefit', 'cost']
    )
    
    weights = np.array([0.6, 0.4])
    artasi = ARTASI(decision_matrix, weights=weights)
    aspiration = artasi.calculate_aspiration_matrix()
    
    assert aspiration.shape == matrix.shape
    assert np.all(aspiration >= 0)
    assert np.all(aspiration <= 1)

def test_artasi_calculate_distance_matrix():
    """Test distance matrix calculation."""
    matrix = np.array([[1, 2], [3, 4]])
    decision_matrix = DecisionMatrix(
        matrix,
        ['A1', 'A2'],
        ['C1', 'C2'],
        ['benefit', 'cost']
    )
    
    artasi = ARTASI(decision_matrix)
    distance = artasi.calculate_distance_matrix()
    
    assert distance.shape == matrix.shape
    assert np.all(distance >= 0)

def test_artasi_calculate_scores():
    """Test score calculation."""
    matrix = np.array([[1, 2], [3, 4]])
    decision_matrix = DecisionMatrix(
        matrix,
        ['A1', 'A2'],
        ['C1', 'C2'],
        ['benefit', 'cost']
    )
    
    artasi = ARTASI(decision_matrix)
    scores = artasi.calculate_scores()
    
    assert len(scores) == len(decision_matrix.alternatives)
    assert np.all(scores >= 0)
    assert np.all(scores <= 1)

def test_artasi_rank():
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
    artasi = ARTASI(decision_matrix)
    results = artasi.rank()
    
    assert 'rankings' in results
    assert 'scores' in results
    assert 'aspiration_levels' in results
    assert 'distance_matrix' in results
    
    rankings = results['rankings']
    assert len(rankings) == len(alternatives)
    assert all('rank' in r for r in rankings)
    assert all('alternative' in r for r in rankings)
    assert all('score' in r for r in rankings)
    
    scores = results['scores']
    assert len(scores) == len(alternatives)
    assert all(alt in scores for alt in alternatives)
    
    aspiration_levels = results['aspiration_levels']
    assert len(aspiration_levels) == len(criteria)
    assert all(crit in aspiration_levels for crit in criteria)

def test_artasi_invalid_inputs():
    """Test handling of invalid inputs."""
    # Test with empty matrix
    with pytest.raises(ValueError):
        matrix = np.array([])
        decision_matrix = DecisionMatrix(matrix, [], [], [])
        ARTASI(decision_matrix)
    
    # Test with invalid weights
    matrix = np.array([[1, 2], [3, 4]])
    decision_matrix = DecisionMatrix(
        matrix,
        ['A1', 'A2'],
        ['C1', 'C2'],
        ['benefit', 'cost']
    )
    
    with pytest.raises(ValueError):
        ARTASI(decision_matrix, weights=np.array([-0.5, 0.5]))
    
    # Test with invalid normalization method
    with pytest.raises(ValueError):
        ARTASI(decision_matrix, normalization_method='invalid')
    
    # Test with invalid aspiration levels
    with pytest.raises(ValueError):
        ARTASI(decision_matrix, aspiration_levels=np.array([-1, 2])) 