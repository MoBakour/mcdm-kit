"""
Tests for DEMATEL implementation.
"""

import numpy as np
import pytest
from mcdm_kit.core.dematel import DEMATEL
from mcdm_kit.data.decision_matrix import DecisionMatrix

def test_dematel_initialization():
    """Test DEMATEL initialization with default and custom parameters."""
    # Create a sample decision matrix
    matrix = np.array([
        [0, 3, 2],
        [2, 0, 1],
        [1, 2, 0]
    ])
    alternatives = ['A1', 'A2', 'A3']
    criteria = ['C1', 'C2', 'C3']
    criteria_types = ['benefit', 'benefit', 'benefit']
    
    decision_matrix = DecisionMatrix(matrix, alternatives, criteria, criteria_types)
    
    # Test with default parameters
    dematel = DEMATEL(decision_matrix)
    assert dematel.decision_matrix == decision_matrix
    assert dematel.threshold is None
    assert dematel.alpha == 0.1
    
    # Test with custom parameters
    dematel = DEMATEL(
        decision_matrix,
        threshold=0.5,
        alpha=0.2
    )
    assert dematel.threshold == 0.5
    assert dematel.alpha == 0.2

def test_dematel_normalize_matrix():
    """Test matrix normalization."""
    matrix = np.array([
        [0, 3, 2],
        [2, 0, 1],
        [1, 2, 0]
    ])
    decision_matrix = DecisionMatrix(
        matrix,
        ['A1', 'A2', 'A3'],
        ['C1', 'C2', 'C3'],
        ['benefit', 'benefit', 'benefit']
    )
    
    dematel = DEMATEL(decision_matrix)
    normalized = dematel.normalize_matrix()
    
    assert normalized.shape == matrix.shape
    assert np.all(normalized >= 0)
    assert np.all(normalized <= 1)
    assert np.isclose(np.max(np.sum(normalized, axis=1)), 1.0)

def test_dematel_calculate_total_relation_matrix():
    """Test total relation matrix calculation."""
    matrix = np.array([
        [0, 3, 2],
        [2, 0, 1],
        [1, 2, 0]
    ])
    decision_matrix = DecisionMatrix(
        matrix,
        ['A1', 'A2', 'A3'],
        ['C1', 'C2', 'C3'],
        ['benefit', 'benefit', 'benefit']
    )
    
    dematel = DEMATEL(decision_matrix)
    total_relation = dematel.calculate_total_relation_matrix()
    
    assert total_relation.shape == matrix.shape
    assert np.all(total_relation >= 0)

def test_dematel_calculate_cause_effect_matrix():
    """Test cause-effect matrix calculation."""
    matrix = np.array([
        [0, 3, 2],
        [2, 0, 1],
        [1, 2, 0]
    ])
    decision_matrix = DecisionMatrix(
        matrix,
        ['A1', 'A2', 'A3'],
        ['C1', 'C2', 'C3'],
        ['benefit', 'benefit', 'benefit']
    )
    
    dematel = DEMATEL(decision_matrix)
    cause_effect, relationships = dematel.calculate_cause_effect_matrix()
    
    assert cause_effect.shape == (3, 2)
    assert isinstance(relationships, list)
    assert all(isinstance(r, dict) for r in relationships)
    assert all('from' in r and 'to' in r and 'influence' in r for r in relationships)

def test_dematel_rank():
    """Test ranking functionality."""
    matrix = np.array([
        [0, 3, 2],
        [2, 0, 1],
        [1, 2, 0]
    ])
    alternatives = ['A1', 'A2', 'A3']
    criteria = ['C1', 'C2', 'C3']
    criteria_types = ['benefit', 'benefit', 'benefit']
    
    decision_matrix = DecisionMatrix(matrix, alternatives, criteria, criteria_types)
    dematel = DEMATEL(decision_matrix)
    results = dematel.rank()
    
    assert 'criteria_analysis' in results
    assert 'influence_relationships' in results
    assert 'total_relation_matrix' in results
    assert 'threshold' in results
    
    analysis = results['criteria_analysis']
    assert len(analysis) == len(criteria)
    assert all('criterion' in a for a in analysis)
    assert all('prominence' in a for a in analysis)
    assert all('relation' in a for a in analysis)
    assert all('prominence_rank' in a for a in analysis)
    assert all('relation_rank' in a for a in analysis)
    
    relationships = results['influence_relationships']
    assert isinstance(relationships, list)
    assert all(isinstance(r, dict) for r in relationships)
    assert all('from' in r and 'to' in r and 'influence' in r for r in relationships)

def test_dematel_invalid_inputs():
    """Test handling of invalid inputs."""
    # Test with empty matrix
    with pytest.raises(ValueError):
        matrix = np.array([])
        decision_matrix = DecisionMatrix(matrix, [], [], [])
        DEMATEL(decision_matrix)
    
    # Test with invalid threshold
    matrix = np.array([[0, 1], [1, 0]])
    decision_matrix = DecisionMatrix(
        matrix,
        ['A1', 'A2'],
        ['C1', 'C2'],
        ['benefit', 'benefit']
    )
    
    with pytest.raises(ValueError):
        DEMATEL(decision_matrix, threshold=-0.5)
    
    # Test with invalid alpha
    with pytest.raises(ValueError):
        DEMATEL(decision_matrix, alpha=0)
    
    with pytest.raises(ValueError):
        DEMATEL(decision_matrix, alpha=1.5) 