"""
Tests for expert data loader.
"""

import os
import json
import numpy as np
import pytest
import pandas as pd
from mcdm_kit.data.loader import ExpertDataLoader
from mcdm_kit.data.decision_matrix import DecisionMatrix

def create_test_csv():
    """Create a test CSV file with expert data."""
    data = {
        'Expert Importance': ['Important (I)', 'Unimportant (UI)', 'Important (I)'],
        'Criterion 1': ['Good (G)', 'Bad (B)', 'Very Good (VG)'],
        'Criterion 2': ['Medium (M)', 'Very Bad (VB)', 'Good (G)'],
        'Alt1_C1': ['Very Good (VG)', 'Medium (M)', 'Good (G)'],
        'Alt1_C2': ['Good (G)', 'Bad (B)', 'Very Good (VG)'],
        'Alt2_C1': ['Medium (M)', 'Very Bad (VB)', 'Good (G)'],
        'Alt2_C2': ['Good (G)', 'Bad (B)', 'Very Good (VG)']
    }
    df = pd.DataFrame(data)
    df.to_csv('test_expert_data.csv', index=False)
    return 'test_expert_data.csv'

def create_test_json():
    """Create a test JSON file with expert data."""
    data = {
        'expert_importance': ['Important (I)', 'Unimportant (UI)', 'Important (I)'],
        'criteria_importance': [
            ['Good (G)', 'Medium (M)'],
            ['Bad (B)', 'Very Bad (VB)'],
            ['Very Good (VG)', 'Good (G)']
        ],
        'alternative_evaluations': [
            ['Very Good (VG)', 'Good (G)'],
            ['Medium (M)', 'Bad (B)'],
            ['Good (G)', 'Very Good (VG)']
        ]
    }
    with open('test_expert_data.json', 'w') as f:
        json.dump(data, f)
    return 'test_expert_data.json'

def test_loader_initialization():
    """Test ExpertDataLoader initialization."""
    loader = ExpertDataLoader('test.csv')
    assert loader.file_path == 'test.csv'
    assert loader.data is None
    assert loader.expert_importance is None
    assert loader.criteria_importance is None
    assert loader.alternative_evaluations is None

def test_load_csv():
    """Test loading data from CSV file."""
    file_path = create_test_csv()
    try:
        loader = ExpertDataLoader(file_path)
        loader.load(
            expert_importance_col='Expert Importance',
            criteria_importance_cols=['Criterion 1', 'Criterion 2'],
            alternative_evaluation_cols=['Alt1_C1', 'Alt1_C2', 'Alt2_C1', 'Alt2_C2']
        )
        
        assert loader.expert_importance is not None
        assert loader.criteria_importance is not None
        assert loader.alternative_evaluations is not None
        
        assert len(loader.expert_importance) == 3
        assert loader.criteria_importance.shape == (3, 2)
        assert loader.alternative_evaluations.shape == (2, 2)  # (n_alternatives, n_criteria)
        
    finally:
        os.remove(file_path)

def test_load_csv_missing_column():
    """Test loading CSV with missing column."""
    file_path = create_test_csv()
    try:
        loader = ExpertDataLoader(file_path)
        with pytest.raises(ValueError):
            loader.load(
                expert_importance_col='NonExistentColumn',
                criteria_importance_cols=['Criterion 1', 'Criterion 2']
            )
    finally:
        os.remove(file_path)

def test_load_csv_partial_columns():
    """Test loading CSV with only some columns specified."""
    file_path = create_test_csv()
    try:
        loader = ExpertDataLoader(file_path)
        loader.load(
            expert_importance_col='Expert Importance',
            criteria_importance_cols=['Criterion 1', 'Criterion 2']
        )
        
        assert loader.expert_importance is not None
        assert loader.criteria_importance is not None
        assert loader.alternative_evaluations is None
        
    finally:
        os.remove(file_path)

def test_load_json():
    """Test loading data from JSON file."""
    file_path = create_test_json()
    try:
        loader = ExpertDataLoader(file_path)
        loader.load()
        
        assert loader.expert_importance is not None
        assert loader.criteria_importance is not None
        assert loader.alternative_evaluations is not None
        
        assert len(loader.expert_importance) == 3
        assert loader.criteria_importance.shape == (3, 2)
        assert loader.alternative_evaluations.shape == (3, 2)  # (n_alternatives, n_criteria)
        
    finally:
        os.remove(file_path)

def test_invalid_file_format():
    """Test loading invalid file format."""
    with pytest.raises(ValueError):
        loader = ExpertDataLoader('test.txt')
        loader.load()

def test_get_expert_weights():
    """Test getting expert weights."""
    file_path = create_test_csv()
    try:
        loader = ExpertDataLoader(file_path)
        loader.load(expert_importance_col='Expert Importance')
        
        weights = loader.get_expert_weights()
        assert len(weights) == 3
        assert np.isclose(np.sum(weights), 1.0)
        assert np.all(weights > 0)
        
    finally:
        os.remove(file_path)

def test_get_criteria_weights():
    """Test getting criteria weights."""
    file_path = create_test_csv()
    try:
        loader = ExpertDataLoader(file_path)
        loader.load(criteria_importance_cols=['Criterion 1', 'Criterion 2'])
        
        weights = loader.get_criteria_weights()
        assert len(weights) == 2
        assert np.isclose(np.sum(weights), 1.0)
        assert np.all(weights > 0)
        
    finally:
        os.remove(file_path)

def test_get_decision_matrix():
    """Test getting decision matrix."""
    file_path = create_test_csv()
    try:
        loader = ExpertDataLoader(file_path)
        loader.load(
            criteria_importance_cols=['Criterion 1', 'Criterion 2'],
            alternative_evaluation_cols=['Alt1_C1', 'Alt1_C2', 'Alt2_C1', 'Alt2_C2']
        )
        
        matrix = loader.get_decision_matrix(
            alternatives=['A1', 'A2'],
            criteria=['C1', 'C2'],
            criteria_types=['benefit', 'cost']
        )
        
        assert matrix is not None
        assert isinstance(matrix, DecisionMatrix)
        assert matrix.matrix.shape == (2, 2)  # (n_alternatives, n_criteria)
        assert matrix.alternatives == ['A1', 'A2']
        assert matrix.criteria == ['C1', 'C2']
        assert matrix.criteria_types == ['benefit', 'cost']
        
    finally:
        os.remove(file_path)

def test_get_aggregated_evaluations():
    """Test getting aggregated evaluations."""
    file_path = create_test_csv()
    try:
        loader = ExpertDataLoader(file_path)
        loader.load(
            expert_importance_col='Expert Importance',
            criteria_importance_cols=['Criterion 1', 'Criterion 2'],
            alternative_evaluation_cols=['Alt1_C1', 'Alt1_C2', 'Alt2_C1', 'Alt2_C2']
        )
        
        results = loader.get_aggregated_evaluations()
        
        assert 'expert_weights' in results
        assert 'criteria_weights' in results
        assert 'decision_matrix' in results
        
        assert len(results['expert_weights']) == 3
        assert len(results['criteria_weights']) == 2
        assert results['decision_matrix'].matrix.shape == (2, 2)  # (n_alternatives, n_criteria)
        
    finally:
        os.remove(file_path)

def test_load_without_calling_load():
    """Test methods without calling load() first."""
    loader = ExpertDataLoader('test.csv')
    
    with pytest.raises(ValueError):
        loader.get_expert_weights()
        
    with pytest.raises(ValueError):
        loader.get_criteria_weights()
        
    with pytest.raises(ValueError):
        loader.get_decision_matrix()
        
    with pytest.raises(ValueError):
        loader.get_aggregated_evaluations()

def test_custom_mappings():
    """Test setting and using custom linguistic term mappings."""
    file_path = create_test_csv()
    try:
        loader = ExpertDataLoader(file_path)
        
        # Test default mappings
        assert loader.get_expert_importance_map() == ExpertDataLoader.DEFAULT_EXPERT_IMPORTANCE_MAP
        assert loader.get_evaluation_map() == ExpertDataLoader.DEFAULT_EVALUATION_MAP
        
        # Set custom mappings
        custom_expert_map = {
            'Low': 1,
            'Medium': 2,
            'High': 3
        }
        custom_eval_map = {
            'Poor': 1,
            'Fair': 2,
            'Good': 3,
            'Excellent': 4
        }
        
        loader.set_expert_importance_map(custom_expert_map)
        loader.set_evaluation_map(custom_eval_map)
        
        # Verify custom mappings were set
        assert loader.get_expert_importance_map() == custom_expert_map
        assert loader.get_evaluation_map() == custom_eval_map
        
        # Test reset functionality
        loader.reset_mappings()
        assert loader.get_expert_importance_map() == ExpertDataLoader.DEFAULT_EXPERT_IMPORTANCE_MAP
        assert loader.get_evaluation_map() == ExpertDataLoader.DEFAULT_EVALUATION_MAP
        
    finally:
        os.remove(file_path)

def test_custom_mappings_with_data():
    """Test using custom mappings with actual data."""
    file_path = create_test_csv()
    try:
        loader = ExpertDataLoader(file_path)
        
        # Create custom mappings that match the test data
        custom_expert_map = {
            'Important (I)': 4,
            'Unimportant (UI)': 1
        }
        custom_eval_map = {
            'Very Good (VG)': 8,
            'Good (G)': 7,
            'Medium (M)': 5,
            'Bad (B)': 3,
            'Very Bad (VB)': 2
        }
        
        loader.set_expert_importance_map(custom_expert_map)
        loader.set_evaluation_map(custom_eval_map)
        
        # Load data with custom mappings
        loader.load(
            expert_importance_col='Expert Importance',
            criteria_importance_cols=['Criterion 1', 'Criterion 2'],
            alternative_evaluation_cols=['Alt1_C1', 'Alt1_C2', 'Alt2_C1', 'Alt2_C2']
        )
        
        # Verify weights are calculated correctly with custom mappings
        expert_weights = loader.get_expert_weights()
        assert len(expert_weights) == 3
        assert np.isclose(np.sum(expert_weights), 1.0)
        
        criteria_weights = loader.get_criteria_weights()
        assert len(criteria_weights) == 2
        assert np.isclose(np.sum(criteria_weights), 1.0)
        
    finally:
        os.remove(file_path) 