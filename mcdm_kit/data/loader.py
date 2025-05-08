"""
Data loader for expert evaluation data.
"""

import csv
import json
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from .decision_matrix import DecisionMatrix

class ExpertDataLoader:
    """
    Loader for expert evaluation data from CSV or JSON files.
    
    The loader can handle:
    1. Expert importance rankings
    2. Criteria importance rankings
    3. Alternative evaluations by experts
    """
    
    # Default linguistic term mappings
    DEFAULT_EXPERT_IMPORTANCE_MAP = {
        'Unimportant (UI)': 1,
        'Less Important (LI)': 2,
        'Medium (M)': 3,
        'Important (I)': 4,
        'Very Important (VI)': 5
    }
    
    DEFAULT_EVALUATION_MAP = {
        'Very Very Bad (VVB)': 1,
        'Very Bad (VB)': 2,
        'Bad (B)': 3,
        'Medium Bad (MB)': 4,
        'Medium (M)': 5,
        'Medium Good (MG)': 6,
        'Good (G)': 7,
        'Very Good (VG)': 8,
        'Very Very Good (VVG)': 9,
        'Extremely Good (EG)': 10
    }
    
    def __init__(self, file_path: str):
        """
        Initialize the expert data loader.
        
        Args:
            file_path (str): Path to the expert data file (CSV or JSON)
        """
        self.file_path = file_path
        self.data = None
        self.expert_importance = None
        self.criteria_importance = None
        self.alternative_evaluations = None
        self.n_criteria = None
        self.n_alternatives = None
        
        # Initialize with default mappings
        self.expert_importance_map = self.DEFAULT_EXPERT_IMPORTANCE_MAP.copy()
        self.evaluation_map = self.DEFAULT_EVALUATION_MAP.copy()
        
    def set_expert_importance_map(self, mapping: Dict[str, int]) -> None:
        """
        Set custom mapping for expert importance terms.
        
        Args:
            mapping (Dict[str, int]): Dictionary mapping linguistic terms to numerical values
        """
        self.expert_importance_map = mapping.copy()
        
    def set_evaluation_map(self, mapping: Dict[str, int]) -> None:
        """
        Set custom mapping for evaluation terms.
        
        Args:
            mapping (Dict[str, int]): Dictionary mapping linguistic terms to numerical values
        """
        self.evaluation_map = mapping.copy()
        
    def get_expert_importance_map(self) -> Dict[str, int]:
        """
        Get current expert importance mapping.
        
        Returns:
            Dict[str, int]: Current mapping of expert importance terms
        """
        return self.expert_importance_map.copy()
        
    def get_evaluation_map(self) -> Dict[str, int]:
        """
        Get current evaluation mapping.
        
        Returns:
            Dict[str, int]: Current mapping of evaluation terms
        """
        return self.evaluation_map.copy()
        
    def reset_mappings(self) -> None:
        """
        Reset mappings to default values.
        """
        self.expert_importance_map = self.DEFAULT_EXPERT_IMPORTANCE_MAP.copy()
        self.evaluation_map = self.DEFAULT_EVALUATION_MAP.copy()
        
    def load(self, 
            expert_importance_col: Optional[str] = None,
            criteria_importance_cols: Optional[List[str]] = None,
            alternative_evaluation_cols: Optional[List[str]] = None) -> None:
        """
        Load the expert data from file.
        
        Args:
            expert_importance_col (str, optional): Name of the column containing expert importance rankings
            criteria_importance_cols (List[str], optional): Names of columns containing criteria importance rankings
            alternative_evaluation_cols (List[str], optional): Names of columns containing alternative evaluations
            
        Raises:
            ValueError: If file format is not supported or data is invalid
        """
        if self.file_path.endswith('.csv'):
            self._load_csv(expert_importance_col, criteria_importance_cols, alternative_evaluation_cols)
        elif self.file_path.endswith('.json'):
            self._load_json()
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")
            
    def _convert_linguistic_terms(self, values: np.ndarray, mapping: Dict[str, int], default: int = 5) -> np.ndarray:
        """
        Convert linguistic terms to numerical values.
        
        Args:
            values (np.ndarray): Array of linguistic terms
            mapping (Dict[str, int]): Mapping from terms to values
            default (int): Default value for unknown terms
            
        Returns:
            np.ndarray: Array of numerical values
        """
        if values.dtype == np.dtype('O'):  # If array contains strings
            return np.array([mapping.get(str(val), default) for val in values])
        return values
            
    def _load_csv(self,
                 expert_importance_col: Optional[str] = None,
                 criteria_importance_cols: Optional[List[str]] = None,
                 alternative_evaluation_cols: Optional[List[str]] = None) -> None:
        """
        Load expert data from CSV file.
        
        Args:
            expert_importance_col (str, optional): Name of the column containing expert importance rankings
            criteria_importance_cols (List[str], optional): Names of columns containing criteria importance rankings
            alternative_evaluation_cols (List[str], optional): Names of columns containing alternative evaluations
        """
        try:
            # Read CSV file
            df = pd.read_csv(self.file_path)
            
            # Extract expert importance if column specified
            if expert_importance_col is not None:
                if expert_importance_col not in df.columns:
                    raise ValueError(f"Expert importance column '{expert_importance_col}' not found in CSV")
                self.expert_importance = df[expert_importance_col].values
            else:
                self.expert_importance = None
            
            # Extract criteria importance if columns specified
            if criteria_importance_cols is not None:
                missing_cols = [col for col in criteria_importance_cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Criteria importance columns not found in CSV: {missing_cols}")
                self.criteria_importance = df[criteria_importance_cols].values
                self.n_criteria = len(criteria_importance_cols)
            else:
                self.criteria_importance = None
                self.n_criteria = None
            
            # Extract alternative evaluations if columns specified
            if alternative_evaluation_cols is not None:
                missing_cols = [col for col in alternative_evaluation_cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Alternative evaluation columns not found in CSV: {missing_cols}")
                
                # Convert linguistic terms to numerical values
                evaluations = []
                for col in alternative_evaluation_cols:
                    values = df[col].values
                    numerical_values = self._convert_linguistic_terms(values, self.evaluation_map)
                    evaluations.append(numerical_values)
                
                # Convert to numpy array
                evaluations = np.array(evaluations)  # Shape: (n_cols, n_rows)
                
                # Determine number of alternatives and criteria
                n_total_cols = len(alternative_evaluation_cols)
                n_criteria = len(criteria_importance_cols) if criteria_importance_cols else 10  # Use number of criteria from criteria_importance_cols
                n_alternatives = n_total_cols // n_criteria
                
                # Reshape to (n_alternatives, n_criteria, n_experts)
                evaluations = evaluations.reshape(n_alternatives, n_criteria, -1)
                
                # Average over experts to get final matrix
                self.alternative_evaluations = np.mean(evaluations, axis=2)
                self.n_alternatives = n_alternatives
                self.n_criteria = n_criteria
            else:
                self.alternative_evaluations = None
                self.n_alternatives = None
            
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {str(e)}")
            
    def _load_json(self) -> None:
        """
        Load expert data from JSON file.
        
        The JSON file should have the following structure:
        {
            "expert_importance": [...],
            "criteria_importance": [...],
            "alternative_evaluations": [...]
        }
        """
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
                
            self.expert_importance = np.array(data['expert_importance'])
            self.criteria_importance = np.array(data['criteria_importance'])
            
            # Convert alternative evaluations to numerical values
            evaluations = np.array(data['alternative_evaluations'])
            self.alternative_evaluations = self._convert_linguistic_terms(evaluations, self.evaluation_map)
            
            self.n_criteria = self.criteria_importance.shape[1]
            self.n_alternatives = len(self.alternative_evaluations)
            
        except Exception as e:
            raise ValueError(f"Error loading JSON file: {str(e)}")
            
    def get_expert_weights(self) -> np.ndarray:
        """
        Get normalized expert weights based on importance rankings.
        
        Returns:
            np.ndarray: Array of normalized expert weights
        """
        if self.expert_importance is None:
            raise ValueError("Data not loaded. Call load() first.")
            
        weights = self._convert_linguistic_terms(self.expert_importance, self.expert_importance_map)
        return weights / np.sum(weights)
        
    def get_criteria_weights(self) -> np.ndarray:
        """
        Get normalized criteria weights based on importance rankings.
        
        Returns:
            np.ndarray: Array of normalized criteria weights
        """
        if self.criteria_importance is None:
            raise ValueError("Data not loaded. Call load() first.")
            
        # Calculate average importance for each criterion
        weights = np.mean([self._convert_linguistic_terms(row, self.evaluation_map) 
                          for row in self.criteria_importance], axis=0)
        return weights / np.sum(weights)
        
    def get_decision_matrix(self,
                          alternatives: Optional[List[str]] = None,
                          criteria: Optional[List[str]] = None,
                          criteria_types: Optional[List[str]] = None) -> DecisionMatrix:
        """
        Create a decision matrix from expert evaluations.
        
        Args:
            alternatives (Optional[List[str]]): List of alternative names
            criteria (Optional[List[str]]): List of criterion names
            criteria_types (Optional[List[str]]): List of criterion types ('benefit' or 'cost')
            
        Returns:
            DecisionMatrix: Decision matrix with aggregated expert evaluations
        """
        if self.alternative_evaluations is None:
            raise ValueError("Data not loaded. Call load() first.")
            
        # Create default names if not provided
        if alternatives is None:
            alternatives = [f'A{i+1}' for i in range(self.n_alternatives)]
        if criteria is None:
            criteria = [f'C{i+1}' for i in range(self.n_criteria)]
        if criteria_types is None:
            criteria_types = ['benefit'] * self.n_criteria
            
        return DecisionMatrix(
            decision_matrix=self.alternative_evaluations,
            alternatives=alternatives[:self.n_alternatives],  # Ensure correct length
            criteria=criteria[:self.n_criteria],  # Ensure correct length
            criteria_types=criteria_types[:self.n_criteria]  # Ensure correct length
        )
        
    def get_aggregated_evaluations(self) -> Dict[str, Any]:
        """
        Get aggregated expert evaluations with weights.
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - expert_weights: Normalized expert weights
                - criteria_weights: Normalized criteria weights
                - decision_matrix: Decision matrix with aggregated evaluations
        """
        return {
            'expert_weights': self.get_expert_weights(),
            'criteria_weights': self.get_criteria_weights(),
            'decision_matrix': self.get_decision_matrix()
        } 