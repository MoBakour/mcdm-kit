"""
Decision Matrix class for handling MCDM data structures.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple, Any
from ..fuzz.picture import PictureFuzzySet
from ..fuzz.base import BaseFuzzySet
from ..fuzz.interval import IntervalFuzzySet
from ..fuzz.type2 import Type2FuzzySet
from ..fuzz.intuitionistic import IntuitionisticFuzzySet
from ..fuzz.spherical import SphericalFuzzySet
from ..fuzz.neutrosophic import NeutrosophicSet
from ..fuzz.pythagorean import PythagoreanFuzzySet
from ..fuzz.fermatean import FermateanFuzzySet
from ..fuzz.hesitant import HesitantFuzzySet

# Map of fuzzy type strings to their corresponding classes
FUZZY_TYPE_MAP = {
    'PFS': PictureFuzzySet,
    'IFS': IntervalFuzzySet,
    'T2FS': Type2FuzzySet,
    'INFS': IntuitionisticFuzzySet,  # Changed from IFS to avoid duplicate key
    'SFS': SphericalFuzzySet,
    'NFS': NeutrosophicSet,
    'PYFS': PythagoreanFuzzySet,  # Changed from PFS to avoid duplicate key
    'FFS': FermateanFuzzySet,
    'HFS': HesitantFuzzySet
}

class DecisionMatrix:
    """Class for handling decision matrices in MCDM problems."""
    
    def __init__(self, 
                 decision_matrix: Union[np.ndarray, pd.DataFrame, List[List[Any]]],
                 alternatives: Optional[List[str]] = None,
                 criteria: Optional[List[str]] = None,
                 criteria_types: Optional[List[str]] = None,
                 fuzzy_type: Optional[str] = None):
        """
        Initialize the decision matrix.
        
        Args:
            decision_matrix: The decision matrix (can be numerical or fuzzy)
            alternatives: List of alternative names
            criteria: List of criterion names
            criteria_types: List of criterion types ('benefit' or 'cost')
            fuzzy_type: Type of fuzzy set ('PFS', 'IFS', 'T2FS', 'INFS', 'SFS', 'NFS', 'PYFS', 'FFS', 'HFS')
        """
        self.fuzzy_type = fuzzy_type
        self._raw_matrix = decision_matrix
        
        if fuzzy_type:
            if fuzzy_type not in FUZZY_TYPE_MAP:
                raise ValueError(f"Unsupported fuzzy type: {fuzzy_type}. Supported types: {list(FUZZY_TYPE_MAP.keys())}")
            self._init_fuzzy_matrix(decision_matrix, alternatives, criteria, criteria_types)
        else:
            self._init_numerical_matrix(decision_matrix, alternatives, criteria, criteria_types)
            
        self._validate()
    
    def _init_fuzzy_matrix(self, 
                          decision_matrix: List[List[Any]],
                          alternatives: Optional[List[str]],
                          criteria: Optional[List[str]],
                          criteria_types: Optional[List[str]]):
        """Initialize matrix with fuzzy sets."""
        if not isinstance(decision_matrix, (list, np.ndarray)):
            raise ValueError("Fuzzy matrix must be a 2D list or array")
            
        # Convert to numpy array of fuzzy sets
        self.fuzzy_matrix = np.array(decision_matrix, dtype=object)
        
        # Get expected fuzzy set type
        expected_type = FUZZY_TYPE_MAP[self.fuzzy_type]
        
        # Create numerical matrix for calculations
        self.matrix = np.zeros_like(self.fuzzy_matrix, dtype=float)
        for i in range(len(self.fuzzy_matrix)):
            for j in range(len(self.fuzzy_matrix[i])):
                fuzzy_set = self.fuzzy_matrix[i, j]
                if not isinstance(fuzzy_set, BaseFuzzySet):
                    raise ValueError(f"Invalid fuzzy set at position ({i}, {j})")
                if not isinstance(fuzzy_set, expected_type):
                    raise ValueError(f"Mixed fuzzy set types detected. Expected {expected_type.__name__}, got {type(fuzzy_set).__name__} at position ({i}, {j})")
                self.matrix[i, j] = fuzzy_set.score()
        
        # Set metadata
        self.alternatives = alternatives or [f"Alt_{i+1}" for i in range(len(self.matrix))]
        self.criteria = criteria or [f"Criterion_{i+1}" for i in range(self.matrix.shape[1])]
        self.criteria_types = criteria_types or ['benefit'] * len(self.criteria)
    
    def _init_numerical_matrix(self,
                              decision_matrix: Union[np.ndarray, pd.DataFrame],
                              alternatives: Optional[List[str]],
                              criteria: Optional[List[str]],
                              criteria_types: Optional[List[str]]):
        """Initialize matrix with numerical values."""
        if isinstance(decision_matrix, pd.DataFrame):
            self.matrix = decision_matrix.values
            if alternatives is None:
                alternatives = decision_matrix.index.tolist()
            if criteria is None:
                criteria = decision_matrix.columns.tolist()
        else:
            self.matrix = np.array(decision_matrix)
            if self.matrix.size == 0:
                raise ValueError("Decision matrix cannot be empty")
            if len(self.matrix.shape) != 2:
                raise ValueError("Decision matrix must be 2-dimensional")
        
        self.alternatives = alternatives or [f"Alt_{i+1}" for i in range(len(self.matrix))]
        self.criteria = criteria or [f"Criterion_{i+1}" for i in range(self.matrix.shape[1])]
        self.criteria_types = criteria_types or ['benefit'] * len(self.criteria)
    
    def _validate(self):
        """Validate the decision matrix and its components."""
        if len(self.alternatives) != len(self.matrix):
            raise ValueError("Number of alternatives doesn't match matrix rows")
        if len(self.criteria) != self.matrix.shape[1]:
            raise ValueError("Number of criteria doesn't match matrix columns")
        if len(self.criteria_types) != len(self.criteria):
            raise ValueError("Number of criterion types doesn't match number of criteria")
    
    def get_fuzzy_details(self) -> Optional[Dict[str, Dict[str, Dict[str, float]]]]:
        """
        Get detailed fuzzy information if matrix contains fuzzy sets.
        
        Returns:
            Optional[Dict]: Nested dictionary of fuzzy details or None if not fuzzy
        """
        if not self.fuzzy_type:
            return None
            
        details = {}
        for i, alt in enumerate(self.alternatives):
            details[alt] = {}
            for j, crit in enumerate(self.criteria):
                fuzzy_set = self.fuzzy_matrix[i, j]
                details[alt][crit] = fuzzy_set.to_dict()
        
        return details
    
    def get_fuzzy_distances(self) -> Optional[Dict[str, float]]:
        """
        Calculate distances between fuzzy sets if matrix contains fuzzy sets.
        
        Returns:
            Optional[Dict]: Dictionary of distances or None if not fuzzy
        """
        if not self.fuzzy_type:
            return None
            
        distances = {}
        for i, alt1 in enumerate(self.alternatives):
            for j, alt2 in enumerate(self.alternatives[i+1:], i+1):
                # Calculate average distance across all criteria
                crit_distances = []
                for k in range(len(self.criteria)):
                    dist = self.fuzzy_matrix[i, k].distance(self.fuzzy_matrix[j, k])
                    crit_distances.append(dist)
                avg_distance = sum(crit_distances) / len(crit_distances)
                distances[f"{alt1} vs {alt2}"] = avg_distance
        
        return distances
    
    @classmethod
    def from_array(cls, 
                  array: Union[List[List], np.ndarray],
                  alternatives: Optional[List[str]] = None,
                  criteria: Optional[List[str]] = None,
                  criteria_types: Optional[List[str]] = None,
                  fuzzy_type: Optional[str] = None) -> 'DecisionMatrix':
        """
        Create a DecisionMatrix from a 2D array.
        
        Args:
            array: 2D array of values (can be numerical or fuzzy)
            alternatives: List of alternative names
            criteria: List of criterion names
            criteria_types: List of criterion types
            fuzzy_type: Type of fuzzy set ('PFS', 'IFS', 'T2FS', 'SFS', 'NFS', 'PFS', 'FFS', 'HFS')
            
        Returns:
            DecisionMatrix: New DecisionMatrix instance
        """
        return cls(array, alternatives, criteria, criteria_types, fuzzy_type)
    
    @classmethod
    def from_csv(cls, 
                filepath: str,
                alternatives_col: Optional[str] = None,
                criteria_types: Optional[List[str]] = None) -> 'DecisionMatrix':
        """
        Create a DecisionMatrix from a CSV file.
        
        Args:
            filepath (str): Path to the CSV file
            alternatives_col (Optional[str]): Column name containing alternative names
            criteria_types (Optional[List[str]]): List of criterion types
            
        Returns:
            DecisionMatrix: New DecisionMatrix instance
        """
        df = pd.read_csv(filepath)
        
        if alternatives_col:
            alternatives = df[alternatives_col].tolist()
            df = df.drop(columns=[alternatives_col])
        else:
            alternatives = None
            
        return cls(df, alternatives=alternatives, criteria_types=criteria_types)
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the decision matrix to a pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame representation of the decision matrix
        """
        if self.fuzzy_type:
            # Convert fuzzy sets to their string representation
            display_matrix = np.array([[str(x) for x in row] for row in self.fuzzy_matrix])
        else:
            display_matrix = self.matrix
            
        return pd.DataFrame(display_matrix, 
                          index=self.alternatives,
                          columns=self.criteria)
    
    def get_benefit_criteria(self) -> List[int]:
        """
        Get indices of benefit criteria.
        
        Returns:
            List[int]: List of indices for benefit criteria
        """
        return [i for i, t in enumerate(self.criteria_types) if t.lower() == 'benefit']
    
    def get_cost_criteria(self) -> List[int]:
        """
        Get indices of cost criteria.
        
        Returns:
            List[int]: List of indices for cost criteria
        """
        return [i for i, t in enumerate(self.criteria_types) if t.lower() == 'cost']
    
    def __str__(self) -> str:
        """String representation of the decision matrix."""
        fuzzy_info = f", fuzzy_type={self.fuzzy_type}" if self.fuzzy_type else ""
        return f"DecisionMatrix(alternatives={len(self.alternatives)}, criteria={len(self.criteria)}{fuzzy_info})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the decision matrix."""
        fuzzy_info = f", fuzzy_type={self.fuzzy_type}" if self.fuzzy_type else ""
        return f"DecisionMatrix(decision_matrix={self.matrix.shape}, alternatives={self.alternatives}, criteria={self.criteria}{fuzzy_info})"