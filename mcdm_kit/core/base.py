"""
Base class for all MCDM methods.
This class defines the interface that all MCDM methods must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
import numpy as np
from ..data.decision_matrix import DecisionMatrix

class BaseMCDMMethod(ABC):
    """Base class for all MCDM methods."""
    
    def __init__(self, decision_matrix: DecisionMatrix):
        """
        Initialize the MCDM method with a decision matrix.
        
        Args:
            decision_matrix (DecisionMatrix): The decision matrix containing alternatives and criteria.
        """
        self.decision_matrix = decision_matrix
        self.weights = None
        self.rankings = None
        
    @abstractmethod
    def calculate_weights(self) -> np.ndarray:
        """
        Calculate the weights for each criterion.
        
        Returns:
            np.ndarray: Array of weights for each criterion.
        """
        pass
    
    @abstractmethod
    def normalize_matrix(self) -> np.ndarray:
        """
        Normalize the decision matrix.
        
        Returns:
            np.ndarray: Normalized decision matrix.
        """
        pass
    
    @abstractmethod
    def calculate_scores(self) -> np.ndarray:
        """
        Calculate the scores for each alternative.
        
        Returns:
            np.ndarray: Array of scores for each alternative.
        """
        pass
    
    @abstractmethod
    def rank(self) -> Dict[str, Any]:
        """
        Rank the alternatives based on their scores.
        
        Returns:
            Dict[str, Any]: Dictionary containing rankings and scores.
        """
        pass
    
    def validate_inputs(self) -> bool:
        """
        Validate the inputs to ensure they are suitable for the method.
        
        Returns:
            bool: True if inputs are valid, False otherwise.
        """
        if self.decision_matrix is None:
            raise ValueError("Decision matrix is required")
        if self.decision_matrix.matrix is None or self.decision_matrix.matrix.size == 0:
            raise ValueError("Decision matrix is empty")
        return True 