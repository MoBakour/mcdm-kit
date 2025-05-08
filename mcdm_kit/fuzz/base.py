"""
Base class for fuzzy set implementations.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple

class BaseFuzzySet:
    """
    Base class for fuzzy set implementations.
    
    This class provides the foundation for all fuzzy set types and defines
    common operations and properties that all fuzzy sets should have.
    """
    
    def __init__(self, 
                 membership: float,
                 non_membership: Optional[float] = None,
                 hesitation: Optional[float] = None):
        """
        Initialize a fuzzy set.
        
        Args:
            membership (float): Degree of membership (μ)
            non_membership (Optional[float]): Degree of non-membership (ν)
            hesitation (Optional[float]): Degree of hesitation (π)
        """
        self.membership = membership
        self.non_membership = non_membership
        self.hesitation = hesitation
        
    def validate(self) -> bool:
        """
        Validate the fuzzy set values.
        
        Returns:
            bool: True if the fuzzy set is valid, False otherwise
        """
        if not 0 <= self.membership <= 1:
            return False
            
        if self.non_membership is not None:
            if not 0 <= self.non_membership <= 1:
                return False
                
        if self.hesitation is not None:
            if not 0 <= self.hesitation <= 1:
                return False
                
        return True
        
    def complement(self) -> 'BaseFuzzySet':
        """
        Calculate the complement of the fuzzy set.
        
        Returns:
            BaseFuzzySet: Complement of the fuzzy set
        """
        return BaseFuzzySet(
            membership=1 - self.membership,
            non_membership=self.non_membership,
            hesitation=self.hesitation
        )
        
    def distance(self, other: 'BaseFuzzySet') -> float:
        """
        Calculate the distance between two fuzzy sets.
        
        Args:
            other (BaseFuzzySet): Another fuzzy set
            
        Returns:
            float: Distance between the fuzzy sets
        """
        return abs(self.membership - other.membership)
        
    def similarity(self, other: 'BaseFuzzySet') -> float:
        """
        Calculate the similarity between two fuzzy sets.
        
        Args:
            other (BaseFuzzySet): Another fuzzy set
            
        Returns:
            float: Similarity between the fuzzy sets
        """
        return 1 - self.distance(other)
        
    def to_dict(self) -> Dict[str, float]:
        """
        Convert the fuzzy set to a dictionary.
        
        Returns:
            Dict[str, float]: Dictionary representation of the fuzzy set
        """
        result = {'membership': self.membership}
        if self.non_membership is not None:
            result['non_membership'] = self.non_membership
        if self.hesitation is not None:
            result['hesitation'] = self.hesitation
        return result
        
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'BaseFuzzySet':
        """
        Create a fuzzy set from a dictionary.
        
        Args:
            data (Dict[str, float]): Dictionary containing fuzzy set values
            
        Returns:
            BaseFuzzySet: New fuzzy set instance
        """
        return cls(
            membership=data['membership'],
            non_membership=data.get('non_membership'),
            hesitation=data.get('hesitation')
        )
        
    def __str__(self) -> str:
        """String representation of the fuzzy set."""
        return f"FuzzySet(μ={self.membership:.3f}, ν={self.non_membership:.3f if self.non_membership is not None else None}, π={self.hesitation:.3f if self.hesitation is not None else None})"
        
    def __repr__(self) -> str:
        """Detailed string representation of the fuzzy set."""
        return self.__str__() 