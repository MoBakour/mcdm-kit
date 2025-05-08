"""
Hesitant Fuzzy Set implementation.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple, Set
from .base import BaseFuzzySet

class HesitantFuzzySet(BaseFuzzySet):
    """
    Hesitant Fuzzy Set implementation.
    
    A Hesitant Fuzzy Set (HFS) is characterized by a set of membership degrees:
    h = {γ₁, γ₂, ..., γₙ} where γᵢ ∈ [0,1]
    
    The membership degrees represent possible values for the membership of an element.
    """
    
    def __init__(self, membership_degrees: Set[float]):
        """
        Initialize a Hesitant Fuzzy Set.
        
        Args:
            membership_degrees (Set[float]): Set of membership degrees
        """
        self.membership_degrees = sorted(membership_degrees)
        # Use average value for base class initialization
        super().__init__(
            membership=np.mean(self.membership_degrees),
            non_membership=1 - np.mean(self.membership_degrees)
        )
        
    def validate(self) -> bool:
        """
        Validate the Hesitant Fuzzy Set values.
        
        Returns:
            bool: True if valid, False otherwise
        """
        if not self.membership_degrees:
            return False
            
        return all(0 <= degree <= 1 for degree in self.membership_degrees)
        
    def complement(self) -> 'HesitantFuzzySet':
        """
        Calculate the complement of the Hesitant Fuzzy Set.
        
        Returns:
            HesitantFuzzySet: Complement of the fuzzy set
        """
        return HesitantFuzzySet({1 - degree for degree in self.membership_degrees})
        
    def distance(self, other: 'HesitantFuzzySet') -> float:
        """
        Calculate the distance between two Hesitant Fuzzy Sets.
        
        Args:
            other (HesitantFuzzySet): Another Hesitant Fuzzy Set
            
        Returns:
            float: Distance between the fuzzy sets
        """
        # Pad the shorter set with its minimum/maximum values
        max_len = max(len(self.membership_degrees), len(other.membership_degrees))
        h1 = self.membership_degrees + [self.membership_degrees[-1]] * (max_len - len(self.membership_degrees))
        h2 = other.membership_degrees + [other.membership_degrees[-1]] * (max_len - len(other.membership_degrees))
        
        return (1/max_len) * sum(abs(a - b) for a, b in zip(h1, h2))
        
    def score(self) -> float:
        """
        Calculate the score of the Hesitant Fuzzy Set.
        
        Returns:
            float: Score value
        """
        return np.mean(self.membership_degrees)
        
    def accuracy(self) -> float:
        """
        Calculate the accuracy of the Hesitant Fuzzy Set.
        
        Returns:
            float: Accuracy value
        """
        return np.std(self.membership_degrees)
        
    def __add__(self, other: 'HesitantFuzzySet') -> 'HesitantFuzzySet':
        """
        Add two Hesitant Fuzzy Sets.
        
        Args:
            other (HesitantFuzzySet): Another Hesitant Fuzzy Set
            
        Returns:
            HesitantFuzzySet: Result of addition
        """
        result = set()
        for a in self.membership_degrees:
            for b in other.membership_degrees:
                result.add(a + b - a * b)
        return HesitantFuzzySet(result)
        
    def __mul__(self, other: Union['HesitantFuzzySet', float]) -> 'HesitantFuzzySet':
        """
        Multiply Hesitant Fuzzy Set by another Hesitant Fuzzy Set or scalar.
        
        Args:
            other (Union[HesitantFuzzySet, float]): Another Hesitant Fuzzy Set or scalar
            
        Returns:
            HesitantFuzzySet: Result of multiplication
        """
        if isinstance(other, (int, float)):
            result = {degree ** other for degree in self.membership_degrees}
            return HesitantFuzzySet(result)
        else:
            result = set()
            for a in self.membership_degrees:
                for b in other.membership_degrees:
                    result.add(a * b)
            return HesitantFuzzySet(result)
            
    def __eq__(self, other: 'HesitantFuzzySet') -> bool:
        """
        Check equality of two Hesitant Fuzzy Sets.
        
        Args:
            other (HesitantFuzzySet): Another Hesitant Fuzzy Set
            
        Returns:
            bool: True if equal, False otherwise
        """
        return self.membership_degrees == other.membership_degrees
                
    def to_dict(self) -> Dict[str, List[float]]:
        """
        Convert the Hesitant Fuzzy Set to a dictionary.
        
        Returns:
            Dict[str, List[float]]: Dictionary representation
        """
        return {
            'membership_degrees': list(self.membership_degrees)
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, List[float]]) -> 'HesitantFuzzySet':
        """
        Create a Hesitant Fuzzy Set from a dictionary.
        
        Args:
            data (Dict[str, List[float]]): Dictionary containing fuzzy set values
            
        Returns:
            HesitantFuzzySet: New Hesitant Fuzzy Set instance
        """
        return cls(set(data['membership_degrees']))
        
    def __str__(self) -> str:
        """String representation of the Hesitant Fuzzy Set."""
        return f"HesitantFuzzySet(h={self.membership_degrees})"
        
    def __repr__(self) -> str:
        """Detailed string representation of the Hesitant Fuzzy Set."""
        return self.__str__() 