"""
Intuitionistic Fuzzy Set implementation.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from .base import BaseFuzzySet

class IntuitionisticFuzzySet(BaseFuzzySet):
    """
    Intuitionistic Fuzzy Set implementation.
    
    An Intuitionistic Fuzzy Set (IFS) is characterized by two functions:
    - Membership degree (μ)
    - Non-membership degree (ν)
    
    These degrees satisfy: 0 ≤ μ + ν ≤ 1
    The hesitation degree (π) is calculated as: π = 1 - (μ + ν)
    """
    
    def __init__(self, 
                 membership: float,
                 non_membership: float):
        """
        Initialize an Intuitionistic Fuzzy Set.
        
        Args:
            membership (float): Degree of membership (μ)
            non_membership (float): Degree of non-membership (ν)
        """
        super().__init__(membership, non_membership)
        self.hesitation = 1 - (membership + non_membership)
        
    def validate(self) -> bool:
        """
        Validate the Intuitionistic Fuzzy Set values.
        
        Returns:
            bool: True if valid, False otherwise
        """
        if not super().validate():
            return False
            
        if not 0 <= self.membership + self.non_membership <= 1:
            return False
            
        return True
        
    def complement(self) -> 'IntuitionisticFuzzySet':
        """
        Calculate the complement of the Intuitionistic Fuzzy Set.
        
        Returns:
            IntuitionisticFuzzySet: Complement of the fuzzy set
        """
        return IntuitionisticFuzzySet(
            membership=self.non_membership,
            non_membership=self.membership
        )
        
    def distance(self, other: 'IntuitionisticFuzzySet') -> float:
        """
        Calculate the distance between two Intuitionistic Fuzzy Sets.
        
        Args:
            other (IntuitionisticFuzzySet): Another Intuitionistic Fuzzy Set
            
        Returns:
            float: Distance between the fuzzy sets
        """
        return (1/2) * (
            abs(self.membership - other.membership) +
            abs(self.non_membership - other.non_membership) +
            abs(self.hesitation - other.hesitation)
        )
        
    def score(self) -> float:
        """
        Calculate the score of the Intuitionistic Fuzzy Set.
        
        Returns:
            float: Score value
        """
        return self.membership - self.non_membership
        
    def accuracy(self) -> float:
        """
        Calculate the accuracy of the Intuitionistic Fuzzy Set.
        
        Returns:
            float: Accuracy value
        """
        return self.hesitation
        
    def __add__(self, other: 'IntuitionisticFuzzySet') -> 'IntuitionisticFuzzySet':
        """
        Add two Intuitionistic Fuzzy Sets.
        
        Args:
            other (IntuitionisticFuzzySet): Another Intuitionistic Fuzzy Set
            
        Returns:
            IntuitionisticFuzzySet: Result of addition
        """
        return IntuitionisticFuzzySet(
            membership=self.membership + other.membership - self.membership * other.membership,
            non_membership=self.non_membership * other.non_membership
        )
        
    def __mul__(self, other: Union['IntuitionisticFuzzySet', float]) -> 'IntuitionisticFuzzySet':
        """
        Multiply Intuitionistic Fuzzy Set by another Intuitionistic Fuzzy Set or scalar.
        
        Args:
            other (Union[IntuitionisticFuzzySet, float]): Another Intuitionistic Fuzzy Set or scalar
            
        Returns:
            IntuitionisticFuzzySet: Result of multiplication
        """
        if isinstance(other, (int, float)):
            return IntuitionisticFuzzySet(
                membership=1 - (1 - self.membership) ** other,
                non_membership=self.non_membership ** other
            )
        else:
            return IntuitionisticFuzzySet(
                membership=self.membership * other.membership,
                non_membership=1 - (1 - self.non_membership) * (1 - other.non_membership)
            )
            
    def __eq__(self, other: 'IntuitionisticFuzzySet') -> bool:
        """
        Check equality of two Intuitionistic Fuzzy Sets.
        
        Args:
            other (IntuitionisticFuzzySet): Another Intuitionistic Fuzzy Set
            
        Returns:
            bool: True if equal, False otherwise
        """
        return (self.membership == other.membership and
                self.non_membership == other.non_membership)
                
    def to_dict(self) -> Dict[str, float]:
        """
        Convert the Intuitionistic Fuzzy Set to a dictionary.
        
        Returns:
            Dict[str, float]: Dictionary representation
        """
        return {
            'membership': self.membership,
            'non_membership': self.non_membership,
            'hesitation': self.hesitation
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'IntuitionisticFuzzySet':
        """
        Create an Intuitionistic Fuzzy Set from a dictionary.
        
        Args:
            data (Dict[str, float]): Dictionary containing fuzzy set values
            
        Returns:
            IntuitionisticFuzzySet: New Intuitionistic Fuzzy Set instance
        """
        return cls(
            membership=data['membership'],
            non_membership=data['non_membership']
        )
        
    def __str__(self) -> str:
        """String representation of the Intuitionistic Fuzzy Set."""
        return f"IntuitionisticFuzzySet(μ={self.membership:.3f}, ν={self.non_membership:.3f}, π={self.hesitation:.3f})"
        
    def __repr__(self) -> str:
        """Detailed string representation of the Intuitionistic Fuzzy Set."""
        return self.__str__() 