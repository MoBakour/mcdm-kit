"""
Pythagorean Fuzzy Set implementation.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from .base import BaseFuzzySet

class PythagoreanFuzzySet(BaseFuzzySet):
    """
    Pythagorean Fuzzy Set implementation.
    
    A Pythagorean Fuzzy Set (PFS) is characterized by two functions:
    - Membership degree (μ)
    - Non-membership degree (ν)
    
    These degrees satisfy: μ² + ν² ≤ 1
    The hesitation degree (π) is calculated as: π = √(1 - (μ² + ν²))
    """
    
    def __init__(self, 
                 membership: float,
                 non_membership: float):
        """
        Initialize a Pythagorean Fuzzy Set.
        
        Args:
            membership (float): Degree of membership (μ)
            non_membership (float): Degree of non-membership (ν)
        """
        super().__init__(membership, non_membership)
        self.hesitation = np.sqrt(1 - (membership**2 + non_membership**2))
        
    def validate(self) -> bool:
        """
        Validate the Pythagorean Fuzzy Set values.
        
        Returns:
            bool: True if valid, False otherwise
        """
        if not super().validate():
            return False
            
        if not 0 <= self.membership**2 + self.non_membership**2 <= 1:
            return False
            
        return True
        
    def complement(self) -> 'PythagoreanFuzzySet':
        """
        Calculate the complement of the Pythagorean Fuzzy Set.
        
        Returns:
            PythagoreanFuzzySet: Complement of the fuzzy set
        """
        return PythagoreanFuzzySet(
            membership=self.non_membership,
            non_membership=self.membership
        )
        
    def distance(self, other: 'PythagoreanFuzzySet') -> float:
        """
        Calculate the distance between two Pythagorean Fuzzy Sets.
        
        Args:
            other (PythagoreanFuzzySet): Another Pythagorean Fuzzy Set
            
        Returns:
            float: Distance between the fuzzy sets
        """
        return (1/2) * (
            abs(self.membership**2 - other.membership**2) +
            abs(self.non_membership**2 - other.non_membership**2) +
            abs(self.hesitation**2 - other.hesitation**2)
        )
        
    def score(self) -> float:
        """
        Calculate the score of the Pythagorean Fuzzy Set.
        
        Returns:
            float: Score value
        """
        return self.membership - self.non_membership
        
    def accuracy(self) -> float:
        """
        Calculate the accuracy of the Pythagorean Fuzzy Set.
        
        Returns:
            float: Accuracy value
        """
        return self.membership + self.non_membership
        
    def __add__(self, other: 'PythagoreanFuzzySet') -> 'PythagoreanFuzzySet':
        """
        Add two Pythagorean Fuzzy Sets.
        
        Args:
            other (PythagoreanFuzzySet): Another Pythagorean Fuzzy Set
            
        Returns:
            PythagoreanFuzzySet: Result of addition
        """
        mu = np.sqrt(self.membership**2 + other.membership**2 - self.membership**2 * other.membership**2)
        nu = self.non_membership * other.non_membership
        return PythagoreanFuzzySet(mu, nu)
        
    def __mul__(self, other: Union['PythagoreanFuzzySet', float]) -> 'PythagoreanFuzzySet':
        """
        Multiply Pythagorean Fuzzy Set by another Pythagorean Fuzzy Set or scalar.
        
        Args:
            other (Union[PythagoreanFuzzySet, float]): Another Pythagorean Fuzzy Set or scalar
            
        Returns:
            PythagoreanFuzzySet: Result of multiplication
        """
        if isinstance(other, (int, float)):
            mu = self.membership**other
            nu = np.sqrt(1 - (1 - self.non_membership**2)**other)
            return PythagoreanFuzzySet(mu, nu)
        else:
            mu = self.membership * other.membership
            nu = np.sqrt(1 - (1 - self.non_membership**2) * (1 - other.non_membership**2))
            return PythagoreanFuzzySet(mu, nu)
            
    def __eq__(self, other: 'PythagoreanFuzzySet') -> bool:
        """
        Check equality of two Pythagorean Fuzzy Sets.
        
        Args:
            other (PythagoreanFuzzySet): Another Pythagorean Fuzzy Set
            
        Returns:
            bool: True if equal, False otherwise
        """
        return (self.membership == other.membership and
                self.non_membership == other.non_membership)
                
    def to_dict(self) -> Dict[str, float]:
        """
        Convert the Pythagorean Fuzzy Set to a dictionary.
        
        Returns:
            Dict[str, float]: Dictionary representation
        """
        return {
            'membership': self.membership,
            'non_membership': self.non_membership,
            'hesitation': self.hesitation
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'PythagoreanFuzzySet':
        """
        Create a Pythagorean Fuzzy Set from a dictionary.
        
        Args:
            data (Dict[str, float]): Dictionary containing fuzzy set values
            
        Returns:
            PythagoreanFuzzySet: New Pythagorean Fuzzy Set instance
        """
        return cls(
            membership=data['membership'],
            non_membership=data['non_membership']
        )
        
    def __str__(self) -> str:
        """String representation of the Pythagorean Fuzzy Set."""
        return f"PythagoreanFuzzySet(μ={self.membership:.3f}, ν={self.non_membership:.3f}, π={self.hesitation:.3f})"
        
    def __repr__(self) -> str:
        """Detailed string representation of the Pythagorean Fuzzy Set."""
        return self.__str__() 