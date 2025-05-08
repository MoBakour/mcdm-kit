"""
Fermatean Fuzzy Set implementation.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from .base import BaseFuzzySet

class FermateanFuzzySet(BaseFuzzySet):
    """
    Fermatean Fuzzy Set implementation.
    
    A Fermatean Fuzzy Set (FFS) is characterized by two functions:
    - Membership degree (μ)
    - Non-membership degree (ν)
    
    These degrees satisfy: μ³ + ν³ ≤ 1
    The hesitation degree (π) is calculated as: π = (1 - (μ³ + ν³))^(1/3)
    """
    
    def __init__(self, 
                 membership: float,
                 non_membership: float):
        """
        Initialize a Fermatean Fuzzy Set.
        
        Args:
            membership (float): Degree of membership (μ)
            non_membership (float): Degree of non-membership (ν)
        """
        super().__init__(membership, non_membership)
        self.hesitation = (1 - (membership**3 + non_membership**3))**(1/3)
        
    def validate(self) -> bool:
        """
        Validate the Fermatean Fuzzy Set values.
        
        Returns:
            bool: True if valid, False otherwise
        """
        if not super().validate():
            return False
            
        if not 0 <= self.membership**3 + self.non_membership**3 <= 1:
            return False
            
        return True
        
    def complement(self) -> 'FermateanFuzzySet':
        """
        Calculate the complement of the Fermatean Fuzzy Set.
        
        Returns:
            FermateanFuzzySet: Complement of the fuzzy set
        """
        return FermateanFuzzySet(
            membership=self.non_membership,
            non_membership=self.membership
        )
        
    def distance(self, other: 'FermateanFuzzySet') -> float:
        """
        Calculate the distance between two Fermatean Fuzzy Sets.
        
        Args:
            other (FermateanFuzzySet): Another Fermatean Fuzzy Set
            
        Returns:
            float: Distance between the fuzzy sets
        """
        return (1/2) * (
            abs(self.membership**3 - other.membership**3) +
            abs(self.non_membership**3 - other.non_membership**3) +
            abs(self.hesitation**3 - other.hesitation**3)
        )
        
    def score(self) -> float:
        """
        Calculate the score of the Fermatean Fuzzy Set.
        
        Returns:
            float: Score value
        """
        return self.membership - self.non_membership
        
    def accuracy(self) -> float:
        """
        Calculate the accuracy of the Fermatean Fuzzy Set.
        
        Returns:
            float: Accuracy value
        """
        return self.membership + self.non_membership
        
    def __add__(self, other: 'FermateanFuzzySet') -> 'FermateanFuzzySet':
        """
        Add two Fermatean Fuzzy Sets.
        
        Args:
            other (FermateanFuzzySet): Another Fermatean Fuzzy Set
            
        Returns:
            FermateanFuzzySet: Result of addition
        """
        mu = (self.membership**3 + other.membership**3 - self.membership**3 * other.membership**3)**(1/3)
        nu = self.non_membership * other.non_membership
        return FermateanFuzzySet(mu, nu)
        
    def __mul__(self, other: Union['FermateanFuzzySet', float]) -> 'FermateanFuzzySet':
        """
        Multiply Fermatean Fuzzy Set by another Fermatean Fuzzy Set or scalar.
        
        Args:
            other (Union[FermateanFuzzySet, float]): Another Fermatean Fuzzy Set or scalar
            
        Returns:
            FermateanFuzzySet: Result of multiplication
        """
        if isinstance(other, (int, float)):
            mu = self.membership**other
            nu = (1 - (1 - self.non_membership**3)**other)**(1/3)
            return FermateanFuzzySet(mu, nu)
        else:
            mu = self.membership * other.membership
            nu = (1 - (1 - self.non_membership**3) * (1 - other.non_membership**3))**(1/3)
            return FermateanFuzzySet(mu, nu)
            
    def __eq__(self, other: 'FermateanFuzzySet') -> bool:
        """
        Check equality of two Fermatean Fuzzy Sets.
        
        Args:
            other (FermateanFuzzySet): Another Fermatean Fuzzy Set
            
        Returns:
            bool: True if equal, False otherwise
        """
        return (self.membership == other.membership and
                self.non_membership == other.non_membership)
                
    def to_dict(self) -> Dict[str, float]:
        """
        Convert the Fermatean Fuzzy Set to a dictionary.
        
        Returns:
            Dict[str, float]: Dictionary representation
        """
        return {
            'membership': self.membership,
            'non_membership': self.non_membership,
            'hesitation': self.hesitation
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'FermateanFuzzySet':
        """
        Create a Fermatean Fuzzy Set from a dictionary.
        
        Args:
            data (Dict[str, float]): Dictionary containing fuzzy set values
            
        Returns:
            FermateanFuzzySet: New Fermatean Fuzzy Set instance
        """
        return cls(
            membership=data['membership'],
            non_membership=data['non_membership']
        )
        
    def __str__(self) -> str:
        """String representation of the Fermatean Fuzzy Set."""
        return f"FermateanFuzzySet(μ={self.membership:.3f}, ν={self.non_membership:.3f}, π={self.hesitation:.3f})"
        
    def __repr__(self) -> str:
        """Detailed string representation of the Fermatean Fuzzy Set."""
        return self.__str__() 