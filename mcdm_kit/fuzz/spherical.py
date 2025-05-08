"""
Spherical Fuzzy Set implementation.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from .base import BaseFuzzySet

class SphericalFuzzySet(BaseFuzzySet):
    """
    Spherical Fuzzy Set implementation.
    
    A Spherical Fuzzy Set (SFS) is characterized by three functions:
    - Membership degree (μ)
    - Neutrality degree (η)
    - Non-membership degree (ν)
    
    These degrees satisfy: μ² + η² + ν² ≤ 1
    """
    
    def __init__(self, 
                 membership: float,
                 neutrality: float,
                 non_membership: float):
        """
        Initialize a Spherical Fuzzy Set.
        
        Args:
            membership (float): Degree of membership (μ)
            neutrality (float): Degree of neutrality (η)
            non_membership (float): Degree of non-membership (ν)
        """
        super().__init__(membership, non_membership)
        self.neutrality = neutrality
        self.hesitation = np.sqrt(1 - (membership**2 + neutrality**2 + non_membership**2))
        
    def validate(self) -> bool:
        """
        Validate the Spherical Fuzzy Set values.
        
        Returns:
            bool: True if valid, False otherwise
        """
        if not super().validate():
            return False
            
        if not 0 <= self.neutrality <= 1:
            return False
            
        if not 0 <= self.membership**2 + self.neutrality**2 + self.non_membership**2 <= 1:
            return False
            
        return True
        
    def complement(self) -> 'SphericalFuzzySet':
        """
        Calculate the complement of the Spherical Fuzzy Set.
        
        Returns:
            SphericalFuzzySet: Complement of the fuzzy set
        """
        return SphericalFuzzySet(
            membership=self.non_membership,
            neutrality=self.neutrality,
            non_membership=self.membership
        )
        
    def distance(self, other: 'SphericalFuzzySet') -> float:
        """
        Calculate the distance between two Spherical Fuzzy Sets.
        
        Args:
            other (SphericalFuzzySet): Another Spherical Fuzzy Set
            
        Returns:
            float: Distance between the fuzzy sets
        """
        return (1/2) * (
            abs(self.membership**2 - other.membership**2) +
            abs(self.neutrality**2 - other.neutrality**2) +
            abs(self.non_membership**2 - other.non_membership**2)
        )
        
    def score(self) -> float:
        """
        Calculate the score of the Spherical Fuzzy Set.
        
        Returns:
            float: Score value
        """
        return self.membership
        
    def accuracy(self) -> float:
        """
        Calculate the accuracy of the Spherical Fuzzy Set.
        
        Returns:
            float: Accuracy value
        """
        return self.membership + self.non_membership
        
    def __add__(self, other: 'SphericalFuzzySet') -> 'SphericalFuzzySet':
        """
        Add two Spherical Fuzzy Sets.
        
        Args:
            other (SphericalFuzzySet): Another Spherical Fuzzy Set
            
        Returns:
            SphericalFuzzySet: Result of addition
        """
        mu = np.sqrt(self.membership**2 + other.membership**2 - self.membership**2 * other.membership**2)
        eta = self.neutrality * other.neutrality
        nu = self.non_membership * other.non_membership
        return SphericalFuzzySet(mu, eta, nu)
        
    def __mul__(self, other: Union['SphericalFuzzySet', float]) -> 'SphericalFuzzySet':
        """
        Multiply Spherical Fuzzy Set by another Spherical Fuzzy Set or scalar.
        
        Args:
            other (Union[SphericalFuzzySet, float]): Another Spherical Fuzzy Set or scalar
            
        Returns:
            SphericalFuzzySet: Result of multiplication
        """
        if isinstance(other, (int, float)):
            mu = self.membership**other
            eta = 1 - (1 - self.neutrality**2)**other
            nu = 1 - (1 - self.non_membership**2)**other
            return SphericalFuzzySet(mu, eta, nu)
        else:
            mu = self.membership * other.membership
            eta = np.sqrt(1 - (1 - self.neutrality**2) * (1 - other.neutrality**2))
            nu = np.sqrt(1 - (1 - self.non_membership**2) * (1 - other.non_membership**2))
            return SphericalFuzzySet(mu, eta, nu)
            
    def __eq__(self, other: 'SphericalFuzzySet') -> bool:
        """
        Check equality of two Spherical Fuzzy Sets.
        
        Args:
            other (SphericalFuzzySet): Another Spherical Fuzzy Set
            
        Returns:
            bool: True if equal, False otherwise
        """
        return (self.membership == other.membership and
                self.neutrality == other.neutrality and
                self.non_membership == other.non_membership)
                
    def to_dict(self) -> Dict[str, float]:
        """
        Convert the Spherical Fuzzy Set to a dictionary.
        
        Returns:
            Dict[str, float]: Dictionary representation
        """
        return {
            'membership': self.membership,
            'neutrality': self.neutrality,
            'non_membership': self.non_membership,
            'hesitation': self.hesitation
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'SphericalFuzzySet':
        """
        Create a Spherical Fuzzy Set from a dictionary.
        
        Args:
            data (Dict[str, float]): Dictionary containing fuzzy set values
            
        Returns:
            SphericalFuzzySet: New Spherical Fuzzy Set instance
        """
        return cls(
            membership=data['membership'],
            neutrality=data['neutrality'],
            non_membership=data['non_membership']
        )
        
    def __str__(self) -> str:
        """String representation of the Spherical Fuzzy Set."""
        return f"SphericalFuzzySet(μ={self.membership:.3f}, η={self.neutrality:.3f}, ν={self.non_membership:.3f}, π={self.hesitation:.3f})"
        
    def __repr__(self) -> str:
        """Detailed string representation of the Spherical Fuzzy Set."""
        return self.__str__() 