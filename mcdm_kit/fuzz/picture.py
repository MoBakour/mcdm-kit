"""
Picture Fuzzy Set implementation.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from .base import BaseFuzzySet

class PictureFuzzySet(BaseFuzzySet):
    """
    Picture Fuzzy Set implementation.
    
    A Picture Fuzzy Set (PFS) is characterized by three functions:
    - Membership degree (μ)
    - Neutrality degree (η)
    - Non-membership degree (ν)
    
    These degrees satisfy: 0 ≤ μ + η + ν ≤ 1
    """
    
    def __init__(self, 
                 membership: float,
                 neutrality: float,
                 non_membership: float):
        """
        Initialize a Picture Fuzzy Set.
        
        Args:
            membership (float): Degree of membership (μ)
            neutrality (float): Degree of neutrality (η)
            non_membership (float): Degree of non-membership (ν)
        """
        super().__init__(membership, non_membership)
        self.neutrality = neutrality
        self.hesitation = 1 - (membership + neutrality + non_membership)
        
    def validate(self) -> bool:
        """
        Validate the Picture Fuzzy Set values.
        
        Returns:
            bool: True if valid, False otherwise
        """
        if not super().validate():
            return False
            
        if not 0 <= self.neutrality <= 1:
            return False
            
        if not 0 <= self.membership + self.neutrality + self.non_membership <= 1:
            return False
            
        return True
        
    def complement(self) -> 'PictureFuzzySet':
        """
        Calculate the complement of the Picture Fuzzy Set.
        
        Returns:
            PictureFuzzySet: Complement of the fuzzy set
        """
        return PictureFuzzySet(
            membership=self.non_membership,
            neutrality=self.neutrality,
            non_membership=self.membership
        )
        
    def distance(self, other: 'PictureFuzzySet') -> float:
        """
        Calculate the distance between two Picture Fuzzy Sets.
        
        Args:
            other (PictureFuzzySet): Another Picture Fuzzy Set
            
        Returns:
            float: Distance between the fuzzy sets
        """
        return (1/4) * (
            abs(self.membership - other.membership) +
            abs(self.neutrality - other.neutrality) +
            abs(self.non_membership - other.non_membership) +
            abs(self.hesitation - other.hesitation)
        )
        
    def score(self) -> float:
        """
        Calculate the score of the Picture Fuzzy Set.
        
        Returns:
            float: Score value
        """
        return self.membership
        
    def accuracy(self) -> float:
        """
        Calculate the accuracy of the Picture Fuzzy Set.
        
        Returns:
            float: Accuracy value
        """
        return self.neutrality
        
    def __add__(self, other: 'PictureFuzzySet') -> 'PictureFuzzySet':
        """
        Add two Picture Fuzzy Sets.
        
        Args:
            other (PictureFuzzySet): Another Picture Fuzzy Set
            
        Returns:
            PictureFuzzySet: Result of addition
        """
        return PictureFuzzySet(
            membership=self.membership + other.membership - self.membership * other.membership,
            neutrality=self.neutrality * other.neutrality,
            non_membership=self.non_membership * other.non_membership
        )
        
    def __mul__(self, other: Union['PictureFuzzySet', float]) -> 'PictureFuzzySet':
        """
        Multiply Picture Fuzzy Set by another Picture Fuzzy Set or scalar.
        
        Args:
            other (Union[PictureFuzzySet, float]): Another Picture Fuzzy Set or scalar
            
        Returns:
            PictureFuzzySet: Result of multiplication
        """
        if isinstance(other, (int, float)):
            return PictureFuzzySet(
                membership=1 - (1 - self.membership) ** other,
                neutrality=self.neutrality ** other,
                non_membership=self.non_membership ** other
            )
        else:
            return PictureFuzzySet(
                membership=self.membership * other.membership,
                neutrality=1 - (1 - self.neutrality) * (1 - other.neutrality),
                non_membership=1 - (1 - self.non_membership) * (1 - other.non_membership)
            )
            
    def __eq__(self, other: 'PictureFuzzySet') -> bool:
        """
        Check equality of two Picture Fuzzy Sets.
        
        Args:
            other (PictureFuzzySet): Another Picture Fuzzy Set
            
        Returns:
            bool: True if equal, False otherwise
        """
        return (self.membership == other.membership and
                self.neutrality == other.neutrality and
                self.non_membership == other.non_membership)
                
    def to_dict(self) -> Dict[str, float]:
        """
        Convert the Picture Fuzzy Set to a dictionary.
        
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
    def from_dict(cls, data: Dict[str, float]) -> 'PictureFuzzySet':
        """
        Create a Picture Fuzzy Set from a dictionary.
        
        Args:
            data (Dict[str, float]): Dictionary containing fuzzy set values
            
        Returns:
            PictureFuzzySet: New Picture Fuzzy Set instance
        """
        return cls(
            membership=data['membership'],
            neutrality=data['neutrality'],
            non_membership=data['non_membership']
        )
        
    def __str__(self) -> str:
        """String representation of the Picture Fuzzy Set."""
        return f"PictureFuzzySet(μ={self.membership:.3f}, η={self.neutrality:.3f}, ν={self.non_membership:.3f}, π={self.hesitation:.3f})"
        
    def __repr__(self) -> str:
        """Detailed string representation of the Picture Fuzzy Set."""
        return self.__str__() 