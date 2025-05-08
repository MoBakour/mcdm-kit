"""
Neutrosophic Set implementation.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from .base import BaseFuzzySet

class NeutrosophicSet(BaseFuzzySet):
    """
    Neutrosophic Set implementation.
    
    A Neutrosophic Set (NS) is characterized by three functions:
    - Truth-membership degree (T)
    - Indeterminacy-membership degree (I)
    - Falsity-membership degree (F)
    
    These degrees are independent and can be any value in [0,1]
    """
    
    def __init__(self, 
                 truth: float,
                 indeterminacy: float,
                 falsity: float):
        """
        Initialize a Neutrosophic Set.
        
        Args:
            truth (float): Truth-membership degree (T)
            indeterminacy (float): Indeterminacy-membership degree (I)
            falsity (float): Falsity-membership degree (F)
        """
        self.truth = truth
        self.indeterminacy = indeterminacy
        self.falsity = falsity
        super().__init__(truth, falsity)
        
    def validate(self) -> bool:
        """
        Validate the Neutrosophic Set values.
        
        Returns:
            bool: True if valid, False otherwise
        """
        if not super().validate():
            return False
            
        if not 0 <= self.indeterminacy <= 1:
            return False
            
        return True
        
    def complement(self) -> 'NeutrosophicSet':
        """
        Calculate the complement of the Neutrosophic Set.
        
        Returns:
            NeutrosophicSet: Complement of the fuzzy set
        """
        return NeutrosophicSet(
            truth=self.falsity,
            indeterminacy=self.indeterminacy,
            falsity=self.truth
        )
        
    def distance(self, other: 'NeutrosophicSet') -> float:
        """
        Calculate the distance between two Neutrosophic Sets.
        
        Args:
            other (NeutrosophicSet): Another Neutrosophic Set
            
        Returns:
            float: Distance between the fuzzy sets
        """
        return (1/3) * (
            abs(self.membership - other.membership) +
            abs(self.indeterminacy - other.indeterminacy) +
            abs(self.non_membership - other.non_membership)
        )
        
    def score(self) -> float:
        """
        Calculate the score of the Neutrosophic Set.
        
        Returns:
            float: Score value
        """
        return self.truth
        
    def accuracy(self) -> float:
        """
        Calculate the accuracy of the Neutrosophic Set.
        
        Returns:
            float: Accuracy value
        """
        return self.indeterminacy
        
    def __add__(self, other: 'NeutrosophicSet') -> 'NeutrosophicSet':
        """
        Add two Neutrosophic Sets.
        
        Args:
            other (NeutrosophicSet): Another Neutrosophic Set
            
        Returns:
            NeutrosophicSet: Result of addition
        """
        return NeutrosophicSet(
            truth=self.membership + other.membership - self.membership * other.membership,
            indeterminacy=self.indeterminacy * other.indeterminacy,
            falsity=self.non_membership * other.non_membership
        )
        
    def __mul__(self, other: Union['NeutrosophicSet', float]) -> 'NeutrosophicSet':
        """
        Multiply Neutrosophic Set by another Neutrosophic Set or scalar.
        
        Args:
            other (Union[NeutrosophicSet, float]): Another Neutrosophic Set or scalar
            
        Returns:
            NeutrosophicSet: Result of multiplication
        """
        if isinstance(other, (int, float)):
            return NeutrosophicSet(
                truth=1 - (1 - self.membership) ** other,
                indeterminacy=self.indeterminacy ** other,
                falsity=self.non_membership ** other
            )
        else:
            return NeutrosophicSet(
                truth=self.membership * other.membership,
                indeterminacy=1 - (1 - self.indeterminacy) * (1 - other.indeterminacy),
                falsity=1 - (1 - self.non_membership) * (1 - other.non_membership)
            )
            
    def __eq__(self, other: 'NeutrosophicSet') -> bool:
        """
        Check equality of two Neutrosophic Sets.
        
        Args:
            other (NeutrosophicSet): Another Neutrosophic Set
            
        Returns:
            bool: True if equal, False otherwise
        """
        return (self.membership == other.membership and
                self.indeterminacy == other.indeterminacy and
                self.non_membership == other.non_membership)
                
    def to_dict(self) -> Dict[str, float]:
        """
        Convert the Neutrosophic Set to a dictionary.
        
        Returns:
            Dict[str, float]: Dictionary representation
        """
        return {
            'truth': self.membership,
            'indeterminacy': self.indeterminacy,
            'falsity': self.non_membership
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'NeutrosophicSet':
        """
        Create a Neutrosophic Set from a dictionary.
        
        Args:
            data (Dict[str, float]): Dictionary containing fuzzy set values
            
        Returns:
            NeutrosophicSet: New Neutrosophic Set instance
        """
        return cls(
            truth=data['truth'],
            indeterminacy=data['indeterminacy'],
            falsity=data['falsity']
        )
        
    def __str__(self) -> str:
        """String representation of the Neutrosophic Set."""
        return f"NeutrosophicSet(T={self.membership:.3f}, I={self.indeterminacy:.3f}, F={self.non_membership:.3f})"
        
    def __repr__(self) -> str:
        """Detailed string representation of the Neutrosophic Set."""
        return self.__str__() 