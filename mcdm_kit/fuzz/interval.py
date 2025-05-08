"""
Interval-Valued Fuzzy Set implementation.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from .base import BaseFuzzySet

class IntervalFuzzySet(BaseFuzzySet):
    """
    Interval-Valued Fuzzy Set implementation.
    
    An Interval-Valued Fuzzy Set (IVFS) is characterized by two intervals:
    - Lower membership degree [μ⁻, μ⁺]
    - Upper membership degree [ν⁻, ν⁺]
    
    where 0 ≤ μ⁻ ≤ μ⁺ ≤ 1 and 0 ≤ ν⁻ ≤ ν⁺ ≤ 1
    """
    
    def __init__(self, 
                 lower_membership: Tuple[float, float],
                 upper_membership: Tuple[float, float]):
        """
        Initialize an Interval-Valued Fuzzy Set.
        
        Args:
            lower_membership (Tuple[float, float]): Lower membership interval [μ⁻, μ⁺]
            upper_membership (Tuple[float, float]): Upper membership interval [ν⁻, ν⁺]
        """
        self.lower_membership = lower_membership
        self.upper_membership = upper_membership
        # Use average values for base class initialization
        super().__init__(
            membership=(lower_membership[0] + lower_membership[1]) / 2,
            non_membership=(upper_membership[0] + upper_membership[1]) / 2
        )
        
    def validate(self) -> bool:
        """
        Validate the Interval-Valued Fuzzy Set values.
        
        Returns:
            bool: True if valid, False otherwise
        """
        # Check lower membership interval
        if not (0 <= self.lower_membership[0] <= self.lower_membership[1] <= 1):
            return False
            
        # Check upper membership interval
        if not (0 <= self.upper_membership[0] <= self.upper_membership[1] <= 1):
            return False
            
        return True
        
    def complement(self) -> 'IntervalFuzzySet':
        """
        Compute the complement of the interval fuzzy set.

        self.lower_membership = (μ⁻, μ⁺)
        self.upper_membership = (ν⁻, ν⁺)

        Returns:
            IntervalFuzzySet with:
            - membership = [1 - μ⁺, 1 - μ⁻]  # Complement of membership
            - non-membership = [1 - ν⁺, 1 - ν⁻]  # Complement of non-membership
        """
        μ_minus, μ_plus = self.lower_membership
        ν_minus, ν_plus = self.upper_membership

        comp_membership = (1 - μ_plus, 1 - μ_minus)    # Complement of membership
        comp_non_membership = (1 - ν_plus, 1 - ν_minus)  # Complement of non-membership

        return IntervalFuzzySet(
            lower_membership=comp_membership,
            upper_membership=comp_non_membership
        )
        
    def distance(self, other: 'IntervalFuzzySet') -> float:
        """
        Calculate the distance between two Interval-Valued Fuzzy Sets.
        
        Args:
            other (IntervalFuzzySet): Another Interval-Valued Fuzzy Set
            
        Returns:
            float: Distance between the fuzzy sets
        """
        return (1/4) * (
            abs(self.lower_membership[0] - other.lower_membership[0]) +
            abs(self.lower_membership[1] - other.lower_membership[1]) +
            abs(self.upper_membership[0] - other.upper_membership[0]) +
            abs(self.upper_membership[1] - other.upper_membership[1])
        )
        
    def score(self) -> float:
        """
        Calculate the score of the Interval-Valued Fuzzy Set.
        
        Returns:
            float: Score value
        """
        return (self.lower_membership[0] + self.lower_membership[1]) / 2
        
    def accuracy(self) -> float:
        """
        Calculate the accuracy of the Interval-Valued Fuzzy Set.
        
        Returns:
            float: Accuracy value
        """
        return (self.lower_membership[1] - self.lower_membership[0])
        
    def __add__(self, other: 'IntervalFuzzySet') -> 'IntervalFuzzySet':
        """
        Add two Interval-Valued Fuzzy Sets.
        
        Args:
            other (IntervalFuzzySet): Another Interval-Valued Fuzzy Set
            
        Returns:
            IntervalFuzzySet: Result of addition
        """
        return IntervalFuzzySet(
            lower_membership=(
                self.lower_membership[0] + other.lower_membership[0] - 
                self.lower_membership[0] * other.lower_membership[0],
                self.lower_membership[1] + other.lower_membership[1] - 
                self.lower_membership[1] * other.lower_membership[1]
            ),
            upper_membership=(
                self.upper_membership[0] * other.upper_membership[0],
                self.upper_membership[1] * other.upper_membership[1]
            )
        )
        
    def __mul__(self, other: Union['IntervalFuzzySet', float]) -> 'IntervalFuzzySet':
        """
        Multiply Interval-Valued Fuzzy Set by another Interval-Valued Fuzzy Set or scalar.
        
        Args:
            other (Union[IntervalFuzzySet, float]): Another Interval-Valued Fuzzy Set or scalar
            
        Returns:
            IntervalFuzzySet: Result of multiplication
        """
        if isinstance(other, (int, float)):
            return IntervalFuzzySet(
                lower_membership=(
                    1 - (1 - self.lower_membership[0]) ** other,
                    1 - (1 - self.lower_membership[1]) ** other
                ),
                upper_membership=(
                    self.upper_membership[0] ** other,
                    self.upper_membership[1] ** other
                )
            )
        else:
            return IntervalFuzzySet(
                lower_membership=(
                    self.lower_membership[0] * other.lower_membership[0],
                    self.lower_membership[1] * other.lower_membership[1]
                ),
                upper_membership=(
                    1 - (1 - self.upper_membership[0]) * (1 - other.upper_membership[0]),
                    1 - (1 - self.upper_membership[1]) * (1 - other.upper_membership[1])
                )
            )
            
    def __eq__(self, other: 'IntervalFuzzySet') -> bool:
        """
        Check equality of two Interval-Valued Fuzzy Sets.
        
        Args:
            other (IntervalFuzzySet): Another Interval-Valued Fuzzy Set
            
        Returns:
            bool: True if equal, False otherwise
        """
        return (np.allclose(self.lower_membership, other.lower_membership) and
                np.allclose(self.upper_membership, other.upper_membership))
                
    def to_dict(self) -> Dict[str, List[float]]:
        """
        Convert the Interval-Valued Fuzzy Set to a dictionary.
        
        Returns:
            Dict[str, List[float]]: Dictionary representation
        """
        return {
            'lower_membership': list(self.lower_membership),
            'upper_membership': list(self.upper_membership)
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, List[float]]) -> 'IntervalFuzzySet':
        """
        Create an Interval-Valued Fuzzy Set from a dictionary.
        
        Args:
            data (Dict[str, List[float]]): Dictionary containing fuzzy set values
            
        Returns:
            IntervalFuzzySet: New Interval-Valued Fuzzy Set instance
        """
        return cls(
            lower_membership=tuple(data['lower_membership']),
            upper_membership=tuple(data['upper_membership'])
        )
        
    def __str__(self) -> str:
        """String representation of the Interval-Valued Fuzzy Set."""
        return (f"IntervalFuzzySet(μ=[{self.lower_membership[0]:.3f}, {self.lower_membership[1]:.3f}], "
                f"ν=[{self.upper_membership[0]:.3f}, {self.upper_membership[1]:.3f}])")
        
    def __repr__(self) -> str:
        """Detailed string representation of the Interval-Valued Fuzzy Set."""
        return self.__str__() 