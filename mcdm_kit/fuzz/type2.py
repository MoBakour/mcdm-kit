"""
Type-2 Fuzzy Set implementation.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from .base import BaseFuzzySet

class Type2FuzzySet(BaseFuzzySet):
    """
    Type-2 Fuzzy Set implementation.
    
    A Type-2 Fuzzy Set (T2FS) is characterized by:
    - Primary membership function (μ)
    - Secondary membership function (f)
    
    The primary membership function maps elements to a set of membership grades,
    and the secondary membership function assigns a weight to each primary membership grade.
    """
    
    def __init__(self, 
                 primary_membership: Callable[[float], List[float]],
                 secondary_membership: Callable[[float, float], float],
                 domain: Tuple[float, float] = (0, 1)):
        """
        Initialize a Type-2 Fuzzy Set.
        
        Args:
            primary_membership (Callable[[float], List[float]]): Function that returns possible membership grades
            secondary_membership (Callable[[float, float], float]): Function that returns secondary membership grade
            domain (Tuple[float, float]): Domain of the fuzzy set (default: [0,1])
        """
        self.primary_membership = primary_membership
        self.secondary_membership = secondary_membership
        self.domain = domain
        
        # Use average values for base class initialization
        x = np.linspace(domain[0], domain[1], 100)
        primary_grades = [np.mean(primary_membership(xi)) for xi in x]
        super().__init__(
            membership=np.mean(primary_grades),
            non_membership=1 - np.mean(primary_grades)
        )
        
    def validate(self) -> bool:
        """
        Validate the Type-2 Fuzzy Set values.
        
        Returns:
            bool: True if valid, False otherwise
        """
        # Check domain
        if not (0 <= self.domain[0] <= self.domain[1] <= 1):
            return False
            
        # Check primary membership function
        x = np.linspace(self.domain[0], self.domain[1], 10)
        for xi in x:
            grades = self.primary_membership(xi)
            if not all(0 <= grade <= 1 for grade in grades):
                return False
                
        # Check secondary membership function
        for xi in x:
            for grade in self.primary_membership(xi):
                if not 0 <= self.secondary_membership(xi, grade) <= 1:
                    return False
                    
        return True
        
    def complement(self) -> 'Type2FuzzySet':
        """
        Calculate the complement of the Type-2 Fuzzy Set.
        
        Returns:
            Type2FuzzySet: Complement of the fuzzy set
        """
        def new_primary(x: float) -> List[float]:
            grades = self.primary_membership(x)
            return sorted([1 - g for g in grades])
            
        def new_secondary(x: float, grade: float) -> float:
            return self.secondary_membership(x, 1 - grade)
            
        return Type2FuzzySet(
            primary_membership=new_primary,
            secondary_membership=new_secondary,
            domain=self.domain
        )
        
    def distance(self, other: 'Type2FuzzySet') -> float:
        """
        Calculate the distance between two Type-2 Fuzzy Sets.
        
        Args:
            other (Type2FuzzySet): Another Type-2 Fuzzy Set
            
        Returns:
            float: Distance between the fuzzy sets
        """
        x = np.linspace(self.domain[0], self.domain[1], 100)
        total_distance = 0
        
        for xi in x:
            # Get primary membership grades
            grades1 = self.primary_membership(xi)
            grades2 = other.primary_membership(xi)
            
            # Calculate secondary membership values
            sec1 = [self.secondary_membership(xi, g) for g in grades1]
            sec2 = [other.secondary_membership(xi, g) for g in grades2]
            
            # Calculate weighted distance
            max_len = max(len(grades1), len(grades2))
            grades1 = grades1 + [grades1[-1]] * (max_len - len(grades1))
            grades2 = grades2 + [grades2[-1]] * (max_len - len(grades2))
            sec1 = sec1 + [sec1[-1]] * (max_len - len(sec1))
            sec2 = sec2 + [sec2[-1]] * (max_len - len(sec2))
            
            distance = sum(abs(g1 - g2) * abs(s1 - s2) 
                         for g1, g2, s1, s2 in zip(grades1, grades2, sec1, sec2))
            total_distance += distance
            
        return total_distance / len(x)
        
    def score(self) -> float:
        """
        Calculate the score of the Type-2 Fuzzy Set.
        
        Returns:
            float: Score value
        """
        x = np.linspace(self.domain[0], self.domain[1], 100)
        total_score = 0
        
        for xi in x:
            grades = self.primary_membership(xi)
            total_score += np.mean(grades)
            
        return total_score / len(x)
        
    def accuracy(self) -> float:
        """
        Calculate the accuracy of the Type-2 Fuzzy Set.
        
        Returns:
            float: Accuracy value
        """
        x = np.linspace(self.domain[0], self.domain[1], 100)
        total_accuracy = 0
        
        for xi in x:
            grades = self.primary_membership(xi)
            total_accuracy += np.std(grades)
            
        return total_accuracy / len(x)
        
    def __add__(self, other: 'Type2FuzzySet') -> 'Type2FuzzySet':
        """
        Add two Type-2 Fuzzy Sets.
        
        Args:
            other (Type2FuzzySet): Another Type-2 Fuzzy Set
            
        Returns:
            Type2FuzzySet: Result of addition
        """
        def new_primary(x: float) -> List[float]:
            grades1 = self.primary_membership(x)
            grades2 = other.primary_membership(x)
            return [g1 + g2 - g1 * g2 for g1 in grades1 for g2 in grades2]
            
        def new_secondary(x: float, grade: float) -> float:
            grades1 = self.primary_membership(x)
            grades2 = other.primary_membership(x)
            sec1 = [self.secondary_membership(x, g) for g in grades1]
            sec2 = [other.secondary_membership(x, g) for g in grades2]
            return np.mean([s1 * s2 for s1 in sec1 for s2 in sec2])
            
        return Type2FuzzySet(
            primary_membership=new_primary,
            secondary_membership=new_secondary,
            domain=self.domain
        )
        
    def __mul__(self, other: Union['Type2FuzzySet', float]) -> 'Type2FuzzySet':
        """
        Multiply Type-2 Fuzzy Set by another Type-2 Fuzzy Set or scalar.
        
        Args:
            other (Union[Type2FuzzySet, float]): Another Type-2 Fuzzy Set or scalar
            
        Returns:
            Type2FuzzySet: Result of multiplication
        """
        if isinstance(other, (int, float)):
            def new_primary(x: float) -> List[float]:
                return [g ** other for g in self.primary_membership(x)]
                
            def new_secondary(x: float, grade: float) -> float:
                return self.secondary_membership(x, grade ** (1/other))
                
            return Type2FuzzySet(
                primary_membership=new_primary,
                secondary_membership=new_secondary,
                domain=self.domain
            )
        else:
            def new_primary(x: float) -> List[float]:
                grades1 = self.primary_membership(x)
                grades2 = other.primary_membership(x)
                return [g1 * g2 for g1 in grades1 for g2 in grades2]
                
            def new_secondary(x: float, grade: float) -> float:
                grades1 = self.primary_membership(x)
                grades2 = other.primary_membership(x)
                sec1 = [self.secondary_membership(x, g) for g in grades1]
                sec2 = [other.secondary_membership(x, g) for g in grades2]
                return np.mean([s1 * s2 for s1 in sec1 for s2 in sec2])
                
            return Type2FuzzySet(
                primary_membership=new_primary,
                secondary_membership=new_secondary,
                domain=self.domain
            )
            
    def __eq__(self, other: 'Type2FuzzySet') -> bool:
        """
        Check equality of two Type-2 Fuzzy Sets.
        
        Args:
            other (Type2FuzzySet): Another Type-2 Fuzzy Set
            
        Returns:
            bool: True if equal, False otherwise
        """
        if self.domain != other.domain:
            return False
            
        x = np.linspace(self.domain[0], self.domain[1], 10)
        for xi in x:
            grades1 = self.primary_membership(xi)
            grades2 = other.primary_membership(xi)
            if len(grades1) != len(grades2):
                return False
                
            for g1, g2 in zip(grades1, grades2):
                if not np.isclose(g1, g2):
                    return False
                    
            for g in grades1:
                if not np.isclose(self.secondary_membership(xi, g), other.secondary_membership(xi, g)):
                    return False
                    
        return True
                
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the Type-2 Fuzzy Set to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        # Note: This is a simplified representation
        x = np.linspace(self.domain[0], self.domain[1], 10)
        primary_dict = {}
        secondary_dict = {}
        
        for xi in x:
            x_str = f"{xi:.6f}"
            grades = self.primary_membership(xi)
            primary_dict[x_str] = grades
            for g in grades:
                g_str = f"{g:.6f}"
                key = f"{x_str}_{g_str}"
                secondary_dict[key] = self.secondary_membership(xi, g)
                
        return {
            'domain': list(self.domain),
            'primary_membership': primary_dict,
            'secondary_membership': secondary_dict
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Type2FuzzySet':
        """
        Create a Type-2 Fuzzy Set from a dictionary.
        
        Args:
            data (Dict[str, Any]): Dictionary containing fuzzy set values
            
        Returns:
            Type2FuzzySet: New Type-2 Fuzzy Set instance
        """
        # Note: This is a simplified reconstruction
        domain = tuple(data['domain'])
        
        def primary_membership(x: float) -> List[float]:
            # Find closest x value in the data
            x_str = min(data['primary_membership'].keys(),
                       key=lambda k: abs(float(k) - x))
            return data['primary_membership'][x_str]
            
        def secondary_membership(x: float, grade: float) -> float:
            # Find closest x value
            x_str = min(data['primary_membership'].keys(),
                       key=lambda k: abs(float(k) - x))
            # Find closest grade value for this x
            grades = data['primary_membership'][x_str]
            closest_grade = min(grades, key=lambda g: abs(g - grade))
            key = f"{x_str}_{closest_grade:.6f}"
            return data['secondary_membership'][key]
            
        return cls(
            primary_membership=primary_membership,
            secondary_membership=secondary_membership,
            domain=domain
        )
        
    def __str__(self) -> str:
        """String representation of the Type-2 Fuzzy Set."""
        return f"Type2FuzzySet(domain={self.domain})"
        
    def __repr__(self) -> str:
        """Detailed string representation of the Type-2 Fuzzy Set."""
        return self.__str__() 