"""
Normalization utility functions for MCDM methods.
"""

import numpy as np
from typing import List, Union

def normalize_matrix(matrix: np.ndarray,
                    criteria_types: List[str],
                    method: str = 'vector') -> np.ndarray:
    """
    Normalize a decision matrix using the specified method.
    
    Args:
        matrix (np.ndarray): Decision matrix to normalize
        criteria_types (List[str]): List of criterion types ('benefit' or 'cost')
        method (str): Normalization method ('vector', 'minmax', or 'sum')
        
    Returns:
        np.ndarray: Normalized decision matrix
        
    Raises:
        ValueError: If invalid normalization method is specified
    """
    if method == 'vector':
        return _vector_normalization(matrix)
    elif method == 'minmax':
        return _minmax_normalization(matrix, criteria_types)
    elif method == 'sum':
        return _sum_normalization(matrix, criteria_types)
    else:
        raise ValueError(f"Invalid normalization method: {method}")

def _vector_normalization(matrix: np.ndarray) -> np.ndarray:
    """
    Normalize matrix using vector normalization.
    
    Args:
        matrix (np.ndarray): Decision matrix to normalize
        
    Returns:
        np.ndarray: Normalized matrix
    """
    # Calculate sum of squares for each column
    sum_squares = np.sqrt(np.sum(matrix ** 2, axis=0))
    
    # Avoid division by zero
    sum_squares[sum_squares == 0] = 1
    
    # Normalize
    return matrix / sum_squares

def _minmax_normalization(matrix: np.ndarray,
                         criteria_types: List[str]) -> np.ndarray:
    """
    Normalize matrix using min-max normalization.
    
    Args:
        matrix (np.ndarray): Decision matrix to normalize
        criteria_types (List[str]): List of criterion types
        
    Returns:
        np.ndarray: Normalized matrix
    """
    normalized = np.zeros_like(matrix, dtype=float)
    
    for j, criterion_type in enumerate(criteria_types):
        col = matrix[:, j]
        min_val = np.min(col)
        max_val = np.max(col)
        
        if max_val == min_val:
            normalized[:, j] = 1
        else:
            if criterion_type.lower() == 'benefit':
                normalized[:, j] = (col - min_val) / (max_val - min_val)
            else:  # cost criterion
                normalized[:, j] = (max_val - col) / (max_val - min_val)
    
    return normalized

def _sum_normalization(matrix: np.ndarray,
                      criteria_types: List[str]) -> np.ndarray:
    """
    Normalize matrix using sum normalization.
    
    Args:
        matrix (np.ndarray): Decision matrix to normalize
        criteria_types (List[str]): List of criterion types
        
    Returns:
        np.ndarray: Normalized matrix
    """
    normalized = np.zeros_like(matrix, dtype=float)
    
    for j, criterion_type in enumerate(criteria_types):
        col = matrix[:, j]
        col_sum = np.sum(col)
        
        if col_sum == 0:
            # If all values are zero, assign equal weights
            normalized[:, j] = 1 / len(col)
        else:
            if criterion_type.lower() == 'benefit':
                normalized[:, j] = col / col_sum
            else:  # cost criterion
                # Handle zero values in cost criteria
                if np.any(col == 0):
                    # Replace zeros with a small value to avoid division by zero
                    col = np.where(col == 0, np.finfo(float).eps, col)
                normalized[:, j] = (1 / col) / np.sum(1 / col)
    
    return normalized 