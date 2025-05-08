"""
Distance calculation utility functions for MCDM methods.
"""

import numpy as np
from typing import Union, List, Tuple

def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two vectors.
    
    Args:
        x (np.ndarray): First vector
        y (np.ndarray): Second vector
        
    Returns:
        float: Euclidean distance
    """
    return np.sqrt(np.sum((x - y) ** 2))

def manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate Manhattan distance between two vectors.
    
    Args:
        x (np.ndarray): First vector
        y (np.ndarray): Second vector
        
    Returns:
        float: Manhattan distance
    """
    return np.sum(np.abs(x - y))

def hamming_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate Hamming distance between two vectors.
    
    Args:
        x (np.ndarray): First vector
        y (np.ndarray): Second vector
        
    Returns:
        float: Hamming distance
    """
    return np.sum(x != y)

def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        x (np.ndarray): First vector
        y (np.ndarray): Second vector
        
    Returns:
        float: Cosine similarity
    """
    dot_product = np.dot(x, y)
    norm_x = np.sqrt(np.sum(x ** 2))
    norm_y = np.sqrt(np.sum(y ** 2))
    
    if norm_x == 0 or norm_y == 0:
        return 0
    
    return dot_product / (norm_x * norm_y)

def fuzzy_distance(x: np.ndarray, y: np.ndarray, p: float = 2) -> float:
    """
    Calculate fuzzy distance between two vectors.
    
    Args:
        x (np.ndarray): First vector
        y (np.ndarray): Second vector
        p (float): Order of the distance (default: 2 for Euclidean)
        
    Returns:
        float: Fuzzy distance
    """
    return np.power(np.sum(np.power(np.abs(x - y), p)), 1/p)

def weighted_distance(x: np.ndarray,
                     y: np.ndarray,
                     weights: np.ndarray,
                     p: float = 2) -> float:
    """
    Calculate weighted distance between two vectors.
    
    Args:
        x (np.ndarray): First vector
        y (np.ndarray): Second vector
        weights (np.ndarray): Weights for each dimension
        p (float): Order of the distance (default: 2 for Euclidean)
        
    Returns:
        float: Weighted distance
    """
    return np.power(np.sum(weights * np.power(np.abs(x - y), p)), 1/p) 