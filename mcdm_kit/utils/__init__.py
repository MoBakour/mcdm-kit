"""
Utility functions for MCDM methods.
"""

from .normalization import normalize_matrix
from .distance import (
    euclidean_distance,
    manhattan_distance,
    hamming_distance,
    cosine_similarity,
    fuzzy_distance,
    weighted_distance,
)

__all__ = [
    'normalize_matrix',
    'euclidean_distance',
    'manhattan_distance',
    'hamming_distance',
    'cosine_similarity',
    'fuzzy_distance',
    'weighted_distance',
] 