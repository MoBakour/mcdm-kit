"""
Example demonstrating the use of Picture Fuzzy Sets with CIMAS and ARTASI methods.
This example shows both approaches for creating fuzzy decision matrices.
"""

import numpy as np
from mcdm_kit.data import DecisionMatrix
from mcdm_kit.fuzz import PictureFuzzySet
from mcdm_kit.core import CIMAS, ARTASI

# Example 1: Using pre-constructed fuzzy sets
matrix_preconstructed = [
    [PictureFuzzySet(0.8, 0.1, 0.1), PictureFuzzySet(0.6, 0.2, 0.2)],
    [PictureFuzzySet(0.5, 0.3, 0.1), PictureFuzzySet(0.9, 0.05, 0.03)]
]

dm1 = DecisionMatrix(
    decision_matrix=matrix_preconstructed,
    alternatives=["Alt1", "Alt2"],
    criteria=["C1", "C2"],
    criteria_types=["benefit", "benefit"],
    fuzzy='PFS'  # Using string-based fuzzy type
)

# Example 2: Using raw tuples
matrix_tuples = [
    [(0.8, 0.1, 0.1), (0.6, 0.2, 0.2)],
    [(0.5, 0.3, 0.1), (0.9, 0.05, 0.03)]
]

dm2 = DecisionMatrix(
    decision_matrix=matrix_tuples,
    alternatives=["Alt1", "Alt2"],
    criteria=["C1", "C2"],
    criteria_types=["benefit", "benefit"],
    fuzzy=PictureFuzzySet  # Using fuzzy set constructor directly
)

# The rest of the analysis remains the same for both approaches
def analyze_matrix(dm, name):
    print(f"\nAnalyzing {name}:")
    
    # Calculate weights using CIMAS
    cimas = CIMAS(decision_matrix=dm)
    weights = cimas.calculate_weights()
    print("Weights:", weights)
    
    # Use weights with ARTASI
    artasi = ARTASI(decision_matrix=dm, weights=weights)
    results = artasi.rank()
    
    print("Rankings:", results['rankings'])
    print("Scores:", results['scores'])
    print("Aspiration Levels:", results['aspiration_levels'])

if __name__ == "__main__":
    analyze_matrix(dm1, "Pre-constructed Fuzzy Sets")
    analyze_matrix(dm2, "Raw Tuple Values")
