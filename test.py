import numpy as np
from mcdm_kit.data import DecisionMatrix
from mcdm_kit.core import TOPSIS, CIMAS
from mcdm_kit.fuzz import PictureFuzzySet, FermateanFuzzySet

# Define the decision matrix
matrix = [
    # C1           C2           C3           C4           C5
    [(0.8, 0.5), (0.9, 0.4), (0.7, 0.6), (0.8, 0.5), (0.9, 0.4)],  # A1
    [(0.7, 0.6), (0.8, 0.5), (0.9, 0.4), (0.7, 0.6), (0.8, 0.5)],  # A2 
    [(0.9, 0.4), (0.7, 0.6), (0.8, 0.5), (0.9, 0.4), (0.7, 0.6)],  # A3
    [(0.8, 0.5), (0.9, 0.4), (0.7, 0.6), (0.8, 0.5), (0.9, 0.4)],  # A4
    [(0.7, 0.6), (0.8, 0.5), (0.9, 0.4), (0.7, 0.6), (0.8, 0.5)]   # A5
]

# Define alternatives and criteria
alternatives = ['A1', 'A2', 'A3', 'A4', 'A5']
criteria = ['C1', 'C2', 'C3', 'C4', 'C5']

# Define criteria types (benefit or cost)
criteria_types = ['benefit', 'benefit', 'benefit', 'benefit', 'benefit']

# Define weights for criteria
weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

# Create decision matrix with picture fuzzy sets
dm = DecisionMatrix(
    decision_matrix=matrix,
    alternatives=alternatives, 
    criteria=criteria,
    criteria_types=criteria_types,
    fuzzy=FermateanFuzzySet
)

# Apply CIMAS method with weights
cimas = CIMAS(dm, weights=weights)
cimas_result = cimas.rank()

# Print results
print("\nCIMAS Results:")
for ranking in cimas_result['rankings']:
    print(f"{ranking['alternative']}: Rank {ranking['rank']}, Score {ranking['score']:.4f}")
