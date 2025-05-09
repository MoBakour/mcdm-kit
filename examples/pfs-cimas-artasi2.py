import numpy as np
from mcdm_kit.data import DecisionMatrix
from mcdm_kit.fuzz import PictureFuzzySet
from mcdm_kit.core import CIMAS, ARTASI

# Define the decision matrix with Picture Fuzzy Sets
matrix = [
    # C1    C2    C3    C4    C5
    [(0.6, 0.2, 0.1), (0.7, 0.1, 0.1), (0.5, 0.3, 0.1), (0.6, 0.2, 0.1), (0.7, 0.1, 0.1)],  # A1
    [(0.5, 0.3, 0.1), (0.6, 0.2, 0.1), (0.7, 0.1, 0.1), (0.5, 0.3, 0.1), (0.6, 0.2, 0.1)],  # A2
    [(0.7, 0.1, 0.1), (0.5, 0.3, 0.1), (0.6, 0.2, 0.1), (0.7, 0.1, 0.1), (0.5, 0.3, 0.1)],  # A3
    [(0.6, 0.2, 0.1), (0.7, 0.1, 0.1), (0.5, 0.3, 0.1), (0.6, 0.2, 0.1), (0.7, 0.1, 0.1)],  # A4
    [(0.5, 0.3, 0.1), (0.6, 0.2, 0.1), (0.7, 0.1, 0.1), (0.5, 0.3, 0.1), (0.6, 0.2, 0.1)]   # A5
]

# Define alternatives and criteria
alternatives = ['A1', 'A2', 'A3', 'A4', 'A5']
criteria = ['C1', 'C2', 'C3', 'C4', 'C5']

# Define criteria types (benefit or cost)
criteria_types = ['benefit', 'benefit', 'benefit', 'benefit', 'benefit']

# Define weights for criteria
weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

# Create decision matrix with Picture Fuzzy Sets
dm = DecisionMatrix(
    decision_matrix=matrix,
    alternatives=alternatives,
    criteria=criteria,
    criteria_types=criteria_types,
    fuzzy=PictureFuzzySet
)

# Apply CIMAS method with weights
cimas = CIMAS(dm, weights=weights)
cimas_result = cimas.rank()

# Apply ARTASI method with weights
artasi = ARTASI(dm, weights=weights)
artasi_result = artasi.rank()

# Print results
print("\nCIMAS Results:")
for ranking in cimas_result['rankings']:
    print(f"{ranking['alternative']}: Rank {ranking['rank']}, Score {ranking['score']:.4f}")

print("\nARTASI Results:")
for ranking in artasi_result['rankings']:
    print(f"{ranking['alternative']}: Rank {ranking['rank']}, Score {ranking['score']:.4f}")