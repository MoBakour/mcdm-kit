import numpy as np
from mcdm_kit.data import DecisionMatrix
from mcdm_kit.core import CIMAS, ARTASI

# Step 1: Define decision matrices for 3 experts
# Each matrix: alternatives × criteria
expert_matrices = [
    np.array([
        [80,  70, 60],
        [85,  65, 55],
        [90,  60, 50]
    ]),
    np.array([
        [75,  72, 58],
        [82,  68, 53],
        [88,  63, 49]
    ]),
    np.array([
        [78,  69, 61],
        [84,  66, 56],
        [89,  62, 52]
    ])
]

# Step 2: Define alternatives, criteria, and types
alternatives = ["Course A", "Course B", "Course C"]
criteria = ["Relevance", "Difficulty", "Student Demand"]
criteria_types = ["benefit", "cost", "benefit"]

# Step 3: Expert importance weights (e.g., normalized)
expert_weights = np.array([0.5, 0.3, 0.2])  # Must sum to 1

# Step 4: Create DecisionMatrix instances for each expert
expert_dms = [
    DecisionMatrix(matrix=matrix, alternatives=alternatives, criteria=criteria, criteria_types=criteria_types)
    for matrix in expert_matrices
]

# Step 5: Aggregate decision matrices using expert weights
# Stack into shape: (n_experts, n_alternatives, n_criteria)
stacked = np.stack([dm.matrix for dm in expert_dms], axis=0)

# Weighted average across experts (axis=0)
weighted_matrix = np.tensordot(expert_weights, stacked, axes=1)

# Step 6: Create the final aggregated DecisionMatrix
group_dm = DecisionMatrix(
    matrix=weighted_matrix,
    alternatives=alternatives,
    criteria=criteria,
    criteria_types=criteria_types
)

# Step 7: Apply CIMAS to calculate weights
cimas = CIMAS(group_dm)
weights = cimas.calculate_weights()

# Step 8: Apply ARTASI to rank using CIMAS weights
artasi = ARTASI(group_dm, weights=weights)
results = artasi.rank()

# Step 9: Output results
print("===== Final Rankings (CIMAS + ARTASI with Expert Importance) =====")
for r in results['rankings']:
    print(f"{r['rank']}: {r['alternative']} (Score: {r['score']:.4f})")
