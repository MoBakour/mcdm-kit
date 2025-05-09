import numpy as np
from mcdm_kit.data import DecisionMatrix
from mcdm_kit.fuzz import PictureFuzzySet
from mcdm_kit.core import CIMAS, ARTASI

# Define PFS matrix
pfs_matrix = [
    [PictureFuzzySet(0.8, 0.1, 0.1), PictureFuzzySet(0.6, 0.2, 0.2)],
    [PictureFuzzySet(0.5, 0.3, 0.1), PictureFuzzySet(0.9, 0.05, 0.03)],
    [PictureFuzzySet(0.7, 0.1, 0.15), PictureFuzzySet(0.65, 0.25, 0.05)]
]

alternatives = ['A1', 'A2', 'A3']
criteria = ['Cost', 'Satisfaction']
criteria_types = ['cost', 'benefit']
weights = np.array([0.5, 0.5])  # Example equal weighting

# Step 1: Create the fuzzy decision matrix
dm = DecisionMatrix(
    decision_matrix=pfs_matrix,
    alternatives=alternatives,
    criteria=criteria,
    criteria_types=criteria_types,
    fuzzy_type='PFS'
)

# Step 2: Run CIMAS
cimas = CIMAS(decision_matrix=dm, weights=weights)
cimas_result = cimas.rank()
cimas_scores = cimas_result['scores']

# Step 3: Run ARTASI
artasi = ARTASI(decision_matrix=dm, weights=weights)
artasi_result = artasi.rank()
artasi_scores = artasi_result['scores']

# Step 4: Combine scores (Hybrid)
hybrid_scores = {
    alt: (cimas_scores[alt] + artasi_scores[alt]) / 2
    for alt in alternatives
}

# Step 5: Final Ranking
sorted_hybrid = sorted(hybrid_scores.items(), key=lambda x: -x[1])
final_rankings = [
    {
        'rank': i + 1,
        'alternative': alt,
        'hybrid_score': round(score, 4)
    }
    for i, (alt, score) in enumerate(sorted_hybrid)
]

# Print results
from pprint import pprint
print("=== PFS-CIMAS-ARTASI Hybrid Rankings ===")
pprint(final_rankings)
