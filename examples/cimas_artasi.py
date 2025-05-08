from mcdm_kit.data import DecisionMatrix
from mcdm_kit.core import CIMAS, ARTASI

# Step 1: Create decision matrix
dm = DecisionMatrix.from_array(
    array=[
        [80, 70, 60],
        [85, 65, 55],
        [90, 60, 50]
    ],
    alternatives=["Course A", "Course B", "Course C"],
    criteria=["Relevance", "Difficulty", "Student Demand"],
    criteria_types=["benefit", "cost", "benefit"]
)

# Step 2: Use CIMAS to calculate criterion weights
cimas = CIMAS(decision_matrix=dm)
cimas.calculate_weights()  # internally normalizes if not set
weights = cimas.weights

# Step 3: Use ARTASI to rank alternatives using CIMAS weights
artasi = ARTASI(decision_matrix=dm, weights=weights)
results = artasi.rank()

# Step 4: Output rankings
print("Rankings:")
for r in results["rankings"]:
    print(f"{r['rank']}: {r['alternative']} (Score: {r['score']:.4f})")
