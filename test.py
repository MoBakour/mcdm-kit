from mcdm_kit.core import AROMAN
from mcdm_kit.fuzz import FermateanFuzzySet
from mcdm_kit.data import DecisionMatrix

# Create Fermatean fuzzy decision matrix
# Each element is a tuple of (membership, non-membership)
decision_data = [
    [(0.8, 0.3), (0.7, 0.4), (0.9, 0.2)],
    [(0.6, 0.5), (0.8, 0.3), (0.7, 0.4)],
    [(0.9, 0.2), (0.6, 0.5), (0.8, 0.3)]
]

# Define alternatives and criteria
alternatives = ['Alternative 1', 'Alternative 2', 'Alternative 3']
criteria = ['Criterion 1', 'Criterion 2', 'Criterion 3']
criteria_types = ['benefit', 'benefit', 'cost']

# Create DecisionMatrix with Fermatean fuzzy sets
decision_matrix = DecisionMatrix(
    decision_matrix=decision_data,
    alternatives=alternatives,
    criteria=criteria,
    criteria_types=criteria_types,
    fuzzy=FermateanFuzzySet  # Specify Fermatean Fuzzy Sets
)

# Define weights for criteria (optional)
weights = [0.4, 0.3, 0.3]

# Create AROMAN instance
aroman = AROMAN(decision_matrix, weights=weights)

# Calculate scores
scores = aroman.calculate_scores()
print("\nScores:")
for alt, score in scores.items():
    print(f"{alt}: {score:.4f}")

# Get rankings
rankings = aroman.rank()
print("\nRankings:")
for alt, rank in rankings.items():
    print(f"{alt}: {rank}")

# Get detailed fuzzy information
fuzzy_details = decision_matrix.get_fuzzy_details()
print("\nFuzzy Details:")
for alt in alternatives:
    print(f"\n{alt}:")
    for crit in criteria:
        details = fuzzy_details[alt][crit]
        print(f"  {crit}: μ={details['membership']:.2f}, ν={details['non_membership']:.2f}")

# Calculate distances between alternatives
fuzzy_distances = decision_matrix.get_fuzzy_distances()
print("\nFuzzy Distances between alternatives:")
for pair, distance in fuzzy_distances.items():
    print(f"{pair}: {distance:.4f}")

