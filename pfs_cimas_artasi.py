"""
This is an example implementation of the PFS-CIMAS-ARTASI method.

Expert data from a CSV file is loaded and processed to extract the expert importance, criteria importance, and course evaluations.
"""

import pandas as pd
import os
import matplotlib.pyplot as plt

# Set up file paths using the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'data', 'data.csv')

pfs_importance_map = {
    "Very Important (VI)":     (0.700, 0.010, 0.010),
    "Important (I)":           (0.600, 0.035, 0.030),
    "Medium (M)":              (0.260, 0.260, 0.260),
    "Unimportant (UI)":        (0.210, 0.270, 0.325),
    "Very Unimportant (VUI)":  (0.015, 0.397, 0.397),
}


pfs_eval_map = {
    "Extremely Good (EG)":  (0.995, 0.000, 0.000),
    "Very Very Good (VVG)": (0.825, 0.015, 0.015),
    "Very Good (VG)":       (0.755, 0.043, 0.050),
    "Good (G)":             (0.650, 0.131, 0.137),
    "Medium Good (MG)":     (0.510, 0.135, 0.250),
    "Medium (M)":           (0.260, 0.260, 0.260),
    "Medium Bad (MB)":      (0.225, 0.390, 0.263),
    "Bad (B)":              (0.150, 0.400, 0.295),
    "Very Bad (VB)":        (0.060, 0.410, 0.400),
    "Very Very Bad (VVB)":  (0.040, 0.400, 0.400),
}

# Load data using absolute path for reliability
try:
    df = pd.read_csv(data_path)
    print(f"Successfully loaded data from {data_path}")
except FileNotFoundError:
    print(f"Error: Data file not found at {data_path}")
    exit(1)

# Drop timestamp
df = df.drop(columns=df.columns[0])

# Extract parts
expert_importance_col = df.columns[0]
criteria_cols = df.columns[1:11]
course_eval_cols = df.columns[11:]  # assuming exactly 10 courses x 10 criteria

expert_importance = df[expert_importance_col]
criteria_importance = df[criteria_cols]
course_evaluation = df[course_eval_cols]


def map_pfs_column(col, mapping):
    return col.map(mapping)

# Map expert importance
expert_pfs = expert_importance.map(pfs_importance_map)

# Map criteria importance
criteria_pfs = criteria_importance.applymap(lambda x: pfs_eval_map.get(x, (0, 0, 0)))

# Map course evaluations
course_eval_pfs = course_evaluation.applymap(lambda x: pfs_eval_map.get(x, (0, 0, 0)))


# Restructure the course evaluation data
# First, let's identify the criteria names from the column headers
criteria_names = [col.split("'")[1] for col in criteria_cols]

# Initialize list to store reshaped records
reshaped_records = []

# We have 3 expert opinions in rows and 10 courses x 10 criteria in columns
num_experts = len(df)
num_courses = 10
num_criteria = 10

# Loop through each expert
for expert_idx in range(num_experts):
    # Loop through each course
    for course_idx in range(num_courses):
        # For each course, get the 10 criteria evaluations
        start_col_idx = 11 + course_idx * num_criteria
        end_col_idx = start_col_idx + num_criteria
        
        # Loop through each criterion for this course
        for i, criterion_name in enumerate(criteria_names):
            col_idx = start_col_idx + i
            if col_idx < len(df.columns):
                term = df.iloc[expert_idx, col_idx]
                mu, eta, nu = pfs_eval_map.get(term, (0.0, 0.0, 0.0))
                
                reshaped_records.append({
                    "Expert": expert_idx + 1,  # Identifying experts by their row number
                    "Course": course_idx + 1,  # Identifying courses by their index
                    "Criterion": criterion_name,
                    "mu": mu,
                    "eta": eta,
                    "nu": nu
                })

# Create the reshaped DataFrame
reshaped_df = pd.DataFrame(reshaped_records)

# Display first few rows to verify
print(reshaped_df.head())




# Step 1: Extract the expert importance values
expert_importance_values = []
for i in range(len(df)):
    term = df.iloc[i, 0]  # First column contains expert importance
    mu, eta, nu = pfs_importance_map.get(term, (0.0, 0.0, 0.0))
    expert_importance_values.append((mu, eta, nu))

# Step 2: Extract criteria importance matrices for each expert
criteria_importance_matrices = []
for i in range(len(df)):
    expert_criteria = []
    for j in range(1, 11):  # Columns 1-10 contain criteria importance
        term = df.iloc[i, j]
        mu, eta, nu = pfs_eval_map.get(term, (0.0, 0.0, 0.0))
        expert_criteria.append((mu, eta, nu))
    criteria_importance_matrices.append(expert_criteria)

# PFS operations
def pfs_score(mu, eta, nu):
    """Calculate the score of a PFS element"""
    return mu - nu

def pfs_accuracy(mu, eta, nu):
    """Calculate the accuracy of a PFS element"""
    return mu + nu

def pfs_multiply(pfs1, pfs2):
    """Multiply two PFS elements using standard operations"""
    mu1, eta1, nu1 = pfs1
    mu2, eta2, nu2 = pfs2
    
    mu = mu1 * mu2
    eta = eta1 * eta2
    nu = nu1 * nu2
    
    return (mu, eta, nu)

def pfs_weighted_average(pfs_list, weights):
    """Calculate weighted average of PFS elements using the arithmetic mean approach"""
    if not pfs_list or not weights or sum(weights) == 0:
        return (0, 0, 0)
    
    # Normalize weights
    total_weight = sum(weights)
    norm_weights = [w / total_weight for w in weights]
    
    # Initialize
    mu_sum = 0
    eta_sum = 0
    nu_sum = 0
    
    # Calculate weighted arithmetic mean
    for (mu, eta, nu), w in zip(pfs_list, norm_weights):
        mu_sum += w * mu
        eta_sum += w * eta
        nu_sum += w * nu
        
    return (mu_sum, eta_sum, nu_sum)

# Step 3: Calculate expert weights based on their importance
expert_scores = [pfs_score(mu, eta, nu) for mu, eta, nu in expert_importance_values]
expert_weights = [max(0, score) for score in expert_scores]  # Ensure non-negative weights

# Normalize expert weights so they sum to 1
total_expert_weight = sum(expert_weights)
if total_expert_weight > 0:
    expert_weights = [w / total_expert_weight for w in expert_weights]
else:
    # If all weights are 0, assign equal weights
    expert_weights = [1.0 / len(expert_weights) for _ in expert_weights]

print("\nExpert Weights:")
for i, weight in enumerate(expert_weights, 1):
    print(f"Expert {i}: {weight:.4f}")

# Step 4: Calculate criteria weights for each expert
criteria_weights_per_expert = []
for expert_criteria in criteria_importance_matrices:
    criteria_scores = [pfs_score(mu, eta, nu) for mu, eta, nu in expert_criteria]
    # Normalize scores to get weights
    total_score = sum(max(0, score) for score in criteria_scores)
    if total_score > 0:
        weights = [max(0, score) / total_score for score in criteria_scores]
    else:
        # Equal weights if all scores are negative or zero
        weights = [1.0 / len(criteria_scores)] * len(criteria_scores)
    criteria_weights_per_expert.append(weights)

# Step 5: Aggregate criteria weights across all experts
aggregated_criteria_weights = []
for j in range(len(criteria_names)):
    expert_weights_for_criterion = []
    for i in range(len(expert_weights)):
        expert_weights_for_criterion.append(criteria_weights_per_expert[i][j] * expert_weights[i])
    
    # Normalize to get final weight for this criterion
    total = sum(expert_weights_for_criterion)
    if total > 0:
        aggregated_criteria_weights.append(sum(expert_weights_for_criterion) / total)
    else:
        # Use equal weights if all experts have zero weight
        aggregated_criteria_weights.append(1.0 / len(criteria_names))

# Step 6: Calculate course scores
course_scores = {}

# Group by course
for course_id in range(1, num_courses + 1):
    # For each criterion
    criterion_weighted_pfs = []
    
    for criterion_idx, criterion_name in enumerate(criteria_names):
        # Get all expert opinions for this course and criterion
        expert_pfs_values = []
        for expert_id in range(1, num_experts + 1):
            filtered_rows = reshaped_df[(reshaped_df["Course"] == course_id) & 
                                      (reshaped_df["Expert"] == expert_id) & 
                                      (reshaped_df["Criterion"] == criterion_name)]
            
            if not filtered_rows.empty:
                row = filtered_rows.iloc[0]
                expert_pfs_values.append((row["mu"], row["eta"], row["nu"]))
        
        # Ensure we have expert opinions for this criterion
        if expert_pfs_values:
            # Weight by expert importance
            criterion_aggregated_pfs = pfs_weighted_average(expert_pfs_values, expert_weights)
            
            # Store with criterion weight for later aggregation
            criterion_weight = aggregated_criteria_weights[criterion_idx]
            criterion_weighted_pfs.append((criterion_aggregated_pfs, criterion_weight))
    
    # Aggregate all criteria for this course using criteria weights
    if criterion_weighted_pfs:
        pfs_values = [pfs for pfs, _ in criterion_weighted_pfs]
        weights = [weight for _, weight in criterion_weighted_pfs]
        
        # Final course PFS calculation
        course_final_pfs = pfs_weighted_average(pfs_values, weights)
        
        # Calculate score and accuracy
        course_scores[course_id] = {
            "pfs": course_final_pfs,
            "score": pfs_score(*course_final_pfs),
            "accuracy": pfs_accuracy(*course_final_pfs)
        }
    else:
        # Default if no data
        course_scores[course_id] = {
            "pfs": (0, 0, 0),
            "score": 0,
            "accuracy": 0
        }

# Step 7: Rank courses based on scores
ranked_courses = sorted(course_scores.items(), key=lambda x: x[1]["score"], reverse=True)

# Print results
print("\n===== PFS-CIMAS-ARTASI RESULTS =====")
print("\nCriteria Weights:")
for i, name in enumerate(criteria_names):
    print(f"{name}: {aggregated_criteria_weights[i]:.4f}")

print("\nCourse Rankings:")
for rank, (course_id, details) in enumerate(ranked_courses, 1):
    mu, eta, nu = details["pfs"]
    print(f"Rank {rank}: Course {course_id} - Score: {details['score']:.4f}, PFS: ({mu:.4f}, {eta:.4f}, {nu:.4f})")

# Create output directory for saving results

# Use absolute path based on script location to ensure files are saved inside workspace
output_dir = os.path.join(script_dir, 'output')
os.makedirs(output_dir, exist_ok=True)

# Export results to CSV
results_df = pd.DataFrame([
    {
        'Course': f"Course {course_id}",
        'Rank': rank,
        'Score': details['score'],
        'mu': details['pfs'][0],
        'eta': details['pfs'][1],
        'nu': details['pfs'][2]
    }
    for rank, (course_id, details) in enumerate(ranked_courses, 1)
])

# Save results to CSV
results_path = os.path.join(output_dir, 'pfs_cimas_artasi_results.csv')
results_df.to_csv(results_path, index=False)
print(f"\nResults saved to '{results_path}'")

# Create visualization of the rankings
try:    
    courses = [f"Course {c}" for c, _ in ranked_courses]
    scores = [details["score"] for _, details in ranked_courses]
    
    plt.figure(figsize=(12, 7))
    bars = plt.bar(courses, scores, color='skyblue')
    plt.xlabel('Courses')
    plt.ylabel('PFS Score')
    plt.title('Course Rankings by PFS-CIMAS-ARTASI')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Save visualization with a better path
    viz_path = os.path.join(output_dir, 'pfs_cimas_artasi_rankings.png')
    plt.savefig(viz_path, dpi=300)
    print(f"Ranking visualization saved as '{viz_path}'")
    
    # Create a second visualization showing criteria weights
    plt.figure(figsize=(12, 7))
    plt.bar(criteria_names, aggregated_criteria_weights, color='lightgreen')
    plt.xlabel('Criteria')
    plt.ylabel('Weight')
    plt.title('Criteria Weights')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save criteria weights visualization
    weights_viz_path = os.path.join(output_dir, 'criteria_weights.png')
    plt.savefig(weights_viz_path, dpi=300)
    print(f"Criteria weights visualization saved as '{weights_viz_path}'")
    
except ImportError:
    print("\nMatplotlib not found. Skipping visualization.")
except Exception as e:
    print(f"\nError creating visualization: {e}")