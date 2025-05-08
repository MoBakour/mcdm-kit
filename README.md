# mcdm_kit

A comprehensive Python framework for Multi-Criteria Decision Making (MCDM) with support for fuzzy extensions and hybrid methods.

## Expected Folder Structure

```
mcdm_kit/
├─ mcdm_kit/
│  ├─ core/
│  │  ├─ __init__.py
│  │  ├─ arlon.py
│  │  ├─ artasi.py
│  │  ├─ base.py
│  │  ├─ cimas.py
│  │  ├─ dematel.py
│  │  ├─ mabac.py
│  │  ├─ topsis.py
│  │  ├─ wenslo.py
│  │  └─ wisp.py
│  ├─ data/
│  │  ├─ __init__.py
│  │  ├─ decision_matrix.py
│  │  └─ loader.py
│  ├─ fuzz/
│  │  ├─ __init__.py
│  │  ├─ base.py
│  │  ├─ fermatean.py
│  │  ├─ hesitant.py
│  │  ├─ interval.py
│  │  ├─ intuitionistic.py
│  │  ├─ neutrosophic.py
│  │  ├─ picture.py
│  │  ├─ pythagorean.py
│  │  ├─ spherical.py
│  │  └─ type2.py
│  ├─ utils/
│  │  ├─ __init__.py
│  │  ├─ distance.py
│  │  └─ normalization.py
│  └─ __init__.py
├─ tests/
│  ├─ test_arlon.py
│  ├─ test_artasi.py
│  ├─ test_cimas.py
│  ├─ test_dematel.py
│  ├─ test_fuzzy_sets_integration.py
│  ├─ test_fuzzy_sets.py
│  ├─ test_loader.py
│  ├─ test_mabac.py
│  ├─ test_topsis.py
│  ├─ test_wenslo.py
│  └─ test_wisp.py
├─ .coverage
├─ .gitignore
├─ experts_data_sample.csv
├─ pfs_cimas_artasi.py
├─ pyproject.toml
├─ README.md
├─ requirements-test.txt
└─ requirements.txt
```

## Covered Topics Includes

Type2 Neutrosophic Mabac
Intuitionistic Fuzzy CIMAS-ARLON
Fermatean Fuzzy
Intuitionistic & Interval-Valued Fuzzy
Interval Valued Pythagorean Fuzzy WISP
Intuitionistic Fuzzy DEMATEL
Two Step Logarithmic Normalization Fuzzy
Neutrosophic WENSLO-ARLON
PF-WENSLO-ARLON
PFS-CIMAS-ARTASI
Interval Valued Spherical Fuzzy CIMAS-WISP
Single-Valued Neutrosophic
Type-2 Neutrosophic TOPSIS
MEREC-AROMAN
WENSLO-ARTASI

## Features

-   **Core MCDM Methods:**

    -   TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)
    -   MABAC (Multi-Attributive Border Approximation area Comparison)
    -   CIMAS (Criterion Impact MeAsurement System)
    -   ARTASI (Additive Ratio Transition to Aspiration Solution Integration)
    -   WENSLO (WEighted Navigation of Standard Level Origins)
    -   WISP (Weighted Integrated Score Performance)
    -   DEMATEL (DEcision MAking Trial and Evaluation Laboratory)
    -   ARLON (Aggregated Ranking of Level-based Ordinal Normalization)

-   **Fuzzy Extensions:**

    -   Picture Fuzzy Sets
    -   Intuitionistic Fuzzy Sets
    -   Spherical Fuzzy Sets
    -   Fermatean Fuzzy Sets
    -   Neutrosophic Sets
    -   Interval Fuzzy Sets
    -   Hesitant Fuzzy Sets
    -   Pythagorean Fuzzy Sets
    -   Type-2 Fuzzy Sets

-   **Hybrid Methods:**

    -   Type-2 Neutrosophic MABAC
    -   Intuitionistic Fuzzy CIMAS-ARLON
    -   Interval-Valued Pythagorean Fuzzy WISP
    -   Intuitionistic Fuzzy DEMATEL
    -   Neutrosophic WENSLO-ARLON
    -   PF-WENSLO-ARLON
    -   PFS-CIMAS-ARTASI
    -   Interval Valued Spherical Fuzzy CIMAS-WISP
    -   Type-2 Neutrosophic TOPSIS
    -   MEREC-AROMAN
    -   WENSLO-ARTASI

-   **Utilities:**
    -   Distance calculations
    -   Similarity measures
    -   Normalization techniques
    -   Ranking algorithms

## Installation

```bash
pip install mcdm-kit
```

## Quick Start

### Basic Example Using TOPSIS

```python
import numpy as np
from mcdm_kit.data import DecisionMatrix
from mcdm_kit.core import TOPSIS

# Create a decision matrix with numerical values
matrix = DecisionMatrix(
    decision_matrix=[[4, 3, 5, 2],
                    [3, 4, 2, 5],
                    [5, 3, 4, 3]],
    alternatives=["Alt1", "Alt2", "Alt3"],
    criteria=["C1", "C2", "C3", "C4"],
    criteria_types=["benefit", "benefit", "cost", "cost"]
)

# Apply TOPSIS method with custom weights and normalization
weights = np.array([0.3, 0.2, 0.3, 0.2])
topsis = TOPSIS(
    decision_matrix=matrix,
    weights=weights,
    normalization_method='vector'  # Options: 'vector', 'minmax', 'sum'
)

# Get rankings and scores
results = topsis.rank()
print("Rankings:", results['rankings'])
print("Scores:", results['scores'])
print("Ideal Solution:", results['ideal_solution'])
```

### Example with Picture Fuzzy Sets

```python
from mcdm_kit.data import DecisionMatrix
from mcdm_kit.core import TOPSIS
from mcdm_kit.fuzz.picture import PictureFuzzySet

# Create a decision matrix with Picture Fuzzy Sets
matrix = DecisionMatrix(
    [[PictureFuzzySet(0.8, 0.1, 0.1), PictureFuzzySet(0.6, 0.2, 0.2)],
     [PictureFuzzySet(0.5, 0.3, 0.1), PictureFuzzySet(0.9, 0.05, 0.03)]],
    alternatives=["Alt1", "Alt2"],
    criteria=["C1", "C2"],
    criteria_types=["benefit", "benefit"],
    fuzzy_type='PFS'  # Specify fuzzy type
)

# Apply TOPSIS method
topsis = TOPSIS(matrix)
results = topsis.rank()

# Get fuzzy details
fuzzy_details = matrix.get_fuzzy_details()
print("Fuzzy Details:", fuzzy_details)

# Get fuzzy distances
distances = matrix.get_fuzzy_distances()
print("Fuzzy Distances:", distances)
```

### Example with CSV Data

```python
from mcdm_kit.data import DecisionMatrix
from mcdm_kit.core import TOPSIS

# Load data from CSV file
matrix = DecisionMatrix.from_csv(
    "experts_data_sample.csv",
    alternatives_col="Alternative",  # Column containing alternative names
    criteria_types=["benefit", "benefit", "cost", "cost"]
)

# Apply TOPSIS method
topsis = TOPSIS(matrix)
results = topsis.rank()

# The CSV file (experts_data_sample.csv) should have this structure:
"""
Alternative,Criterion1,Criterion2,Criterion3,Criterion4
Alt1,4,3,5,2
Alt2,3,4,2,5
Alt3,5,3,4,3
"""
```

### Example with Hybrid Method (CIMAS-ARTASI)

```python
from mcdm_kit.data import DecisionMatrix
from mcdm_kit.core import CIMAS, ARTASI
from mcdm_kit.fuzz.picture import PictureFuzzySet

# Create a decision matrix with Picture Fuzzy Sets
matrix = DecisionMatrix(
    [[PictureFuzzySet(0.8, 0.1, 0.1), PictureFuzzySet(0.6, 0.2, 0.2)],
     [PictureFuzzySet(0.5, 0.3, 0.1), PictureFuzzySet(0.9, 0.05, 0.03)]],
    alternatives=["Alt1", "Alt2"],
    criteria=["C1", "C2"],
    criteria_types=["benefit", "benefit"],
    fuzzy_type='PFS'
)

# Calculate weights using CIMAS
cimas = CIMAS(decision_matrix=matrix)
weights = cimas.calculate_weights()

# Use weights with ARTASI
artasi = ARTASI(decision_matrix=matrix, weights=weights)
results = artasi.rank()

print("Weights:", weights)
print("Rankings:", results['rankings'])
print("Scores:", results['scores'])
print("Aspiration Levels:", results['aspiration_levels'])
```

### Expected Results

The framework is expected to provide classes and methods of implementation of different MCDM models, and allow integration with various kinds of fuzzy sets approaches.
The framework should be able to accept CSV or JSON data from users, data that describes expert input. Then, methods should run on this data and provide ranking outputs.
An example of data is provided in [this sample file](./experts_data_sample.csv)
