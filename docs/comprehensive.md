# MCDM Kit Comprehensive Documentation

This document provides a complete reference for the MCDM Kit library, including all MCDM methods, fuzzy set types, utilities, and data handling capabilities.

## Table of Contents

1. [Core Components](#core-components)
    - [DecisionMatrix](#decisionmatrix)
    - [Base MCDM Method](#base-mcdm-method)
2. [MCDM Methods](#mcdm-methods)
    - [TOPSIS](#topsis)
    - [WISP](#wisp)
    - [CIMAS](#cimas)
    - [ARTASI](#artasi)
    - [WENSLO](#wenslo)
    - [MABAC](#mabac)
    - [ARLON](#arlon)
    - [DEMATEL](#dematel)
    - [AROMAN](#aroman)
3. [Fuzzy Set Types](#fuzzy-set-types)
    - [Base Fuzzy Set](#base-fuzzy-set)
    - [Picture Fuzzy Sets](#picture-fuzzy-sets)
    - [Interval Fuzzy Sets](#interval-fuzzy-sets)
    - [Fermatean Fuzzy Sets](#fermatean-fuzzy-sets)
    - [Hesitant Fuzzy Sets](#hesitant-fuzzy-sets)
    - [Intuitionistic Fuzzy Sets](#intuitionistic-fuzzy-sets)
    - [Pythagorean Fuzzy Sets](#pythagorean-fuzzy-sets)
    - [Spherical Fuzzy Sets](#spherical-fuzzy-sets)
    - [Neutrosophic Sets](#neutrosophic-sets)
    - [Type-2 Fuzzy Sets](#type-2-fuzzy-sets)
4. [Utilities](#utilities)
    - [Distance Calculations](#distance-calculations)
    - [Normalization Methods](#normalization-methods)
5. [Data Handling](#data-handling)
    - [Data Loading](#data-loading)
    - [Matrix Operations](#matrix-operations)

## Core Components

### DecisionMatrix

The `DecisionMatrix` class is the fundamental data structure for all MCDM operations.

```python
from mcdm_kit.data import DecisionMatrix
```

#### Constructor Parameters

```python
DecisionMatrix(
    decision_matrix: Union[np.ndarray, List[List[Any]]],
    alternatives: Optional[List[str]] = None,
    criteria: Optional[List[str]] = None,
    criteria_types: Optional[List[str]] = None,
    weights: Optional[List[float]] = None,
    fuzzy: Optional[Union[str, Type[BaseFuzzySet]]] = None
)
```

-   `decision_matrix`: Input matrix (numpy array or list of lists)
-   `alternatives`: List of alternative names (optional)
-   `criteria`: List of criterion names (optional)
-   `criteria_types`: List of criterion types ('benefit' or 'cost') (optional)
-   `weights`: List of criterion weights (optional)
-   `fuzzy`: Fuzzy set type (string or class) (optional)

#### Methods

-   `get_matrix()`: Returns the decision matrix
-   `get_weights()`: Returns the criterion weights
-   `get_criteria_types()`: Returns the criterion types
-   `get_alternatives()`: Returns the alternative names
-   `get_criteria()`: Returns the criterion names
-   `get_fuzzy_details()`: Returns detailed fuzzy set information
-   `get_fuzzy_distances()`: Returns distances between fuzzy sets
-   `validate()`: Validates the decision matrix structure
-   `normalize(method: str = 'vector')`: Normalizes the decision matrix
-   `apply_weights()`: Applies weights to the decision matrix

### Base MCDM Method

The `BaseMCDMMethod` class provides common functionality for all MCDM methods.

```python
from mcdm_kit.core import BaseMCDMMethod
```

#### Constructor Parameters

```python
BaseMCDMMethod(
    decision_matrix: Union[DecisionMatrix, np.ndarray],
    weights: Optional[List[float]] = None,
    criteria_types: Optional[List[str]] = None
)
```

#### Common Methods

-   `validate_inputs()`: Validates input parameters
-   `normalize_matrix()`: Normalizes the decision matrix
-   `calculate_weighted_matrix()`: Calculates weighted normalized matrix
-   `rank()`: Ranks alternatives (to be implemented by subclasses)

## MCDM Methods

### TOPSIS

**Technique for Order of Preference by Similarity to Ideal Solution**

```python
from mcdm_kit.core import TOPSIS
```

#### Constructor Parameters

```python
TOPSIS(
    decision_matrix: Union[DecisionMatrix, np.ndarray],
    weights: Optional[List[float]] = None,
    criteria_types: Optional[List[str]] = None,
    normalization_method: str = 'vector'
)
```

#### Methods

-   `normalize_matrix()`: Normalizes using specified method
-   `calculate_weighted_matrix()`: Applies weights to normalized matrix
-   `calculate_ideal_solutions()`: Determines ideal and anti-ideal solutions
-   `calculate_distances()`: Computes distances to ideal solutions
-   `calculate_scores()`: Calculates relative closeness coefficients
-   `rank()`: Returns rankings and scores

### WISP

**Weighted Integrated Score Performance**

```python
from mcdm_kit.core import WISP
```

#### Constructor Parameters

```python
WISP(
    decision_matrix: Union[DecisionMatrix, np.ndarray],
    weights: Optional[List[float]] = None,
    criteria_types: Optional[List[str]] = None,
    normalization_method: str = 'vector',
    performance_thresholds: Optional[List[float]] = None
)
```

#### Methods

-   `normalize_matrix()`: Normalizes using specified method
-   `calculate_performance_scores()`: Computes performance scores
-   `calculate_weighted_scores()`: Applies weights to performance scores
-   `rank()`: Returns rankings and scores

### CIMAS

**Criterion Impact MeAsurement System**

```python
from mcdm_kit.core import CIMAS
```

#### Constructor Parameters

```python
CIMAS(
    decision_matrix: Union[DecisionMatrix, np.ndarray],
    weights: Optional[List[float]] = None,
    criteria_types: Optional[List[str]] = None,
    normalization_method: str = 'vector'
)
```

#### Methods

-   `normalize_matrix()`: Normalizes using specified method
-   `calculate_impact_matrix()`: Computes criterion impact matrix
-   `calculate_weighted_impacts()`: Applies weights to impacts
-   `rank()`: Returns rankings and impact scores

### ARTASI

**Additive Ratio Transition to Aspiration Solution Integration**

```python
from mcdm_kit.core import ARTASI
```

#### Constructor Parameters

```python
ARTASI(
    decision_matrix: Union[DecisionMatrix, np.ndarray],
    weights: Optional[List[float]] = None,
    criteria_types: Optional[List[str]] = None,
    normalization_method: str = 'vector',
    aspiration_levels: Optional[List[float]] = None
)
```

#### Methods

-   `normalize_matrix()`: Normalizes using specified method
-   `calculate_aspiration_distances()`: Computes distances to aspiration levels
-   `calculate_weighted_distances()`: Applies weights to distances
-   `rank()`: Returns rankings and distance scores

### WENSLO

**WEighted Navigation of Standard Level Origins**

```python
from mcdm_kit.core import WENSLO
```

#### Constructor Parameters

```python
WENSLO(
    decision_matrix: Union[DecisionMatrix, np.ndarray],
    weights: Optional[List[float]] = None,
    criteria_types: Optional[List[str]] = None,
    normalization_method: str = 'vector',
    standard_levels: Optional[List[float]] = None
)
```

#### Methods

-   `normalize_matrix()`: Normalizes using specified method
-   `calculate_standard_distances()`: Computes distances to standard levels
-   `calculate_weighted_distances()`: Applies weights to distances
-   `rank()`: Returns rankings and distance scores

### MABAC

**Multi-Attributive Border Approximation area Comparison**

```python
from mcdm_kit.core import MABAC
```

#### Constructor Parameters

```python
MABAC(
    decision_matrix: Union[DecisionMatrix, np.ndarray],
    weights: Optional[List[float]] = None,
    criteria_types: Optional[List[str]] = None,
    normalization_method: str = 'vector'
)
```

#### Methods

-   `normalize_matrix()`: Normalizes using specified method
-   `calculate_border_area()`: Computes border approximation area
-   `calculate_distances()`: Computes distances to border area
-   `rank()`: Returns rankings and distance scores

### ARLON

**Aggregated Ranking of Level-based Ordinal Normalization**

```python
from mcdm_kit.core import ARLON
```

#### Constructor Parameters

```python
ARLON(
    decision_matrix: Union[DecisionMatrix, np.ndarray],
    weights: Optional[List[float]] = None,
    criteria_types: Optional[List[str]] = None,
    normalization_method: str = 'vector',
    levels: int = 5
)
```

#### Methods

-   `normalize_matrix()`: Normalizes using specified method
-   `calculate_ordinal_levels()`: Computes ordinal levels
-   `calculate_weighted_levels()`: Applies weights to levels
-   `rank()`: Returns rankings and level scores

### DEMATEL

**DEcision MAking Trial and Evaluation Laboratory**

```python
from mcdm_kit.core import DEMATEL
```

#### Constructor Parameters

```python
DEMATEL(
    decision_matrix: Union[DecisionMatrix, np.ndarray],
    threshold: Optional[float] = None,
    alpha: float = 0.1
)
```

#### Methods

-   `calculate_direct_relation_matrix()`: Computes direct relation matrix
-   `calculate_normalized_matrix()`: Normalizes relation matrix
-   `calculate_total_relation_matrix()`: Computes total relation matrix
-   `calculate_prominence_relation()`: Computes prominence and relation
-   `rank()`: Returns rankings and influence scores

### AROMAN

**Additive Ratio Assessment with Multiple Criteria**

```python
from mcdm_kit.core import AROMAN
```

#### Constructor Parameters

```python
AROMAN(
    decision_matrix: Union[DecisionMatrix, np.ndarray],
    weights: Optional[List[float]] = None,
    criteria_types: Optional[List[str]] = None
)
```

#### Methods

-   `normalize_matrix()`: Normalizes using vector normalization
-   `calculate_weighted_matrix()`: Applies weights to normalized matrix
-   `calculate_ideal_solutions()`: Determines ideal and anti-ideal solutions
-   `calculate_scores()`: Computes relative closeness coefficients
-   `rank()`: Returns rankings and scores

## Fuzzy Set Types

### Base Fuzzy Set

The `BaseFuzzySet` class provides common functionality for all fuzzy set types.

```python
from mcdm_kit.fuzz import BaseFuzzySet
```

#### Common Methods

-   `validate()`: Validates fuzzy set parameters
-   `get_membership()`: Returns membership degree
-   `get_non_membership()`: Returns non-membership degree
-   `get_hesitation()`: Returns hesitation degree
-   `distance(other: BaseFuzzySet)`: Calculates distance to another fuzzy set

### Picture Fuzzy Sets

```python
from mcdm_kit.fuzz import PictureFuzzySet
```

```python
PictureFuzzySet(
    membership: float,
    neutral: float,
    non_membership: float
)
```

### Interval Fuzzy Sets

```python
from mcdm_kit.fuzz import IntervalFuzzySet
```

```python
IntervalFuzzySet(
    membership_lower: float,
    membership_upper: float,
    non_membership_lower: float,
    non_membership_upper: float
)
```

### Fermatean Fuzzy Sets

```python
from mcdm_kit.fuzz import FermateanFuzzySet
```

```python
FermateanFuzzySet(
    membership: float,
    non_membership: float
)
```

### Hesitant Fuzzy Sets

```python
from mcdm_kit.fuzz import HesitantFuzzySet
```

```python
HesitantFuzzySet(
    membership_values: List[float]
)
```

### Intuitionistic Fuzzy Sets

```python
from mcdm_kit.fuzz import IntuitionisticFuzzySet
```

```python
IntuitionisticFuzzySet(
    membership: float,
    non_membership: float
)
```

### Pythagorean Fuzzy Sets

```python
from mcdm_kit.fuzz import PythagoreanFuzzySet
```

```python
PythagoreanFuzzySet(
    membership: float,
    non_membership: float
)
```

### Spherical Fuzzy Sets

```python
from mcdm_kit.fuzz import SphericalFuzzySet
```

```python
SphericalFuzzySet(
    membership: float,
    neutral: float,
    non_membership: float
)
```

### Neutrosophic Sets

```python
from mcdm_kit.fuzz import NeutrosophicSet
```

```python
NeutrosophicSet(
    truth: float,
    indeterminacy: float,
    falsity: float
)
```

### Type-2 Fuzzy Sets

```python
from mcdm_kit.fuzz import Type2FuzzySet
```

```python
Type2FuzzySet(
    primary_membership: float,
    secondary_membership: float,
    footprint_of_uncertainty: Tuple[float, float]
)
```

## Utilities

### Distance Calculations

```python
from mcdm_kit.utils.distance import (
    euclidean_distance,
    manhattan_distance,
    hamming_distance,
    cosine_similarity,
    fuzzy_distance,
    weighted_distance
)
```

-   `euclidean_distance(x: np.ndarray, y: np.ndarray)`: Euclidean distance
-   `manhattan_distance(x: np.ndarray, y: np.ndarray)`: Manhattan distance
-   `hamming_distance(x: np.ndarray, y: np.ndarray)`: Hamming distance
-   `cosine_similarity(x: np.ndarray, y: np.ndarray)`: Cosine similarity
-   `fuzzy_distance(x: np.ndarray, y: np.ndarray, p: float = 2)`: Fuzzy distance
-   `weighted_distance(x: np.ndarray, y: np.ndarray, weights: np.ndarray, p: float = 2)`: Weighted distance

### Normalization Methods

```python
from mcdm_kit.utils.normalization import (
    normalize_matrix,
    _vector_normalization,
    _minmax_normalization,
    _sum_normalization
)
```

-   `normalize_matrix(matrix: np.ndarray, criteria_types: List[str], method: str = 'vector')`: Matrix normalization
-   `_vector_normalization(matrix: np.ndarray)`: Vector normalization
-   `_minmax_normalization(matrix: np.ndarray, criteria_types: List[str])`: Min-max normalization
-   `_sum_normalization(matrix: np.ndarray, criteria_types: List[str])`: Sum normalization

## Data Handling

### Data Loading

```python
from mcdm_kit.data.loader import load_from_csv
```

```python
load_from_csv(
    filepath: str,
    alternatives_col: str = 'Alternative',
    criteria_cols: List[str] = None,
    criteria_types: List[str] = None,
    weights: List[float] = None,
    fuzzy: Optional[Union[str, Type[BaseFuzzySet]]] = None
)
```

### Matrix Operations

```python
from mcdm_kit.data import (
    validate_matrix,
    validate_weights,
    validate_criteria_types,
    validate_fuzzy_values
)
```

-   `validate_matrix(matrix: np.ndarray)`: Validates matrix structure
-   `validate_weights(weights: List[float], n_criteria: int)`: Validates weights
-   `validate_criteria_types(criteria_types: List[str], n_criteria: int)`: Validates criteria types
-   `validate_fuzzy_values(matrix: np.ndarray, fuzzy_type: Union[str, Type[BaseFuzzySet]])`: Validates fuzzy values

### PFS-CIMAS-ARTASI Usage Example

This example demonstrates how to use Picture Fuzzy Sets (PFS) with CIMAS and ARTASI methods for multi-criteria decision making.

```python
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
weights = [0.2, 0.2, 0.2, 0.2, 0.2]

# Create decision matrix with Picture Fuzzy Sets
dm = DecisionMatrix(
    decision_matrix=matrix,
    alternatives=alternatives,
    criteria=criteria,
    criteria_types=criteria_types,
    weights=weights,
    fuzzy=PictureFuzzySet
)

# Apply CIMAS method
cimas = CIMAS(dm)
cimas_rankings, cimas_scores = cimas.rank()

# Apply ARTASI method
artasi = ARTASI(dm)
artasi_rankings, artasi_scores = artasi.rank()

# Print results
print("\nCIMAS Results:")
for alt, rank, score in zip(alternatives, cimas_rankings, cimas_scores):
    print(f"{alt}: Rank {rank}, Score {score:.4f}")

print("\nARTASI Results:")
for alt, rank, score in zip(alternatives, artasi_rankings, artasi_scores):
    print(f"{alt}: Rank {rank}, Score {score:.4f}")
```

This example shows:

1. How to create a decision matrix using Picture Fuzzy Sets
2. How to define alternatives, criteria, criteria types, and weights
3. How to apply both CIMAS and ARTASI methods to the same decision matrix
4. How to interpret and display the results

The Picture Fuzzy Sets in this example are represented as tuples of (membership, neutral, non-membership) values, where:

-   membership: degree of positive membership
-   neutral: degree of neutral membership
-   non-membership: degree of negative membership

Each value in the tuple must be between 0 and 1, and their sum must not exceed 1.
