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
6. [Usage Examples](#usage-examples)
    - [PFS-CIMAS-ARTASI Example](#pfs-cimas-artasi-usage-example)

## Core Components

### DecisionMatrix

The `DecisionMatrix` class is the fundamental data structure for all MCDM operations.

```python
from mcdm_kit.data import DecisionMatrix
```

#### Constructor Parameters

```python
DecisionMatrix(
    decision_matrix: Union[np.ndarray, pd.DataFrame, List[List[Any]]],
    alternatives: Optional[List[str]] = None,
    criteria: Optional[List[str]] = None,
    criteria_types: Optional[List[str]] = None,
    fuzzy: Optional[Union[str, Type[BaseFuzzySet]]] = None
)
```

-   `decision_matrix`: Input matrix (numpy array, pandas DataFrame, or list of lists)
-   `alternatives`: List of alternative names (optional)
-   `criteria`: List of criterion names (optional)
-   `criteria_types`: List of criterion types ('benefit' or 'cost') (optional)
-   `fuzzy`: Fuzzy set type (string or class) (optional)

#### Methods

-   `get_fuzzy_details()`: Returns detailed fuzzy set information if matrix contains fuzzy sets
-   `get_fuzzy_distances()`: Returns distances between fuzzy sets if matrix contains fuzzy sets
-   `get_benefit_criteria()`: Returns indices of benefit criteria
-   `get_cost_criteria()`: Returns indices of cost criteria
-   `to_dataframe()`: Converts the decision matrix to a pandas DataFrame
-   `from_array()`: Class method to create a DecisionMatrix from a 2D array
-   `from_csv()`: Class method to create a DecisionMatrix from a CSV file

#### Supported Fuzzy Types

The following fuzzy types are supported:

-   'PFS': Picture Fuzzy Sets
-   'IFS': Interval Fuzzy Sets
-   'T2FS': Type-2 Fuzzy Sets
-   'INFS': Intuitionistic Fuzzy Sets
-   'SFS': Spherical Fuzzy Sets
-   'NFS': Neutrosophic Sets
-   'PYFS': Pythagorean Fuzzy Sets
-   'FFS': Fermatean Fuzzy Sets
-   'HFS': Hesitant Fuzzy Sets

### Base MCDM Method

The `BaseMCDMMethod` class provides common functionality for all MCDM methods.

```python
from mcdm_kit.core import BaseMCDMMethod
```

#### Constructor Parameters

```python
BaseMCDMMethod(
    decision_matrix: DecisionMatrix
)
```

#### Abstract Methods

-   `calculate_weights()`: Calculate the weights for each criterion
-   `normalize_matrix()`: Normalize the decision matrix
-   `calculate_scores()`: Calculate the scores for each alternative
-   `rank()`: Rank the alternatives based on their scores

#### Common Methods

-   `validate_inputs()`: Validate the inputs to ensure they are suitable for the method

## MCDM Methods

### TOPSIS

**Technique for Order of Preference by Similarity to Ideal Solution**

```python
from mcdm_kit.core import TOPSIS
```

#### Constructor Parameters

```python
TOPSIS(
    decision_matrix: DecisionMatrix,
    weights: Optional[np.ndarray] = None,
    normalization_method: str = 'vector'
)
```

-   `decision_matrix`: The decision matrix
-   `weights`: Weights for criteria (optional, equal weights used if None)
-   `normalization_method`: Method for normalizing the decision matrix ('vector', 'minmax', or 'sum')

#### Methods

-   `calculate_weights()`: Calculate or validate weights for criteria
-   `normalize_matrix()`: Normalize the decision matrix
-   `calculate_ideal_solutions()`: Calculate ideal and anti-ideal solutions
-   `calculate_distances()`: Calculate distances to ideal and anti-ideal solutions
-   `calculate_scores()`: Calculate TOPSIS scores for each alternative
-   `rank()`: Rank alternatives based on TOPSIS scores

The `rank()` method returns a dictionary containing:

-   `rankings`: List of dictionaries with rank, alternative name, and score
-   `scores`: Dictionary mapping alternative names to their scores
-   `ideal_solution`: Dictionary mapping criteria to their ideal values
-   `anti_ideal_solution`: Dictionary mapping criteria to their anti-ideal values

### WISP

**Weighted Integrated Score Performance**

```python
from mcdm_kit.core import WISP
```

#### Constructor Parameters

```python
WISP(
    decision_matrix: DecisionMatrix,
    weights: Optional[np.ndarray] = None,
    normalization_method: str = 'vector',
    performance_thresholds: Optional[np.ndarray] = None
)
```

-   `decision_matrix`: The decision matrix
-   `weights`: Weights for criteria (optional, equal weights used if None)
-   `normalization_method`: Method for normalizing the decision matrix ('vector', 'minmax', or 'sum')
-   `performance_thresholds`: Performance thresholds for each criterion (optional, mean values used if None)

#### Methods

-   `calculate_weights()`: Calculate or validate weights for criteria
-   `calculate_performance_thresholds()`: Calculate performance thresholds for each criterion
-   `normalize_matrix()`: Normalize the decision matrix using WISP-specific normalization
-   `calculate_weighted_matrix()`: Calculate the weighted normalized matrix
-   `calculate_performance_matrix()`: Calculate the performance matrix
-   `calculate_scores()`: Calculate WISP scores for each alternative
-   `rank()`: Rank alternatives based on WISP scores

The `rank()` method returns a dictionary containing:

-   `rankings`: List of dictionaries with rank, alternative name, and score
-   `scores`: Dictionary mapping alternative names to their scores
-   `performance_matrix`: The calculated performance matrix
-   `weighted_matrix`: The weighted normalized matrix

### CIMAS

**Criterion Impact MeAsurement System**

```python
from mcdm_kit.core import CIMAS
```

#### Constructor Parameters

```python
CIMAS(
    decision_matrix: DecisionMatrix,
    weights: Optional[np.ndarray] = None,
    normalization_method: str = 'vector'
)
```

-   `decision_matrix`: The decision matrix
-   `weights`: Weights for criteria (optional, equal weights used if None)
-   `normalization_method`: Method for normalizing the decision matrix ('vector', 'minmax', or 'sum')

#### Methods

-   `calculate_weights()`: Calculate or validate weights for criteria
-   `normalize_matrix()`: Normalize the decision matrix using CIMAS-specific normalization
-   `calculate_weighted_matrix()`: Calculate the weighted normalized matrix
-   `calculate_impact_matrix()`: Calculate the impact matrix
-   `calculate_scores()`: Calculate CIMAS scores for each alternative
-   `rank()`: Rank alternatives based on CIMAS scores

The `rank()` method returns a dictionary containing:

-   `rankings`: List of dictionaries with rank, alternative name, and score
-   `scores`: Dictionary mapping alternative names to their scores
-   `impact_matrix`: The calculated impact matrix
-   `weighted_matrix`: The weighted normalized matrix

### ARTASI

**Additive Ratio Transition to Aspiration Solution Integration**

```python
from mcdm_kit.core import ARTASI
```

#### Constructor Parameters

```python
ARTASI(
    decision_matrix: DecisionMatrix,
    weights: Optional[np.ndarray] = None,
    normalization_method: str = 'vector',
    aspiration_levels: Optional[np.ndarray] = None
)
```

-   `decision_matrix`: The decision matrix
-   `weights`: Weights for criteria (optional, equal weights used if None)
-   `normalization_method`: Method for normalizing the decision matrix ('vector', 'minmax', or 'sum')
-   `aspiration_levels`: Aspiration levels for each criterion (optional, maximum values used for benefit criteria and minimum values for cost criteria if None)

#### Methods

-   `calculate_weights()`: Calculate or validate weights for criteria
-   `calculate_aspiration_levels()`: Calculate aspiration levels for each criterion
-   `normalize_matrix()`: Normalize the decision matrix using ARTASI-specific normalization
-   `calculate_weighted_matrix()`: Calculate the weighted normalized matrix
-   `calculate_aspiration_matrix()`: Calculate the aspiration matrix
-   `calculate_distance_matrix()`: Calculate the distance matrix from the aspiration solution
-   `calculate_scores()`: Calculate ARTASI scores for each alternative
-   `rank()`: Rank alternatives based on ARTASI scores

The `rank()` method returns a dictionary containing:

-   `rankings`: List of dictionaries with rank, alternative name, and score
-   `scores`: Dictionary mapping alternative names to their scores
-   `distance_matrix`: The calculated distance matrix
-   `aspiration_matrix`: The calculated aspiration matrix
-   `weighted_matrix`: The weighted normalized matrix

### WENSLO

**WEighted Navigation of Standard Level Origins**

```python
from mcdm_kit.core import WENSLO
```

#### Constructor Parameters

```python
WENSLO(
    decision_matrix: DecisionMatrix,
    weights: Optional[np.ndarray] = None,
    normalization_method: str = 'vector',
    standard_levels: Optional[np.ndarray] = None
)
```

-   `decision_matrix`: The decision matrix
-   `weights`: Weights for criteria (optional, equal weights used if None)
-   `normalization_method`: Method for normalizing the decision matrix ('vector', 'minmax', or 'sum')
-   `standard_levels`: Standard levels for each criterion (optional, mean values used if None)

#### Methods

-   `calculate_weights()`: Calculate or validate weights for criteria
-   `calculate_standard_levels()`: Calculate standard levels for each criterion
-   `normalize_matrix()`: Normalize the decision matrix using WENSLO-specific normalization
-   `calculate_weighted_matrix()`: Calculate the weighted normalized matrix
-   `calculate_standard_matrix()`: Calculate the standard matrix
-   `calculate_distance_matrix()`: Calculate the distance matrix from the standard levels
-   `calculate_scores()`: Calculate WENSLO scores for each alternative
-   `rank()`: Rank alternatives based on WENSLO scores

The `rank()` method returns a dictionary containing:

-   `rankings`: List of dictionaries with rank, alternative name, and score
-   `scores`: Dictionary mapping alternative names to their scores
-   `distance_matrix`: The calculated distance matrix
-   `standard_matrix`: The calculated standard matrix
-   `weighted_matrix`: The weighted normalized matrix

### MABAC

**Multi-Attributive Border Approximation area Comparison**

```python
from mcdm_kit.core import MABAC
```

#### Constructor Parameters

```python
MABAC(
    decision_matrix: DecisionMatrix,
    weights: Optional[np.ndarray] = None,
    normalization_method: str = 'vector'
)
```

-   `decision_matrix`: The decision matrix
-   `weights`: Weights for criteria (optional, equal weights used if None)
-   `normalization_method`: Method for normalizing the decision matrix ('vector', 'minmax', or 'sum')

#### Methods

-   `calculate_weights()`: Calculate or validate weights for criteria
-   `normalize_matrix()`: Normalize the decision matrix using MABAC-specific normalization
-   `calculate_weighted_matrix()`: Calculate the weighted normalized matrix
-   `calculate_border_matrix()`: Calculate the border approximation area matrix
-   `calculate_distance_matrix()`: Calculate the distance matrix from the border approximation area
-   `calculate_scores()`: Calculate MABAC scores for each alternative
-   `rank()`: Rank alternatives based on MABAC scores

The `rank()` method returns a dictionary containing:

-   `rankings`: List of dictionaries with rank, alternative name, and score
-   `scores`: Dictionary mapping alternative names to their scores
-   `border_matrix`: Dictionary mapping criteria to their border approximation values
-   `distance_matrix`: The calculated distance matrix

### ARLON

**Aggregated Ranking of Level-based Ordinal Normalization**

```python
from mcdm_kit.core import ARLON
```

#### Constructor Parameters

```python
ARLON(
    decision_matrix: DecisionMatrix,
    weights: Optional[np.ndarray] = None,
    normalization_method: str = 'vector',
    levels: Optional[int] = None
)
```

-   `decision_matrix`: The decision matrix
-   `weights`: Weights for criteria (optional, equal weights used if None)
-   `normalization_method`: Method for normalizing the decision matrix ('vector', 'minmax', or 'sum')
-   `levels`: Number of levels for ordinal normalization (optional, 5 levels used if None)

#### Methods

-   `calculate_weights()`: Calculate or validate weights for criteria
-   `normalize_matrix()`: Normalize the decision matrix using ARLON-specific normalization
-   `calculate_ordinal_matrix()`: Calculate the ordinal matrix based on normalized values
-   `calculate_weighted_matrix()`: Calculate the weighted ordinal matrix
-   `calculate_scores()`: Calculate ARLON scores for each alternative
-   `rank()`: Rank alternatives based on ARLON scores

The `rank()` method returns a dictionary containing:

-   `rankings`: List of dictionaries with rank, alternative name, and score
-   `scores`: Dictionary mapping alternative names to their scores
-   `ordinal_matrix`: The calculated ordinal matrix
-   `weighted_matrix`: The weighted ordinal matrix

### DEMATEL

**DEcision MAking Trial and Evaluation Laboratory**

```python
from mcdm_kit.core import DEMATEL
```

#### Constructor Parameters

```python
DEMATEL(
    decision_matrix: DecisionMatrix,
    threshold: Optional[float] = None,
    alpha: float = 0.1
)
```

-   `decision_matrix`: The decision matrix
-   `threshold`: Threshold for influence relationships (optional, mean + std used if None)
-   `alpha`: Alpha parameter for normalization (default: 0.1)

#### Methods

-   `normalize_matrix()`: Normalize the decision matrix using DEMATEL-specific normalization
-   `calculate_total_relation_matrix()`: Calculate the total relation matrix
-   `calculate_cause_effect_matrix()`: Calculate the cause-effect matrix and influence relationships
-   `calculate_scores()`: Calculate DEMATEL scores for each criterion
-   `calculate_weights()`: Calculate weights for criteria based on DEMATEL analysis
-   `rank()`: Analyze criteria relationships and provide rankings

The `rank()` method returns a dictionary containing:

-   `criteria_analysis`: List of dictionaries with criterion name, prominence, relation, and their rankings
-   `influence_relationships`: List of influence relationships between criteria
-   `total_relation_matrix`: The calculated total relation matrix
-   `threshold`: The threshold used for influence relationships

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

-   `decision_matrix`: Decision matrix or DecisionMatrix object
-   `weights`: List of criteria weights (optional, equal weights used if None)
-   `criteria_types`: List of criterion types ('benefit' or 'cost') (optional, all 'benefit' if None)

#### Methods

-   `normalize_matrix()`: Normalize the decision matrix using vector normalization
-   `calculate_weighted_matrix()`: Calculate the weighted normalized decision matrix
-   `calculate_ideal_solutions()`: Calculate ideal and anti-ideal solutions
-   `calculate_scores()`: Calculate AROMAN scores for each alternative
-   `rank()`: Rank alternatives based on their AROMAN scores

The `calculate_scores()` method returns a dictionary mapping alternative names to their scores.

The `rank()` method returns a dictionary mapping alternative names to their rankings (1 is best).

## Fuzzy Set Types

### Base Fuzzy Set

```python
from mcdm_kit.fuzz import BaseFuzzySet
```

#### Constructor Parameters

```python
BaseFuzzySet(
    membership: float,
    non_membership: Optional[float] = None,
    hesitation: Optional[float] = None
)
```

-   `membership`: Degree of membership (μ)
-   `non_membership`: Degree of non-membership (ν) (optional)
-   `hesitation`: Degree of hesitation (π) (optional)

#### Methods

-   `validate()`: Validate the fuzzy set values
-   `complement()`: Calculate the complement of the fuzzy set
-   `distance(other)`: Calculate the distance between two fuzzy sets
-   `similarity(other)`: Calculate the similarity between two fuzzy sets
-   `to_dict()`: Convert the fuzzy set to a dictionary
-   `from_dict(data)`: Create a fuzzy set from a dictionary (class method)

### Picture Fuzzy Sets

```python
from mcdm_kit.fuzz import PictureFuzzySet
```

#### Constructor Parameters

```python
PictureFuzzySet(
    membership: float,
    neutrality: float,
    non_membership: float
)
```

-   `membership`: Degree of membership (μ)
-   `neutrality`: Degree of neutrality (η)
-   `non_membership`: Degree of non-membership (ν)

#### Methods

-   `validate()`: Validate the Picture Fuzzy Set values
-   `complement()`: Calculate the complement of the Picture Fuzzy Set
-   `distance(other)`: Calculate the distance between two Picture Fuzzy Sets
-   `score()`: Calculate the score of the Picture Fuzzy Set
-   `accuracy()`: Calculate the accuracy of the Picture Fuzzy Set
-   `to_dict()`: Convert the Picture Fuzzy Set to a dictionary
-   `from_dict(data)`: Create a Picture Fuzzy Set from a dictionary (class method)

#### Operators

-   `+`: Add two Picture Fuzzy Sets
-   `*`: Multiply Picture Fuzzy Set by another Picture Fuzzy Set or scalar
-   `==`: Check equality of two Picture Fuzzy Sets

The degrees satisfy: 0 ≤ μ + η + ν ≤ 1

### Interval Fuzzy Sets

```python
from mcdm_kit.fuzz import IntervalFuzzySet
```

#### Constructor Parameters

```python
IntervalFuzzySet(
    lower_membership: Tuple[float, float],
    upper_membership: Tuple[float, float]
)
```

-   `lower_membership`: Lower membership interval [μ⁻, μ⁺]
-   `upper_membership`: Upper membership interval [ν⁻, ν⁺]

#### Methods

-   `validate()`: Validate the Interval-Valued Fuzzy Set values
-   `complement()`: Calculate the complement of the Interval-Valued Fuzzy Set
-   `distance(other)`: Calculate the distance between two Interval-Valued Fuzzy Sets
-   `score()`: Calculate the score of the Interval-Valued Fuzzy Set
-   `accuracy()`: Calculate the accuracy of the Interval-Valued Fuzzy Set
-   `to_dict()`: Convert the Interval-Valued Fuzzy Set to a dictionary
-   `from_dict(data)`: Create an Interval-Valued Fuzzy Set from a dictionary (class method)

#### Operators

-   `+`: Add two Interval-Valued Fuzzy Sets
-   `*`: Multiply Interval-Valued Fuzzy Set by another Interval-Valued Fuzzy Set or scalar
-   `==`: Check equality of two Interval-Valued Fuzzy Sets

The intervals satisfy: 0 ≤ μ⁻ ≤ μ⁺ ≤ 1 and 0 ≤ ν⁻ ≤ ν⁺ ≤ 1

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

#### Constructor Parameters

```python
HesitantFuzzySet(
    membership_degrees: Set[float]
)
```

-   `membership_degrees`: Set of membership degrees {γ₁, γ₂, ..., γₙ} where γᵢ ∈ [0,1]

#### Methods

-   `validate()`: Validate the Hesitant Fuzzy Set values
-   `complement()`: Calculate the complement of the Hesitant Fuzzy Set
-   `distance(other)`: Calculate the distance between two Hesitant Fuzzy Sets
-   `score()`: Calculate the score of the Hesitant Fuzzy Set
-   `accuracy()`: Calculate the accuracy of the Hesitant Fuzzy Set
-   `to_dict()`: Convert the Hesitant Fuzzy Set to a dictionary
-   `from_dict(data)`: Create a Hesitant Fuzzy Set from a dictionary (class method)

#### Operators

-   `+`: Add two Hesitant Fuzzy Sets
-   `*`: Multiply Hesitant Fuzzy Set by another Hesitant Fuzzy Set or scalar
-   `==`: Check equality of two Hesitant Fuzzy Sets

The membership degrees represent possible values for the membership of an element.

### Intuitionistic Fuzzy Sets

```python
from mcdm_kit.fuzz import IntuitionisticFuzzySet
```

#### Constructor Parameters

```python
IntuitionisticFuzzySet(
    membership: float,
    non_membership: float
)
```

-   `membership`: Degree of membership (μ) where μ ∈ [0,1]
-   `non_membership`: Degree of non-membership (ν) where ν ∈ [0,1]

#### Methods

-   `validate()`: Validate the Intuitionistic Fuzzy Set values
-   `complement()`: Calculate the complement of the Intuitionistic Fuzzy Set
-   `distance(other)`: Calculate the distance between two Intuitionistic Fuzzy Sets
-   `score()`: Calculate the score of the Intuitionistic Fuzzy Set
-   `accuracy()`: Calculate the accuracy of the Intuitionistic Fuzzy Set
-   `to_dict()`: Convert the Intuitionistic Fuzzy Set to a dictionary
-   `from_dict(data)`: Create an Intuitionistic Fuzzy Set from a dictionary (class method)

#### Operators

-   `+`: Add two Intuitionistic Fuzzy Sets
-   `*`: Multiply Intuitionistic Fuzzy Set by another Intuitionistic Fuzzy Set or scalar
-   `==`: Check equality of two Intuitionistic Fuzzy Sets

The degrees must satisfy: 0 ≤ μ + ν ≤ 1
The hesitation degree (π) is calculated as: π = 1 - (μ + ν)

### Pythagorean Fuzzy Sets

```python
from mcdm_kit.fuzz import PythagoreanFuzzySet
```

#### Constructor Parameters

```python
PythagoreanFuzzySet(
    membership: float,
    non_membership: float
)
```

-   `membership`: Degree of membership (μ) where μ ∈ [0,1]
-   `non_membership`: Degree of non-membership (ν) where ν ∈ [0,1]

#### Methods

-   `validate()`: Validate the Pythagorean Fuzzy Set values
-   `complement()`: Calculate the complement of the Pythagorean Fuzzy Set
-   `distance(other)`: Calculate the distance between two Pythagorean Fuzzy Sets
-   `score()`: Calculate the score of the Pythagorean Fuzzy Set
-   `accuracy()`: Calculate the accuracy of the Pythagorean Fuzzy Set
-   `to_dict()`: Convert the Pythagorean Fuzzy Set to a dictionary
-   `from_dict(data)`: Create a Pythagorean Fuzzy Set from a dictionary (class method)

#### Operators

-   `+`: Add two Pythagorean Fuzzy Sets
-   `*`: Multiply Pythagorean Fuzzy Set by another Pythagorean Fuzzy Set or scalar
-   `==`: Check equality of two Pythagorean Fuzzy Sets

The degrees must satisfy: μ² + ν² ≤ 1
The hesitation degree (π) is calculated as: π = √(1 - (μ² + ν²))

### Spherical Fuzzy Sets

```python
from mcdm_kit.fuzz import SphericalFuzzySet
```

#### Constructor Parameters

```python
SphericalFuzzySet(
    membership: float,
    neutrality: float,
    non_membership: float
)
```

-   `membership`: Degree of membership (μ) where μ ∈ [0,1]
-   `neutrality`: Degree of neutrality (η) where η ∈ [0,1]
-   `non_membership`: Degree of non-membership (ν) where ν ∈ [0,1]

#### Methods

-   `validate()`: Validate the Spherical Fuzzy Set values
-   `complement()`: Calculate the complement of the Spherical Fuzzy Set
-   `distance(other)`: Calculate the distance between two Spherical Fuzzy Sets
-   `score()`: Calculate the score of the Spherical Fuzzy Set
-   `accuracy()`: Calculate the accuracy of the Spherical Fuzzy Set
-   `to_dict()`: Convert the Spherical Fuzzy Set to a dictionary
-   `from_dict(data)`: Create a Spherical Fuzzy Set from a dictionary (class method)

#### Operators

-   `+`: Add two Spherical Fuzzy Sets
-   `*`: Multiply Spherical Fuzzy Set by another Spherical Fuzzy Set or scalar
-   `==`: Check equality of two Spherical Fuzzy Sets

The degrees must satisfy: μ² + η² + ν² ≤ 1
The hesitation degree (π) is calculated as: π = √(1 - (μ² + η² + ν²))

### Neutrosophic Sets

```python
from mcdm_kit.fuzz import NeutrosophicSet
```

#### Constructor Parameters

```python
NeutrosophicSet(
    truth: float,
    indeterminacy: float,
    falsity: float
)
```

-   `truth`: Truth-membership degree (T) where T ∈ [0,1]
-   `indeterminacy`: Indeterminacy-membership degree (I) where I ∈ [0,1]
-   `falsity`: Falsity-membership degree (F) where F ∈ [0,1]

#### Methods

-   `validate()`: Validate the Neutrosophic Set values
-   `complement()`: Calculate the complement of the Neutrosophic Set
-   `distance(other)`: Calculate the distance between two Neutrosophic Sets
-   `score()`: Calculate the score of the Neutrosophic Set
-   `accuracy()`: Calculate the accuracy of the Neutrosophic Set
-   `to_dict()`: Convert the Neutrosophic Set to a dictionary
-   `from_dict(data)`: Create a Neutrosophic Set from a dictionary (class method)

#### Operators

-   `+`: Add two Neutrosophic Sets
-   `*`: Multiply Neutrosophic Set by another Neutrosophic Set or scalar
-   `==`: Check equality of two Neutrosophic Sets

The degrees are independent and can be any value in [0,1]

### Type-2 Fuzzy Sets

```python
from mcdm_kit.fuzz import Type2FuzzySet
```

#### Constructor Parameters

```python
Type2FuzzySet(
    primary_membership: Callable[[float], List[float]],
    secondary_membership: Callable[[float, float], float],
    domain: Tuple[float, float] = (0, 1)
)
```

-   `primary_membership`: Function that returns possible membership grades for a given x value
-   `secondary_membership`: Function that returns secondary membership grade for a given x value and primary grade
-   `domain`: Domain of the fuzzy set (default: [0,1])

#### Methods

-   `validate()`: Validate the Type-2 Fuzzy Set values
-   `complement()`: Calculate the complement of the Type-2 Fuzzy Set
-   `distance(other)`: Calculate the distance between two Type-2 Fuzzy Sets
-   `score()`: Calculate the score of the Type-2 Fuzzy Set
-   `accuracy()`: Calculate the accuracy of the Type-2 Fuzzy Set
-   `to_dict()`: Convert the Type-2 Fuzzy Set to a dictionary
-   `from_dict(data)`: Create a Type-2 Fuzzy Set from a dictionary (class method)

#### Operators

-   `+`: Add two Type-2 Fuzzy Sets
-   `*`: Multiply Type-2 Fuzzy Set by another Type-2 Fuzzy Set or scalar
-   `==`: Check equality of two Type-2 Fuzzy Sets

A Type-2 Fuzzy Set is characterized by:

-   Primary membership function (μ) that maps elements to a set of membership grades
-   Secondary membership function (f) that assigns a weight to each primary membership grade

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

## Usage Examples

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
```

This example shows:

1. How to create a decision matrix using Picture Fuzzy Sets
2. How to define alternatives, criteria, and criteria types
3. How to pass weights to the MCDM methods (CIMAS and ARTASI)
4. How to interpret and display the results

The Picture Fuzzy Sets in this example are represented as tuples of (membership, neutral, non-membership) values, where:

-   membership: degree of positive membership
-   neutral: degree of neutral membership
-   non-membership: degree of negative membership

Each value in the tuple must be between 0 and 1, and their sum must not exceed 1.
