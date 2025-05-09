# Multi-Criteria Decision Making (MCDM) Methods

This document provides a comprehensive overview of all MCDM methods implemented in the mcdm_kit package. Each method is described with its purpose, key features, and usage guidelines.

## Table of Contents

1. [TOPSIS](#topsis)
2. [WISP](#wisp)
3. [CIMAS](#cimas)
4. [ARTASI](#artasi)
5. [WENSLO](#wenslo)
6. [MABAC](#mabac)
7. [ARLON](#arlon)
8. [DEMATEL](#dematel)
9. [AROMAN](#aroman)

## TOPSIS

**Technique for Order of Preference by Similarity to Ideal Solution**

TOPSIS is a widely used MCDM method that ranks alternatives based on their distance from ideal and anti-ideal solutions.

### Key Features

-   Calculates distances to both ideal and anti-ideal solutions
-   Supports multiple normalization methods (vector, minmax, sum)
-   Handles both benefit and cost criteria
-   Provides comprehensive ranking results including ideal and anti-ideal solutions

### Usage

```python
from mcdm_kit.core import TOPSIS
from mcdm_kit.data import DecisionMatrix

# Create decision matrix
matrix = DecisionMatrix(...)

# Initialize TOPSIS
topsis = TOPSIS(
    decision_matrix=matrix,
    weights=None,  # Optional: provide custom weights
    normalization_method='vector'  # Optional: choose normalization method
)

# Get rankings
results = topsis.rank()
```

## WISP

**Weighted Integrated Score Performance**

WISP is a performance-based MCDM method that evaluates alternatives using integrated performance scores and specific normalization procedures.

### Key Features

-   Uses performance thresholds for evaluation
-   Supports multiple normalization methods
-   Calculates weighted performance scores
-   Provides detailed performance analysis

### Usage

```python
from mcdm_kit.core import WISP
from mcdm_kit.data import DecisionMatrix

# Create decision matrix
matrix = DecisionMatrix(...)

# Initialize WISP
wisp = WISP(
    decision_matrix=matrix,
    weights=None,  # Optional: provide custom weights
    normalization_method='vector',  # Optional: choose normalization method
    performance_thresholds=None  # Optional: provide custom thresholds
)

# Get rankings
results = wisp.rank()
```

## CIMAS

**Criterion Impact MeAsurement System**

The CIMAS method ranks alternatives based on their proximity to ideal and anti-ideal solutions using a weighted normalized decision matrix.

### Key Features

-   Calculates criterion impact scores
-   Supports multiple normalization methods
-   Provides impact matrix analysis
-   Handles both benefit and cost criteria

### Usage

```python
from mcdm_kit.core import CIMAS
from mcdm_kit.data import DecisionMatrix

# Create decision matrix
matrix = DecisionMatrix(...)

# Initialize CIMAS
cimas = CIMAS(
    decision_matrix=matrix,
    weights=None,  # Optional: provide custom weights
    normalization_method='minmax'  # Optional: choose normalization method
)

# Get rankings
results = cimas.rank()
```

## ARTASI

**Additive Ratio Transition to Aspiration Solution Integration**

ARTASI evaluates alternatives based on their distance from aspiration levels using a specific normalization procedure.

### Key Features

-   Uses aspiration levels for evaluation
-   Supports multiple normalization methods
-   Calculates distance from aspiration solution
-   Provides comprehensive ranking results

### Usage

```python
from mcdm_kit.core import ARTASI
from mcdm_kit.data import DecisionMatrix

# Create decision matrix
matrix = DecisionMatrix(...)

# Initialize ARTASI
artasi = ARTASI(
    decision_matrix=matrix,
    weights=None,  # Optional: provide custom weights
    normalization_method='vector',  # Optional: choose normalization method
    aspiration_levels=None  # Optional: provide custom aspiration levels
)

# Get rankings
results = artasi.rank()
```

## WENSLO

**WEighted Navigation of Standard Level Origins**

WENSLO evaluates alternatives based on their distance from standard levels using a specific normalization procedure.

### Key Features

-   Uses standard levels for evaluation
-   Supports multiple normalization methods
-   Calculates distance from standard levels
-   Provides detailed ranking results

### Usage

```python
from mcdm_kit.core import WENSLO
from mcdm_kit.data import DecisionMatrix

# Create decision matrix
matrix = DecisionMatrix(...)

# Initialize WENSLO
wenslo = WENSLO(
    decision_matrix=matrix,
    weights=None,  # Optional: provide custom weights
    normalization_method='vector',  # Optional: choose normalization method
    standard_levels=None  # Optional: provide custom standard levels
)

# Get rankings
results = wenslo.rank()
```

## MABAC

**Multi-Attributive Border Approximation area Comparison**

MABAC evaluates alternatives based on their distance from the border approximation area.

### Key Features

-   Calculates border approximation area
-   Supports multiple normalization methods
-   Provides distance matrix analysis
-   Handles both benefit and cost criteria

### Usage

```python
from mcdm_kit.core import MABAC
from mcdm_kit.data import DecisionMatrix

# Create decision matrix
matrix = DecisionMatrix(...)

# Initialize MABAC
mabac = MABAC(
    decision_matrix=matrix,
    weights=None,  # Optional: provide custom weights
    normalization_method='vector'  # Optional: choose normalization method
)

# Get rankings
results = mabac.rank()
```

## ARLON

**Aggregated Ranking of Level-based Ordinal Normalization**

ARLON uses level-based ordinal normalization and specific aggregation procedures to rank alternatives.

### Key Features

-   Uses ordinal normalization with configurable levels
-   Supports multiple normalization methods
-   Provides ordinal matrix analysis
-   Handles both benefit and cost criteria

### Usage

```python
from mcdm_kit.core import ARLON
from mcdm_kit.data import DecisionMatrix

# Create decision matrix
matrix = DecisionMatrix(...)

# Initialize ARLON
arlon = ARLON(
    decision_matrix=matrix,
    weights=None,  # Optional: provide custom weights
    normalization_method='vector',  # Optional: choose normalization method
    levels=5  # Optional: specify number of ordinal levels
)

# Get rankings
results = arlon.rank()
```

## DEMATEL

**DEcision MAking Trial and Evaluation Laboratory**

DEMATEL is a structural modeling method that identifies cause-effect relationships between criteria.

### Key Features

-   Identifies cause-effect relationships
-   Calculates influence relationships
-   Provides prominence and relation analysis
-   Supports threshold-based relationship filtering

### Usage

```python
from mcdm_kit.core import DEMATEL
from mcdm_kit.data import DecisionMatrix

# Create decision matrix
matrix = DecisionMatrix(...)

# Initialize DEMATEL
dematel = DEMATEL(
    decision_matrix=matrix,
    threshold=None,  # Optional: provide custom threshold
    alpha=0.1  # Optional: specify alpha parameter
)

# Get analysis results
results = dematel.rank()
```

## AROMAN

**Additive Ratio Assessment with Multiple Criteria**

AROMAN is a comprehensive MCDM method that evaluates alternatives based on their relative distances from ideal and anti-ideal solutions, with support for both crisp and fuzzy values.

### Key Features

-   Normalizes and weights decision matrix
-   Evaluates alternatives using ideal solutions
-   Supports both crisp and fuzzy values
-   Handles benefit and cost criteria

### Usage

```python
from mcdm_kit.core import AROMAN
from mcdm_kit.data import DecisionMatrix

# Create decision matrix (crisp values)
matrix = DecisionMatrix(...)

# Initialize AROMAN
aroman = AROMAN(
    decision_matrix=matrix,
    weights=None,  # Optional: provide custom weights
    criteria_types=None  # Optional: specify criteria types
)

# Get scores and rankings
scores = aroman.calculate_scores()
rankings = aroman.rank()
```

## Common Features

All MCDM methods in this package share the following features:

1. **Input Validation**

    - Validates decision matrix
    - Validates weights (if provided)
    - Validates normalization methods
    - Validates method-specific parameters

2. **Normalization Methods**

    - Vector normalization
    - Min-max normalization
    - Sum normalization

3. **Weight Handling**

    - Support for custom weights
    - Automatic equal weight distribution
    - Weight normalization

4. **Result Format**

    - Consistent ranking format
    - Detailed scores and matrices
    - Method-specific analysis results

5. **Error Handling**
    - Comprehensive input validation
    - Clear error messages
    - Graceful handling of edge cases
