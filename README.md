# MCDM Kit

A comprehensive toolkit for Multi-Criteria Decision Making (MCDM) problems.

## Features

-   Multiple MCDM methods implementation
-   Support for various fuzzy set types
-   Flexible decision matrix creation
-   Easy-to-use API
-   Comprehensive documentation

## Installation

```bash
pip install mcdm_kit
```

## Quick Start

### Basic Usage

```python
import numpy as np
from mcdm_kit.data import DecisionMatrix
from mcdm_kit.core import TOPSIS

# Create a decision matrix
matrix = np.array([
    [8, 7, 9, 6],
    [7, 8, 6, 8],
    [9, 6, 7, 7]
])

dm = DecisionMatrix(
    decision_matrix=matrix,
    alternatives=['A1', 'A2', 'A3'],
    criteria=['C1', 'C2', 'C3', 'C4'],
    criteria_types=['benefit', 'benefit', 'benefit', 'cost']
)

# Use TOPSIS method
topsis = TOPSIS(dm)
results = topsis.rank()

print("Rankings:", results['rankings'])
print("Scores:", results['scores'])
```

### Working with Fuzzy Sets

MCDM Kit supports various types of fuzzy sets. There are two ways to create a fuzzy decision matrix:

#### Method 1: Using Pre-constructed Fuzzy Sets

```python
from mcdm_kit.fuzz import PictureFuzzySet
from mcdm_kit.data import DecisionMatrix

# Create a matrix of Picture Fuzzy Sets
matrix = [
    [PictureFuzzySet(0.8, 0.1, 0.1), PictureFuzzySet(0.7, 0.2, 0.1)],
    [PictureFuzzySet(0.6, 0.3, 0.1), PictureFuzzySet(0.8, 0.1, 0.1)]
]

# Create the decision matrix
dm = DecisionMatrix(
    decision_matrix=matrix,
    alternatives=['A1', 'A2'],
    criteria=['C1', 'C2'],
    criteria_types=['benefit', 'benefit'],
    fuzzy='PFS'  # Specify fuzzy type using string
)
```

#### Method 2: Using Raw Values and Fuzzy Set Constructor

```python
from mcdm_kit.fuzz import PictureFuzzySet
from mcdm_kit.data import DecisionMatrix

# Create a matrix of raw values
matrix = [
    [(0.8, 0.1, 0.1), (0.7, 0.2, 0.1)],
    [(0.6, 0.3, 0.1), (0.8, 0.1, 0.1)]
]

# Create the decision matrix
dm = DecisionMatrix(
    decision_matrix=matrix,
    alternatives=['A1', 'A2'],
    criteria=['C1', 'C2'],
    criteria_types=['benefit', 'benefit'],
    fuzzy=PictureFuzzySet  # Pass the fuzzy set constructor directly
)
```

## Supported MCDM Methods

-   TOPSIS (Technique for Order Preference by Similarity to an Ideal Solution)
-   WISP (Weighted Ideal Solution Point)
-   CIMAS (Comprehensive Ideal Solution Method)
-   ARTASI (Analytical Ranking Technique for Alternative Selection)
-   WENSLOW (Weighted Entropy-based Solution for Linear Ordering)
-   MABAC (Multi-Attributive Border Approximation Area Comparison)
-   ARLON (Analytical Ranking with Linear Ordering)
-   DEMATEL (Decision Making Trial and Evaluation Laboratory)

## Supported Fuzzy Set Types

-   Picture Fuzzy Sets (PFS)
-   Interval Fuzzy Sets (IFS)
-   Type-2 Fuzzy Sets (T2FS)
-   Intuitionistic Fuzzy Sets (INFS)
-   Spherical Fuzzy Sets (SFS)
-   Neutrosophic Sets (NFS)
-   Pythagorean Fuzzy Sets (PYFS)
-   Fermatean Fuzzy Sets (FFS)
-   Hesitant Fuzzy Sets (HFS)

## Documentation

For detailed documentation, please visit our [documentation page](docs/index.md).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
