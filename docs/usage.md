# Usage Guide

This guide will show you how to use MCDM Kit for solving multi-criteria decision-making problems.

## Creating a Decision Matrix

The `DecisionMatrix` class is the core data structure for MCDM problems. You can create it in several ways:

### From a NumPy Array

```python
import numpy as np
from mcdm_kit.data import DecisionMatrix

# Create a decision matrix with 3 alternatives and 4 criteria
matrix = np.array([
    [8, 7, 9, 6],  # Alternative 1
    [7, 8, 6, 8],  # Alternative 2
    [9, 6, 7, 7]   # Alternative 3
])

# Create the decision matrix with custom names
alternatives = ['A1', 'A2', 'A3']
criteria = ['C1', 'C2', 'C3', 'C4']
criteria_types = ['benefit', 'benefit', 'benefit', 'cost']

dm = DecisionMatrix(
    matrix=matrix,
    alternatives=alternatives,
    criteria=criteria,
    criteria_types=criteria_types
)
```

### From a Pandas DataFrame

```python
import pandas as pd
from mcdm_kit.data import DecisionMatrix

# Create a DataFrame
df = pd.DataFrame({
    'C1': [8, 7, 9],
    'C2': [7, 8, 6],
    'C3': [9, 6, 7],
    'C4': [6, 8, 7]
}, index=['A1', 'A2', 'A3'])

# Create the decision matrix
dm = DecisionMatrix(
    matrix=df,
    criteria_types=['benefit', 'benefit', 'benefit', 'cost']
)
```

### From a CSV File

```python
from mcdm_kit.data import DecisionMatrix

# Create from CSV file
dm = DecisionMatrix.from_csv(
    'data.csv',
    alternatives_col='Alternative',
    criteria_types=['benefit', 'benefit', 'benefit', 'cost']
)
```

## Using MCDM Methods

MCDM Kit provides various methods for solving decision problems. Here's an example using the TOPSIS method:

```python
from mcdm_kit.core import TOPSIS

# Create a decision matrix (using the example from above)
matrix = np.array([
    [8, 7, 9, 6],
    [7, 8, 6, 8],
    [9, 6, 7, 7]
])

dm = DecisionMatrix(
    matrix=matrix,
    alternatives=['A1', 'A2', 'A3'],
    criteria=['C1', 'C2', 'C3', 'C4'],
    criteria_types=['benefit', 'benefit', 'benefit', 'cost']
)

# Create and run TOPSIS
topsis = TOPSIS(dm)
results = topsis.rank()

# Print results
print("Rankings:", results['rankings'])
print("Scores:", results['scores'])
```

## Working with Fuzzy Sets

MCDM Kit supports various types of fuzzy sets. Here's an example using Picture Fuzzy Sets:

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
    matrix=matrix,
    alternatives=['A1', 'A2'],
    criteria=['C1', 'C2'],
    criteria_types=['benefit', 'benefit'],
    fuzzy_type='PFS'
)

# Get fuzzy details
fuzzy_details = dm.get_fuzzy_details()
print("Fuzzy Details:", fuzzy_details)
```

## Advanced Usage

### Custom Weights

You can provide custom weights for criteria:

```python
from mcdm_kit.core import TOPSIS

# Create TOPSIS with custom weights
topsis = TOPSIS(dm, weights=[0.3, 0.3, 0.2, 0.2])
results = topsis.rank()
```

### Multiple Methods Comparison

You can compare results from different MCDM methods:

```python
from mcdm_kit.core import TOPSIS, WISP, CIMAS

# Create instances of different methods
topsis = TOPSIS(dm)
wisp = WISP(dm)
cimas = CIMAS(dm)

# Get rankings from each method
topsis_results = topsis.rank()
wisp_results = wisp.rank()
cimas_results = cimas.rank()

# Compare results
print("TOPSIS Rankings:", topsis_results['rankings'])
print("WISP Rankings:", wisp_results['rankings'])
print("CIMAS Rankings:", cimas_results['rankings'])
```

For more examples and advanced usage, please refer to the [Examples](examples.md) section.
