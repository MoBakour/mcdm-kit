# Fuzzy Set Implementations

This document provides a comprehensive overview of the fuzzy set implementations available in the MCDM Kit library.

## Base Fuzzy Set

The `BaseFuzzySet` class serves as the foundation for all fuzzy set types. It provides common operations and properties that all fuzzy sets should have:

-   Initialization with membership and non-membership degrees
-   Validation of fuzzy set values
-   Basic operations like complement, distance, and similarity
-   Conversion to/from dictionary representations

## Interval-Valued Fuzzy Sets

The `IntervalFuzzySet` class implements interval-valued fuzzy sets, characterized by:

-   Lower membership interval [μ⁻, μ⁺]
-   Upper membership interval [ν⁻, ν⁺]

where 0 ≤ μ⁻ ≤ μ⁺ ≤ 1 and 0 ≤ ν⁻ ≤ ν⁺ ≤ 1

Key features:

-   Validation of interval bounds
-   Complement calculation
-   Distance measurement between sets
-   Score and accuracy calculations
-   Addition and multiplication operations

Usage:

```python
from mcdm_kit.fuzz import IntervalFuzzySet

# Create an interval-valued fuzzy set
ivfs = IntervalFuzzySet(
    lower_membership=(0.3, 0.5),
    upper_membership=(0.4, 0.6)
)

# Or
ivfs = IntervalFuzzySet((0.3, 0.5), (0.4, 0.6))
```

## Type-2 Fuzzy Sets

The `Type2FuzzySet` class implements type-2 fuzzy sets, characterized by:

-   Primary membership function (μ)
-   Secondary membership function (f)

The primary membership function maps elements to a set of membership grades, and the secondary membership function assigns a weight to each primary membership grade.

Key features:

-   Validation of membership functions
-   Complement calculation
-   Distance measurement between sets
-   Score and accuracy calculations
-   Addition and multiplication operations
-   Support for custom domain ranges

Usage:

```python
from mcdm_kit.fuzz import Type2FuzzySet

def primary_membership(x):
    return [0.3, 0.5, 0.7]

def secondary_membership(x, grade):
    return 0.8

# Create a type-2 fuzzy set
t2fs = Type2FuzzySet(
    primary_membership=primary_membership,
    secondary_membership=secondary_membership
)

# Or
t2fs = Type2FuzzySet(primary_membership, secondary_membership)
```

## Intuitionistic Fuzzy Sets

The `IntuitionisticFuzzySet` class implements intuitionistic fuzzy sets, characterized by:

-   Membership degree (μ)
-   Non-membership degree (ν)

These degrees satisfy: 0 ≤ μ + ν ≤ 1
The hesitation degree (π) is calculated as: π = 1 - (μ + ν)

Key features:

-   Validation of membership and non-membership degrees
-   Complement calculation
-   Distance measurement between sets
-   Score and accuracy calculations
-   Addition and multiplication operations

Usage:

```python
from mcdm_kit.fuzz import IntuitionisticFuzzySet

# Create an intuitionistic fuzzy set
ifs = IntuitionisticFuzzySet(
    membership=0.7,
    non_membership=0.2
)

# Or
ifs = IntuitionisticFuzzySet(0.7, 0.2)
```

## Spherical Fuzzy Sets

The `SphericalFuzzySet` class implements spherical fuzzy sets, characterized by:

-   Membership degree (μ)
-   Neutrality degree (η)
-   Non-membership degree (ν)

These degrees satisfy: μ² + η² + ν² ≤ 1

Key features:

-   Validation of membership, neutrality, and non-membership degrees
-   Complement calculation
-   Distance measurement between sets
-   Score and accuracy calculations
-   Addition and multiplication operations

Usage:

```python
from mcdm_kit.fuzz import SphericalFuzzySet

# Create a spherical fuzzy set
sfs = SphericalFuzzySet(
    membership=0.6,
    neutrality=0.3,
    non_membership=0.4
)

# Or
sfs = SphericalFuzzySet(0.6, 0.3, 0.4)
```

## Neutrosophic Sets

The `NeutrosophicSet` class implements neutrosophic sets, characterized by:

-   Truth-membership degree (T)
-   Indeterminacy-membership degree (I)
-   Falsity-membership degree (F)

These degrees are independent and can be any value in [0,1]

Key features:

-   Validation of truth, indeterminacy, and falsity degrees
-   Complement calculation
-   Distance measurement between sets
-   Score and accuracy calculations
-   Addition and multiplication operations

Usage:

```python
from mcdm_kit.fuzz import NeutrosophicSet

# Create a neutrosophic set
ns = NeutrosophicSet(
    truth=0.8,
    indeterminacy=0.3,
    falsity=0.2
)

# Or
ns = NeutrosophicSet(0.8, 0.3, 0.2)
```

## Pythagorean Fuzzy Sets

The `PythagoreanFuzzySet` class implements Pythagorean fuzzy sets, characterized by:

-   Membership degree (μ)
-   Non-membership degree (ν)

These degrees satisfy: μ² + ν² ≤ 1
The hesitation degree (π) is calculated as: π = √(1 - (μ² + ν²))

Key features:

-   Validation of membership and non-membership degrees
-   Complement calculation
-   Distance measurement between sets
-   Score and accuracy calculations
-   Addition and multiplication operations

Usage:

```python
from mcdm_kit.fuzz import PythagoreanFuzzySet

# Create a Pythagorean fuzzy set
pfs = PythagoreanFuzzySet(
    membership=0.7,
    non_membership=0.4
)

# Or
pfs = PythagoreanFuzzySet(0.7, 0.4)
```

## Fermatean Fuzzy Sets

The `FermateanFuzzySet` class implements Fermatean fuzzy sets, characterized by:

-   Membership degree (μ)
-   Non-membership degree (ν)

These degrees satisfy: μ³ + ν³ ≤ 1
The hesitation degree (π) is calculated as: π = (1 - (μ³ + ν³))^(1/3)

Key features:

-   Validation of membership and non-membership degrees
-   Complement calculation
-   Distance measurement between sets
-   Score and accuracy calculations
-   Addition and multiplication operations

Usage:

```python
from mcdm_kit.fuzz import FermateanFuzzySet

# Create a Fermatean fuzzy set
ffs = FermateanFuzzySet(
    membership=0.6,
    non_membership=0.3
)

# Or
ffs = FermateanFuzzySet(0.6, 0.3)
```

## Picture Fuzzy Sets

The `PictureFuzzySet` class implements picture fuzzy sets, characterized by:

-   Membership degree (μ)
-   Neutrality degree (η)
-   Non-membership degree (ν)

These degrees satisfy: 0 ≤ μ + η + ν ≤ 1

Key features:

-   Validation of membership, neutrality, and non-membership degrees
-   Complement calculation
-   Distance measurement between sets
-   Score and accuracy calculations
-   Addition and multiplication operations

Usage:

```python
from mcdm_kit.fuzz import PictureFuzzySet

# Create a picture fuzzy set
pfs = PictureFuzzySet(
    membership=0.7,
    neutrality=0.2,
    non_membership=0.1
)

# Or
pfs = PictureFuzzySet(0.7, 0.2, 0.1)
```

## Hesitant Fuzzy Sets

The `HesitantFuzzySet` class implements hesitant fuzzy sets, characterized by:

-   A set of membership degrees: h = {γ₁, γ₂, ..., γₙ} where γᵢ ∈ [0,1]

The membership degrees represent possible values for the membership of an element.

Key features:

-   Validation of membership degrees
-   Complement calculation
-   Distance measurement between sets
-   Score and accuracy calculations
-   Addition and multiplication operations
-   Support for multiple membership values

Usage:

```python
from mcdm_kit.fuzz import HesitantFuzzySet

# Create a hesitant fuzzy set
hfs = HesitantFuzzySet({0.3, 0.5, 0.7})

# Or
hfs = HesitantFuzzySet([0.3, 0.5, 0.7])
```

## Common Operations

All fuzzy set implementations support the following common operations:

1. **Validation**: Ensures the fuzzy set values are within valid ranges
2. **Complement**: Calculates the complement of the fuzzy set
3. **Distance**: Measures the distance between two fuzzy sets
4. **Score**: Calculates a score value for the fuzzy set
5. **Accuracy**: Calculates an accuracy value for the fuzzy set
6. **Addition**: Adds two fuzzy sets together
7. **Multiplication**: Multiplies a fuzzy set by another fuzzy set or a scalar
8. **Equality**: Checks if two fuzzy sets are equal
9. **Dictionary Conversion**: Converts the fuzzy set to/from a dictionary representation

## Best Practices

1. Always validate fuzzy set values before using them in calculations
2. Use appropriate fuzzy set types based on the problem requirements
3. Consider the computational complexity of operations when choosing a fuzzy set type
4. Handle edge cases and exceptions appropriately
5. Use the provided conversion methods for data persistence
6. Consider the interpretability of results when choosing fuzzy set operations
