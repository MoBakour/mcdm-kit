# Installation

## Requirements

MCDM Kit requires Python 3.7 or higher and the following dependencies:

-   NumPy
-   Pandas
-   SciPy

## Installation

You can install MCDM Kit using pip:

```bash
pip install mcdm_kit
```

## Development Installation

If you want to install the package in development mode, you can clone the repository and install it locally:

```bash
git clone https://github.com/yourusername/mcdm_kit.git
cd mcdm_kit
pip install -e .
```

## Verifying Installation

To verify that MCDM Kit has been installed correctly, you can run Python and try importing the package:

```python
import mcdm_kit
print(mcdm_kit.__version__)
```

If no error occurs, the installation was successful.

## Optional Dependencies

Some features of MCDM Kit may require additional dependencies:

-   For fuzzy set operations: `scikit-fuzzy`
-   For visualization: `matplotlib`, `seaborn`
-   For testing: `pytest`

You can install these optional dependencies using pip:

```bash
pip install scikit-fuzzy matplotlib seaborn pytest
```

## Troubleshooting

If you encounter any issues during installation, please check:

1. Your Python version (should be 3.7 or higher)
2. Your pip version (should be up to date)
3. Your system's package manager (if using Linux)

For more help, please open an issue on our GitHub repository.
