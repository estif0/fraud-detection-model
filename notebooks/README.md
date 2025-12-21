# Notebooks

Jupyter notebooks for exploratory data analysis and model development.

## Notebooks

1. **01-eda-fraud-data.ipynb** - E-commerce fraud data analysis
   - Data loading and inspection
   - Data quality assessment and cleaning
   - Univariate and bivariate analysis
   - Class imbalance analysis
   - Temporal pattern exploration

2. **02-eda-creditcard.ipynb** - Credit card transaction analysis
   - PCA features analysis (V1-V28)
   - Amount and time pattern analysis
   - Correlation analysis
   - Class distribution visualization

## Running Notebooks

```bash
# Start Jupyter
jupyter notebook

# Or use JupyterLab
jupyter lab
```

Navigate to the notebooks directory and open the desired notebook.

## Note

All notebooks import reusable code from `src/` modules rather than duplicating logic. This ensures consistency and maintainability.
