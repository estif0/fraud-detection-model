# Notebooks

Jupyter notebooks for exploratory data analysis, model development, and evaluation.

## Notebooks

### Task 1: Exploratory Data Analysis ✅

1. **01-eda-fraud-data.ipynb** - E-commerce fraud data analysis
   - Data loading and inspection
   - Data quality assessment and cleaning
   - Univariate and bivariate analysis
   - Class imbalance analysis
   - Temporal pattern exploration
   - Geographic fraud patterns (IP to country mapping)

2. **02-eda-creditcard.ipynb** - Credit card transaction analysis
   - PCA features analysis (V1-V28)
   - Amount and time pattern analysis
   - Correlation analysis
   - Class distribution visualization
   - Extreme imbalance handling strategies

### Task 2: Model Building & Training ✅

3. **04-modeling.ipynb** - Fraud detection model development
   - Data preparation and stratified train-test split
   - Feature-target separation
   - Baseline model (Logistic Regression)
   - Ensemble models (Random Forest, XGBoost, LightGBM)
   - Hyperparameter tuning (GridSearchCV)
   - Optimal threshold optimization
   - 5-fold stratified cross-validation
   - Comprehensive model comparison and selection
   - Model persistence and versioning

### Task 3: Model Explainability ✅

4. **05-shap-explainability.ipynb** - Model interpretation and business insights
   - Model and test data loading
   - Built-in feature importance extraction and visualization
   - SHAP value calculation (TreeExplainer)
   - SHAP global analysis:
     - Summary plots (beeswarm)
     - Bar plots (mean absolute SHAP values)
     - Dependence plots (feature interactions)
   - SHAP local analysis:
     - Force plots for True Positives, False Positives, False Negatives
     - Waterfall plots showing feature contributions
   - Comparison: Built-in vs SHAP importance rankings
   - Top fraud driver identification
   - Business recommendations generation
   - Report export and visualization saving

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
