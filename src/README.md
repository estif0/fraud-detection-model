# Source Code Modules

This directory contains all the core Python modules for the fraud detection project, implementing data preprocessing, feature engineering, model training, evaluation, and explainability analysis.

## Module Overview

### 1. `data_preprocessing.py`

**Purpose**: Data loading, cleaning, validation, and preprocessing operations.

**Classes**:
- **`DataLoader`**: Load transaction data from CSV files
  - `load_fraud_data()`: Load e-commerce fraud transaction data
  - `load_ip_mapping()`: Load IP address to country mapping
  - `load_creditcard_data()`: Load credit card transaction data
  - `save_processed_data()`: Save processed data to CSV
  - `get_data_info()`: Get summary information about datasets

- **`DataCleaner`**: Clean and validate data
  - `check_missing_values()`: Identify missing data with summary statistics
  - `handle_missing_values()`: Multiple strategies (drop, fill, mean, median, mode)
  - `remove_duplicates()`: Remove duplicate rows with flexible options
  - `validate_data_types()`: Validate and convert data types
  - `generate_cleaning_report()`: Comprehensive cleaning summary

- **`IPMapper`** *(Coming in Step 1.5)*: Map IP addresses to countries
  - Convert IP strings to integers
  - Range-based country lookup
  - Geolocation analysis

- **`ImbalanceHandler`** *(Coming in Step 1.7)*: Handle class imbalance
  - SMOTE oversampling
  - Random undersampling
  - Combined sampling strategies

**Usage Example**:
```python
from src.data_preprocessing import DataLoader

# Initialize loader
loader = DataLoader(data_dir='data/raw', processed_dir='data/processed')

# Load data
fraud_data = loader.load_fraud_data()
ip_mapping = loader.load_ip_mapping()
creditcard_data = loader.load_creditcard_data()

# Save processed data
loader.save_processed_data(fraud_data, 'cleaned_fraud_data.csv')
```

---

### 2. `EDA_fraud.py`

**Purpose**: Exploratory data analysis for e-commerce fraud data.

**Classes**:
- **`FraudDataEDA`**: Comprehensive EDA for fraud dataset
  - `univariate_analysis()`: Distribution plots and statistics for individual features
  - `bivariate_analysis()`: Feature relationships with fraud label
  - `analyze_class_imbalance()`: Calculate and visualize class imbalance
  - `temporal_analysis()`: Fraud patterns by hour/day
  - `categorical_analysis()`: Detailed categorical feature analysis
  - `generate_eda_report()`: Comprehensive EDA summary report

**Usage Example**:
```python
from src.EDA_fraud import FraudDataEDA

# Initialize EDA analyzer
eda = FraudDataEDA(fraud_data, target_column='class', output_dir='reports/images')

# Run analyses
eda.univariate_analysis()
eda.bivariate_analysis()
eda.analyze_class_imbalance()
eda.temporal_analysis()
eda.categorical_analysis()

# Generate comprehensive report
report = eda.generate_eda_report()
```

---

### 3. `EDA_creditcard.py`

**Purpose**: Exploratory data analysis for credit card transaction data.

**Classes**:
- **`CreditCardEDA`**: Comprehensive EDA for credit card dataset
  - `pca_features_analysis()`: Analyze V1-V28 distributions
  - `amount_analysis()`: Transaction amount patterns
  - `time_analysis()`: Temporal patterns from `Time`
  - `correlation_analysis()`: Feature correlations
  - `analyze_class_imbalance()`: Fraud vs legitimate ratio
  - `generate_eda_report()`: Summary insights

---

### 4. `feature_engineering.py`

**Purpose**: Create and transform features for modeling.

**Classes**:
- **`FeatureEngineer`**: Feature creation and transformation
  - `create_time_features()`: hour_of_day, day_of_week
  - `calculate_time_since_signup()`: minutes since signup
  - `calculate_transaction_velocity()`: rolling counts per user
  - `calculate_transaction_frequency()`: transactions per user
  - `create_aggregated_features()`: user-level statistics
  - `encode_categorical_features()`: One-hot encoding
  - `scale_numerical_features()`: Standard/MinMax scaling

---

### 5. `model_training.py` *(Coming in Step 2.1)*

**Purpose**: Train machine learning models for fraud detection.

**Classes**:
- **`ModelTrainer`**: Data preparation and splitting
- **`BaselineModel`**: Logistic regression baseline
- **`EnsembleModel`**: Random Forest, XGBoost, LightGBM
- **`CrossValidator`**: Stratified K-fold cross-validation

---

### 6. `model_evaluation.py` *(Coming in Step 2.2)*

**Purpose**: Evaluate model performance with appropriate metrics.

**Classes**:
- **`ModelEvaluator`**: Comprehensive model evaluation
  - AUC-PR, F1-Score, Confusion Matrix
  - ROC and PR curves
  - Classification reports
- **`ModelComparator`**: Compare multiple models

---

### 7. `shap_analysis.py` *(Coming in Step 3.1)*

**Purpose**: Model explainability using SHAP values.

**Classes**:
- **`ExplainabilityAnalyzer`**: SHAP-based model interpretation
  - Feature importance extraction
  - Global SHAP analysis
  - Local prediction explanations
- **`RecommendationGenerator`**: Business insights from SHAP

---

## Architecture Principles

### Object-Oriented Design
All modules follow OOP principles:
- **Encapsulation**: Related functionality grouped in classes
- **Single Responsibility**: Each class has one clear purpose
- **Reusability**: Classes can be imported and used across notebooks and scripts
- **Testability**: All classes have corresponding unit tests

### Documentation Standards
Every module includes:
- Module-level docstring explaining purpose
- Class docstrings with attributes and examples
- Method docstrings with Args, Returns, Raises, and Examples
- Inline comments for complex logic

### Code Quality
- Type hints for function parameters and returns
- Consistent naming conventions (PEP 8)
- Error handling with informative messages
- Logging for tracking operations

---

## Development Workflow

### Adding a New Module

1. **Create module file** in `src/` directory
2. **Define classes** with proper OOP structure
3. **Add comprehensive docstrings** (Google/NumPy style)
4. **Write unit tests** in `tests/` directory
5. **Update this README** with module documentation
6. **Use in notebooks** by importing from `src/`

### Testing

All modules have corresponding test files in `tests/` directory:
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_data_preprocessing.py -v

# Check code coverage
pytest --cov=src tests/
```

---

## Dependencies

Core libraries used across modules:
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **scikit-learn**: Machine learning models and preprocessing
- **imbalanced-learn**: Class imbalance handling
- **xgboost**: Gradient boosting models
- **lightgbm**: LightGBM models
- **shap**: Model explainability
- **matplotlib/seaborn**: Visualization

See `requirements.txt` for complete dependency list.

---

## Module Status

| Module                   | Status     | Step    | Description                          |
| ------------------------ | ---------- | ------- | ------------------------------------ |
| `data_preprocessing.py`  | ✅ Complete | 1.1-1.2 | DataLoader & DataCleaner implemented |
| `EDA_fraud.py`           | ✅ Complete | 1.3     | FraudDataEDA with 6 analysis methods |
| `EDA_creditcard.py`      | ✅ Complete | 1.4     | Credit card EDA                      |
| `feature_engineering.py` | ✅ Complete | 1.6     | Feature creation                     |
| `model_training.py`      | ⏳ Pending  | 2.1     | Model training                       |
| `model_evaluation.py`    | ⏳ Pending  | 2.2     | Model evaluation                     |
| `shap_analysis.py`       | ⏳ Pending  | 3.1     | Explainability                       |

---

## Contact

For questions or issues with any module, please refer to the main project README or documentation in `docs/local/`.
