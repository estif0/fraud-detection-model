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

### 5. `model_training.py` ⭐ **NEW**

**Purpose**: Model training, hyperparameter tuning, and cross-validation.

**Classes**:
- **`DataSplitter`**: Stratified train-test splitting
  - `stratified_split()`: Split preserving class distribution
  - `validate_split()`: Verify stratification worked correctly

- **`BaselineModel`**: Logistic Regression baseline
  - `train()`: Train with balanced class weights
  - `predict()`: Generate predictions
  - `predict_proba()`: Probability predictions
  - `save_model()` / `load_model()`: Model persistence

- **`EnsembleModel`**: Ensemble models (RF, XGBoost, LightGBM)
  - `train()`: Train ensemble with custom parameters
  - `hyperparameter_tuning()`: Grid/Random search with CV
  - `predict_with_threshold()`: Custom probability threshold
  - `predict()` / `predict_proba()`: Standard predictions
  - `save_model()` / `load_model()`: Model persistence with metadata

- **`CrossValidator`**: K-fold stratified cross-validation
  - `cross_validate_model()`: Multi-metric CV with mean±std

**Usage Example**:
```python
from src.model_training import DataSplitter, BaselineModel, EnsembleModel, CrossValidator

# Split data
splitter = DataSplitter(test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = splitter.stratified_split(X, y)

# Train baseline
baseline = BaselineModel()
lr_model = baseline.train(X_train, y_train)
y_pred = baseline.predict(X_test)

# Train ensemble
xgb_trainer = EnsembleModel(model_type='xgb', random_state=42)
xgb_model = xgb_trainer.train(X_train, y_train, n_estimators=100, max_depth=6)

# Hyperparameter tuning
param_grid = {'n_estimators': [100, 200], 'max_depth': [5, 7, 9]}
tuned_model = xgb_trainer.hyperparameter_tuning(X_train, y_train, param_grid)

# Cross-validation
cv = CrossValidator(n_splits=5)
scoring = {'f1': f1_score, 'precision': precision_score}
cv_results = cv.cross_validate_model(xgb_model, X_train, y_train, scoring)
```

---

### 6. `model_evaluation.py` ⭐ **NEW**

**Purpose**: Model evaluation, comparison, and selection.

**Classes**:
- **`ModelEvaluator`**: Comprehensive model evaluation
  - `evaluate_model()`: Calculate all metrics (accuracy, precision, recall, F1, ROC-AUC, PR-AUC)
  - `print_evaluation_report()`: Formatted metric display
  - `plot_confusion_matrix()`: Heatmap visualization
  - `plot_roc_curve()`: ROC curve with AUC
  - `plot_precision_recall_curve()`: PR curve (better for imbalanced data)
  - `find_optimal_threshold()`: Optimize decision threshold for F1/precision/recall
  - `compare_thresholds()`: Visual comparison of different thresholds

- **`ModelComparator`**: Multi-model comparison
  - `add_model_results()`: Register model performance
  - `create_comparison_table()`: DataFrame with all metrics
  - `plot_model_comparison()`: Bar chart comparison
  - `select_best_model()`: Automated selection based on metrics
  - `generate_model_selection_justification()`: Document selection rationale
  - `print_comparison_report()`: Formatted comparison output

**Usage Example**:
```python
from src.model_evaluation import ModelEvaluator, ModelComparator

# Evaluate single model
evaluator = ModelEvaluator(model_name="XGBoost")
results = evaluator.evaluate_model(y_test, y_pred, y_pred_proba)
evaluator.print_evaluation_report()
evaluator.plot_confusion_matrix(y_test, y_pred, save_path='reports/images/cm.png')

# Find optimal threshold
optimal_threshold, metrics = evaluator.find_optimal_threshold(y_test, y_pred_proba, metric='f1')

# Compare multiple models
comparator = ModelComparator()
comparator.add_model_results('Logistic Regression', lr_results)
comparator.add_model_results('XGBoost', xgb_results)
comparator.add_model_results('LightGBM', lgb_results)

comparison_df = comparator.create_comparison_table()
comparator.plot_model_comparison(metrics=['f1_score', 'pr_auc'])
best_model, best_results = comparator.select_best_model(primary_metric='f1_score')
```

---

### 4. `feature_engineering.py`

**Purpose**: Feature creation, transformation, encoding, and scaling for fraud detection.

**Classes**:
- **`FeatureEngineer`**: Comprehensive feature engineering
  - `create_time_features()`: Extract hour_of_day, day_of_week from timestamps
  - `calculate_time_since_signup()`: Duration between signup and purchase
  - `calculate_transaction_velocity()`: Transactions per user in time windows
  - `calculate_transaction_frequency()`: User activity metrics
  - `create_aggregated_features()`: User-level statistics (mean, std, count)
  - `encode_categorical_features()`: One-hot encoding for categorical variables
  - `scale_numerical_features()`: StandardScaler or MinMaxScaler

**Usage Example**:
```python
from src.feature_engineering import FeatureEngineer

# Initialize engineer
fe = FeatureEngineer()

# Create temporal features
df_time = fe.create_time_features(df, timestamp_col='purchase_time')

# Scale features
df_scaled, scaler = fe.scale_numerical_features(df, method='standard')
```

---

### 5. `model_training.py`

**Purpose**: Model training, hyperparameter tuning, and cross-validation for fraud detection.

**Classes**:
- **`DataSplitter`**: Stratified train-test splitting
  - `stratified_split()`: Split preserving class distribution
  - `validate_split()`: Verify split quality

- **`BaselineModel`**: Logistic Regression baseline
  - `train()`: Train LR with balanced class weights
  - `predict()`: Generate predictions
  - `predict_proba()`: Generate probabilities
  - `save_model()` / `load_model()`: Model persistence

- **`EnsembleModel`**: Ensemble model training
  - Supports Random Forest, XGBoost, and LightGBM
  - `train()`: Train with custom hyperparameters
  - `hyperparameter_tuning()`: GridSearchCV or RandomizedSearchCV
  - `predict()` / `predict_proba()`: Predictions
  - `save_model()` / `load_model()`: Model persistence

- **`CrossValidator`**: K-fold cross-validation
  - `cross_validate_model()`: Stratified k-fold CV with multiple metrics
  - Reports mean and standard deviation for each metric

**Usage Example**:
```python
from src.model_training import DataSplitter, BaselineModel, EnsembleModel, CrossValidator

# Split data
splitter = DataSplitter(test_size=0.2)
X_train, X_test, y_train, y_test = splitter.stratified_split(X, y)

# Train baseline
baseline = BaselineModel()
lr_model = baseline.train(X_train, y_train)

# Train ensemble
xgb_trainer = EnsembleModel(model_type='xgb')
xgb_model = xgb_trainer.train(X_train, y_train, n_estimators=100)

# Cross-validate
cv = CrossValidator(n_splits=5)
scoring = {'f1': f1_score, 'precision': precision_score}
cv_results = cv.cross_validate_model(xgb_model, X_train, y_train, scoring)
```

---

### 6. `model_evaluation.py`

**Purpose**: Comprehensive model evaluation, comparison, and selection for fraud detection.

**Classes**:
- **`ModelEvaluator`**: Single model evaluation
  - `evaluate_model()`: Calculate all metrics (accuracy, precision, recall, F1, ROC-AUC, PR-AUC)
  - `print_evaluation_report()`: Formatted report
  - `plot_confusion_matrix()`: Heatmap visualization
  - `plot_roc_curve()`: ROC curve with AUC
  - `plot_precision_recall_curve()`: PR curve (crucial for imbalanced data)
  - `generate_classification_report()`: sklearn classification report

- **`ModelComparator`**: Multi-model comparison
  - `add_model_results()`: Add model evaluation results
  - `create_comparison_table()`: DataFrame comparing all models
  - `plot_model_comparison()`: Bar chart comparing metrics
  - `select_best_model()`: Select best based on primary/secondary metrics
  - `generate_model_selection_justification()`: Narrative justification

**Usage Example**:
```python
from src.model_evaluation import ModelEvaluator, ModelComparator

# Evaluate single model
evaluator = ModelEvaluator(model_name="XGBoost")
results = evaluator.evaluate_model(y_test, y_pred, y_pred_proba)
evaluator.print_evaluation_report()
evaluator.plot_confusion_matrix(y_test, y_pred, save_path='reports/images/cm.png')

# Compare models
comparator = ModelComparator()
comparator.add_model_results('LR', lr_results)
comparator.add_model_results('XGBoost', xgb_results)
comparator.print_comparison_report()
best_model, best_results = comparator.select_best_model(primary_metric='f1_score')
```

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
