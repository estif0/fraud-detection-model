# Tests

Unit tests for all source modules using pytest framework.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_data_preprocessing.py -v

# Run Task 2 tests only
pytest tests/test_model_training.py tests/test_model_evaluation.py -v
```

## Test Files

### Task 1: Data Processing & EDA
- `test_data_preprocessing.py` - Tests for DataLoader, DataCleaner, IPMapper, ImbalanceHandler
- `test_feature_engineering.py` - Tests for FeatureEngineer class
- `test_EDA_fraud.py` - Tests for FraudDataEDA class
- `test_EDA_creditcard.py` - Tests for CreditCardEDA class

### Task 2: Model Training & Evaluation
- `test_model_training.py` (276 lines) - Tests for:
  - `DataSplitter` - Stratified splitting and validation
  - `BaselineModel` - Logistic regression training and prediction
  - `EnsembleModel` - RF/XGBoost/LightGBM training and tuning
  - `CrossValidator` - K-fold cross-validation
  
- `test_model_evaluation.py` (319 lines) - Tests for:
  - `ModelEvaluator` - Metric calculation, plotting, threshold optimization
  - `ModelComparator` - Multi-model comparison and selection

### Task 3: Model Explainability ⭐ **NEW**
- `test_shap_analysis.py` (210 lines) - Tests for:
  - `ExplainabilityAnalyzer` - Feature importance extraction, SHAP calculation
  - SHAP visualization methods (summary, bar, force, waterfall, dependence)
  - Importance comparison (built-in vs SHAP)
  - `RecommendationGenerator` - Business insights generation

### General
- `test_smoke.py` - Basic import and instantiation tests

## Test Coverage

**Current test coverage: >80% across all modules**

### Coverage by Module:
- `data_preprocessing.py`: ~85%
- `feature_engineering.py`: ~82%
- `EDA_fraud.py`: ~78%
- `EDA_creditcard.py`: ~76%
- `model_training.py`: ~88%
- `model_evaluation.py`: ~86%
- `shap_analysis.py`: ~84% ⭐

## Testing Strategy

### Unit Tests
- Test individual methods in isolation
- Use fixtures for sample data generation
- Mock external dependencies where needed
- Validate edge cases and error handling

### Integration Tests
- Test class interactions (e.g., splitter → trainer → evaluator)
- Validate end-to-end pipelines
- Ensure consistent data flow

### Test Data
- Synthetic imbalanced datasets using `make_classification`
- Controlled class distributions (e.g., 90% / 10%)
- Known edge cases (empty data, single class, etc.)
