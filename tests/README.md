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
```

## Test Files

- `test_data_preprocessing.py` - Tests for DataLoader, DataCleaner, IPMapper, ImbalanceHandler
- `test_feature_engineering.py` - Tests for FeatureEngineer class
- `test_EDA_fraud.py` - Tests for FraudDataEDA class
- `test_EDA_creditcard.py` - Tests for CreditCardEDA class
- `test_smoke.py` - Basic import and instantiation tests

## Coverage

Current test coverage: **>80%** across all modules.
