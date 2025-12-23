"""Unit tests for model_training module.

This module contains tests for:
- DataSplitter
- BaselineModel
- EnsembleModel
- CrossValidator
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_training import DataSplitter, BaselineModel, EnsembleModel, CrossValidator


@pytest.fixture
def sample_data():
    """Create sample imbalanced classification data for testing."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.9, 0.1],  # Imbalanced
        random_state=42
    )
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y_series = pd.Series(y, name='target')
    return X_df, y_series


class TestDataSplitter:
    """Test cases for DataSplitter class."""
    
    def test_initialization(self):
        """Test DataSplitter initialization."""
        splitter = DataSplitter(test_size=0.3, random_state=42)
        assert splitter.test_size == 0.3
        assert splitter.random_state == 42
    
    def test_stratified_split(self, sample_data):
        """Test stratified train-test split."""
        X, y = sample_data
        splitter = DataSplitter(test_size=0.2, random_state=42)
        
        X_train, X_test, y_train, y_test = splitter.stratified_split(X, y)
        
        # Check shapes
        assert X_train.shape[0] == int(len(X) * 0.8)
        assert X_test.shape[0] == int(len(X) * 0.2)
        assert y_train.shape[0] == X_train.shape[0]
        assert y_test.shape[0] == X_test.shape[0]
    
    def test_validate_split(self, sample_data):
        """Test split validation."""
        X, y = sample_data
        splitter = DataSplitter(test_size=0.2, random_state=42)
        
        X_train, X_test, y_train, y_test = splitter.stratified_split(X, y)
        validation_report = splitter.validate_split(y_train, y_test)
        
        # Check report structure
        assert 'train_size' in validation_report
        assert 'test_size' in validation_report
        assert 'train_class_distribution' in validation_report
        assert 'test_class_distribution' in validation_report
        assert 'split_is_valid' in validation_report
        
        # Check split is valid
        assert validation_report['split_is_valid'] == True


class TestBaselineModel:
    """Test cases for BaselineModel class."""
    
    def test_initialization(self):
        """Test BaselineModel initialization."""
        model = BaselineModel(random_state=42, max_iter=500)
        assert model.random_state == 42
        assert model.max_iter == 500
        assert model.model is None
    
    def test_train(self, sample_data):
        """Test model training."""
        X, y = sample_data
        model = BaselineModel(random_state=42)
        
        trained_model = model.train(X, y, class_weight='balanced')
        
        assert trained_model is not None
        assert isinstance(trained_model, LogisticRegression)
        assert model.model is not None
    
    def test_predict(self, sample_data):
        """Test prediction."""
        X, y = sample_data
        model = BaselineModel(random_state=42)
        model.train(X, y)
        
        predictions = model.predict(X)
        
        assert len(predictions) == len(X)
        assert set(predictions).issubset({0, 1})
    
    def test_predict_proba(self, sample_data):
        """Test probability prediction."""
        X, y = sample_data
        model = BaselineModel(random_state=42)
        model.train(X, y)
        
        probabilities = model.predict_proba(X)
        
        assert probabilities.shape == (len(X), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_predict_without_training(self, sample_data):
        """Test that prediction fails without training."""
        X, y = sample_data
        model = BaselineModel(random_state=42)
        
        with pytest.raises(ValueError):
            model.predict(X)


class TestEnsembleModel:
    """Test cases for EnsembleModel class."""
    
    def test_initialization_rf(self):
        """Test Random Forest initialization."""
        model = EnsembleModel(model_type='rf', random_state=42)
        assert model.model_type == 'rf'
        assert model.random_state == 42
        assert model.model is None
    
    def test_initialization_xgb(self):
        """Test XGBoost initialization."""
        model = EnsembleModel(model_type='xgb', random_state=42)
        assert model.model_type == 'xgb'
    
    def test_initialization_lgb(self):
        """Test LightGBM initialization."""
        model = EnsembleModel(model_type='lgb', random_state=42)
        assert model.model_type == 'lgb'
    
    def test_invalid_model_type(self):
        """Test that invalid model type raises error."""
        with pytest.raises(ValueError):
            EnsembleModel(model_type='invalid')
    
    def test_train_random_forest(self, sample_data):
        """Test Random Forest training."""
        X, y = sample_data
        model = EnsembleModel(model_type='rf', random_state=42)
        
        trained_model = model.train(X, y, n_estimators=10)
        
        assert trained_model is not None
        assert isinstance(trained_model, RandomForestClassifier)
    
    def test_train_xgboost(self, sample_data):
        """Test XGBoost training."""
        X, y = sample_data
        model = EnsembleModel(model_type='xgb', random_state=42)
        
        trained_model = model.train(X, y, n_estimators=10)
        
        assert trained_model is not None
        assert isinstance(trained_model, xgb.XGBClassifier)
    
    def test_train_lightgbm(self, sample_data):
        """Test LightGBM training."""
        X, y = sample_data
        model = EnsembleModel(model_type='lgb', random_state=42)
        
        trained_model = model.train(X, y, n_estimators=10)
        
        assert trained_model is not None
        assert isinstance(trained_model, lgb.LGBMClassifier)
    
    def test_predict(self, sample_data):
        """Test ensemble model prediction."""
        X, y = sample_data
        model = EnsembleModel(model_type='rf', random_state=42)
        model.train(X, y, n_estimators=10)
        
        predictions = model.predict(X)
        
        assert len(predictions) == len(X)
        assert set(predictions).issubset({0, 1})
    
    def test_predict_proba(self, sample_data):
        """Test ensemble model probability prediction."""
        X, y = sample_data
        model = EnsembleModel(model_type='rf', random_state=42)
        model.train(X, y, n_estimators=10)
        
        probabilities = model.predict_proba(X)
        
        assert probabilities.shape == (len(X), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)


class TestCrossValidator:
    """Test cases for CrossValidator class."""
    
    def test_initialization(self):
        """Test CrossValidator initialization."""
        cv = CrossValidator(n_splits=3, random_state=42)
        assert cv.n_splits == 3
        assert cv.random_state == 42
    
    def test_cross_validate_model(self, sample_data):
        """Test cross-validation."""
        X, y = sample_data
        cv = CrossValidator(n_splits=3, random_state=42)
        
        from sklearn.metrics import f1_score, precision_score
        scoring_functions = {
            'f1': f1_score,
            'precision': precision_score
        }
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        results = cv.cross_validate_model(model, X, y, scoring_functions)
        
        # Check result structure
        assert 'f1_mean' in results
        assert 'f1_std' in results
        assert 'f1_scores' in results
        assert 'precision_mean' in results
        assert 'precision_std' in results
        assert 'precision_scores' in results
        
        # Check scores are reasonable
        assert 0 <= results['f1_mean'] <= 1
        assert results['f1_std'] >= 0
        assert len(results['f1_scores']) == 3
