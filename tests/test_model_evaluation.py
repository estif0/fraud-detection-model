"""Unit tests for model_evaluation module.

This module contains tests for:
- ModelEvaluator
- ModelComparator
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import sys
import os
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model_evaluation import ModelEvaluator, ModelComparator


@pytest.fixture
def sample_predictions():
    """Create sample predictions for testing."""
    np.random.seed(42)

    # Create true labels (imbalanced)
    y_true = np.array([0] * 90 + [1] * 10)

    # Create predictions (some correct, some incorrect)
    y_pred = y_true.copy()
    # Introduce some errors
    y_pred[5] = 1  # False positive
    y_pred[95] = 0  # False negative

    # Create prediction probabilities
    y_pred_proba = np.random.rand(100)
    y_pred_proba[y_true == 1] += 0.3  # Fraud cases have higher probabilities
    y_pred_proba = np.clip(y_pred_proba, 0, 1)

    return pd.Series(y_true), y_pred, y_pred_proba


class TestModelEvaluator:
    """Test cases for ModelEvaluator class."""

    def test_initialization(self):
        """Test ModelEvaluator initialization."""
        evaluator = ModelEvaluator(model_name="TestModel")
        assert evaluator.model_name == "TestModel"
        assert evaluator.evaluation_results == {}

    def test_evaluate_model_basic(self, sample_predictions):
        """Test basic model evaluation."""
        y_true, y_pred, y_pred_proba = sample_predictions
        evaluator = ModelEvaluator(model_name="TestModel")

        results = evaluator.evaluate_model(y_true, y_pred, y_pred_proba)

        # Check all expected metrics are present
        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results
        assert "f1_score" in results
        assert "roc_auc" in results
        assert "pr_auc" in results
        assert "true_positives" in results
        assert "true_negatives" in results
        assert "false_positives" in results
        assert "false_negatives" in results
        assert "specificity" in results

        # Check metric values are in valid range
        assert 0 <= results["accuracy"] <= 1
        assert 0 <= results["precision"] <= 1
        assert 0 <= results["recall"] <= 1
        assert 0 <= results["f1_score"] <= 1

    def test_evaluate_model_without_proba(self, sample_predictions):
        """Test evaluation without probability predictions."""
        y_true, y_pred, _ = sample_predictions
        evaluator = ModelEvaluator(model_name="TestModel")

        results = evaluator.evaluate_model(y_true, y_pred, y_pred_proba=None)

        # Check basic metrics are present
        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results
        assert "f1_score" in results

        # Check probability-based metrics are not present
        assert "roc_auc" not in results
        assert "pr_auc" not in results

    def test_print_evaluation_report(self, sample_predictions, capsys):
        """Test printing evaluation report."""
        y_true, y_pred, y_pred_proba = sample_predictions
        evaluator = ModelEvaluator(model_name="TestModel")

        results = evaluator.evaluate_model(y_true, y_pred, y_pred_proba)
        evaluator.print_evaluation_report(results)

        captured = capsys.readouterr()
        assert "TestModel" in captured.out
        assert "Accuracy" in captured.out
        assert "Precision" in captured.out

    def test_plot_confusion_matrix(self, sample_predictions, tmp_path):
        """Test confusion matrix plotting."""
        y_true, y_pred, _ = sample_predictions
        evaluator = ModelEvaluator(model_name="TestModel")

        save_path = tmp_path / "confusion_matrix.png"
        fig = evaluator.plot_confusion_matrix(y_true, y_pred, save_path=str(save_path))

        assert fig is not None
        assert save_path.exists()

    def test_plot_roc_curve(self, sample_predictions, tmp_path):
        """Test ROC curve plotting."""
        y_true, _, y_pred_proba = sample_predictions
        evaluator = ModelEvaluator(model_name="TestModel")

        save_path = tmp_path / "roc_curve.png"
        fig = evaluator.plot_roc_curve(y_true, y_pred_proba, save_path=str(save_path))

        assert fig is not None
        assert save_path.exists()

    def test_plot_precision_recall_curve(self, sample_predictions, tmp_path):
        """Test PR curve plotting."""
        y_true, _, y_pred_proba = sample_predictions
        evaluator = ModelEvaluator(model_name="TestModel")

        save_path = tmp_path / "pr_curve.png"
        fig = evaluator.plot_precision_recall_curve(
            y_true, y_pred_proba, save_path=str(save_path)
        )

        assert fig is not None
        assert save_path.exists()

    def test_generate_classification_report(self, sample_predictions):
        """Test classification report generation."""
        y_true, y_pred, _ = sample_predictions
        evaluator = ModelEvaluator(model_name="TestModel")

        report = evaluator.generate_classification_report(y_true, y_pred)

        assert isinstance(report, str)
        assert "precision" in report
        assert "recall" in report
        assert "f1-score" in report


class TestModelComparator:
    """Test cases for ModelComparator class."""

    @pytest.fixture
    def sample_model_results(self):
        """Create sample model results for comparison."""
        results1 = {
            "accuracy": 0.95,
            "precision": 0.80,
            "recall": 0.75,
            "f1_score": 0.77,
            "roc_auc": 0.90,
            "pr_auc": 0.85,
            "specificity": 0.96,
        }

        results2 = {
            "accuracy": 0.93,
            "precision": 0.85,
            "recall": 0.78,
            "f1_score": 0.81,
            "roc_auc": 0.92,
            "pr_auc": 0.88,
            "specificity": 0.94,
        }

        results3 = {
            "accuracy": 0.94,
            "precision": 0.82,
            "recall": 0.80,
            "f1_score": 0.81,
            "roc_auc": 0.91,
            "pr_auc": 0.87,
            "specificity": 0.95,
        }

        return results1, results2, results3

    def test_initialization(self):
        """Test ModelComparator initialization."""
        comparator = ModelComparator()
        assert comparator.models_results == {}

    def test_add_model_results(self, sample_model_results):
        """Test adding model results."""
        results1, results2, results3 = sample_model_results
        comparator = ModelComparator()

        comparator.add_model_results("Model1", results1)
        comparator.add_model_results("Model2", results2)
        comparator.add_model_results("Model3", results3)

        assert len(comparator.models_results) == 3
        assert "Model1" in comparator.models_results
        assert "Model2" in comparator.models_results
        assert "Model3" in comparator.models_results

    def test_create_comparison_table(self, sample_model_results):
        """Test comparison table creation."""
        results1, results2, results3 = sample_model_results
        comparator = ModelComparator()

        comparator.add_model_results("Model1", results1)
        comparator.add_model_results("Model2", results2)
        comparator.add_model_results("Model3", results3)

        df = comparator.create_comparison_table(
            metrics=["f1_score", "precision", "recall"]
        )

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 3)  # 3 models, 3 metrics
        assert "f1_score" in df.columns
        assert "precision" in df.columns
        assert "recall" in df.columns

    def test_select_best_model(self, sample_model_results):
        """Test best model selection."""
        results1, results2, results3 = sample_model_results
        comparator = ModelComparator()

        comparator.add_model_results("Model1", results1)
        comparator.add_model_results("Model2", results2)
        comparator.add_model_results("Model3", results3)

        best_name, best_results = comparator.select_best_model(
            primary_metric="f1_score", secondary_metric="pr_auc"
        )

        # Model2 and Model3 both have f1_score of 0.81
        # Model2 has higher pr_auc (0.88 vs 0.87), so it should be selected
        assert best_name == "Model2"
        assert best_results["f1_score"] == 0.81

    def test_select_best_model_empty(self):
        """Test best model selection with no models."""
        comparator = ModelComparator()

        with pytest.raises(ValueError):
            comparator.select_best_model()

    def test_plot_model_comparison(self, sample_model_results, tmp_path):
        """Test model comparison plotting."""
        results1, results2, results3 = sample_model_results
        comparator = ModelComparator()

        comparator.add_model_results("Model1", results1)
        comparator.add_model_results("Model2", results2)
        comparator.add_model_results("Model3", results3)

        save_path = tmp_path / "comparison.png"
        fig = comparator.plot_model_comparison(
            metrics=["f1_score", "precision", "recall"], save_path=str(save_path)
        )

        assert fig is not None
        assert save_path.exists()

    def test_generate_model_selection_justification(self, sample_model_results):
        """Test justification generation."""
        results1, results2, results3 = sample_model_results
        comparator = ModelComparator()

        comparator.add_model_results("Model1", results1)
        comparator.add_model_results("Model2", results2)

        justification = comparator.generate_model_selection_justification(
            "Model2", interpretability_notes="Model is highly interpretable."
        )

        assert isinstance(justification, str)
        assert "Model2" in justification
        assert "F1-Score" in justification
        assert "interpretable" in justification.lower()
