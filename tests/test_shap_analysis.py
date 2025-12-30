"""Unit tests for shap_analysis module.

This module contains tests for the ExplainabilityAnalyzer and
RecommendationGenerator classes.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from src.shap_analysis import ExplainabilityAnalyzer, RecommendationGenerator


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5

    # Create feature data
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    # Create binary target
    y = pd.Series(np.random.randint(0, 2, n_samples), name="target")

    return X, y


@pytest.fixture
def trained_model(sample_data):
    """Create a trained model for testing."""
    X, y = sample_data

    # Split data
    split_idx = int(0.7 * len(X))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Train simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3)
    model.fit(X_train, y_train)

    return model, X_test, y_test


@pytest.fixture
def analyzer(trained_model):
    """Create an ExplainabilityAnalyzer instance."""
    model, X_test, y_test = trained_model
    return ExplainabilityAnalyzer(model, X_test, y_test)


class TestExplainabilityAnalyzer:
    """Test cases for ExplainabilityAnalyzer class."""

    def test_initialization(self, trained_model):
        """Test analyzer initialization."""
        model, X_test, y_test = trained_model
        analyzer = ExplainabilityAnalyzer(model, X_test, y_test)

        assert analyzer.model is model
        assert analyzer.X_test.equals(X_test)
        assert analyzer.y_test.equals(y_test)
        assert analyzer.feature_names == list(X_test.columns)
        assert analyzer.shap_values is None
        assert analyzer.explainer is None

    def test_extract_feature_importance(self, analyzer):
        """Test feature importance extraction."""
        importance_df = analyzer.extract_feature_importance()

        assert isinstance(importance_df, pd.DataFrame)
        assert "feature" in importance_df.columns
        assert "importance" in importance_df.columns
        assert len(importance_df) == len(analyzer.feature_names)
        # Check sorted descending
        assert importance_df["importance"].is_monotonic_decreasing

    def test_extract_feature_importance_no_attribute(self):
        """Test error when model has no feature_importances_."""

        class DummyModel:
            def predict(self, X):
                return np.zeros(len(X))

        X = pd.DataFrame(np.random.randn(10, 3), columns=["a", "b", "c"])
        y = pd.Series(np.zeros(10))

        analyzer = ExplainabilityAnalyzer(DummyModel(), X, y)

        with pytest.raises(AttributeError):
            analyzer.extract_feature_importance()

    def test_plot_feature_importance(self, analyzer, tmp_path):
        """Test feature importance plotting."""
        save_path = tmp_path / "importance.png"

        fig = analyzer.plot_feature_importance(top_n=5, save_path=str(save_path))

        assert isinstance(fig, plt.Figure)
        assert save_path.exists()
        plt.close(fig)

    def test_compare_importance_methods(self, analyzer):
        """Test comparison of importance methods."""
        comparison_df = analyzer.compare_importance_methods()

        assert isinstance(comparison_df, pd.DataFrame)
        assert "feature" in comparison_df.columns
        assert "importance" in comparison_df.columns

    def test_initialize_shap_explainer(self, analyzer):
        """Test SHAP explainer initialization."""
        analyzer.initialize_shap_explainer()

        assert analyzer.explainer is not None

    def test_calculate_shap_values(self, analyzer):
        """Test SHAP values calculation."""
        analyzer.initialize_shap_explainer()
        shap_values = analyzer.calculate_shap_values(max_samples=20)

        assert shap_values is not None
        assert isinstance(shap_values, np.ndarray)
        assert shap_values.shape[0] == 20  # max_samples
        assert shap_values.shape[1] == len(analyzer.feature_names)

    def test_calculate_shap_values_without_explainer(self, analyzer):
        """Test error when SHAP values calculated without explainer."""
        with pytest.raises(ValueError, match="Explainer not initialized"):
            analyzer.calculate_shap_values()

    def test_plot_shap_summary(self, analyzer, tmp_path):
        """Test SHAP summary plot creation."""
        analyzer.initialize_shap_explainer()
        analyzer.calculate_shap_values(max_samples=20)

        save_path = tmp_path / "shap_summary.png"

        # Just test that it doesn't raise errors
        # Visual output testing is difficult in unit tests
        try:
            analyzer.plot_shap_summary(max_display=5, save_path=str(save_path))
            plt.close("all")
        except Exception as e:
            pytest.fail(f"plot_shap_summary raised {e}")

    def test_plot_shap_summary_without_values(self, analyzer):
        """Test error when plotting without SHAP values."""
        with pytest.raises(ValueError, match="SHAP values not calculated"):
            analyzer.plot_shap_summary()

    def test_analyze_prediction(self, analyzer):
        """Test individual prediction analysis."""
        analyzer.initialize_shap_explainer()
        analyzer.calculate_shap_values(max_samples=20)

        analysis = analyzer.analyze_prediction(sample_index=0)

        assert isinstance(analysis, dict)
        assert "sample_index" in analysis
        assert "prediction" in analysis
        assert "prediction_proba" in analysis
        assert "actual_label" in analysis
        assert "is_correct" in analysis
        assert "prediction_type" in analysis
        assert "feature_contributions" in analysis
        assert "top_5_contributors" in analysis

        # Check prediction type is valid
        assert analysis["prediction_type"] in [
            "True Positive (TP)",
            "True Negative (TN)",
            "False Positive (FP)",
            "False Negative (FN)",
        ]

    def test_get_prediction_type(self, analyzer):
        """Test prediction type classification."""
        assert analyzer._get_prediction_type(1, 1) == "True Positive (TP)"
        assert analyzer._get_prediction_type(0, 0) == "True Negative (TN)"
        assert analyzer._get_prediction_type(1, 0) == "False Positive (FP)"
        assert analyzer._get_prediction_type(0, 1) == "False Negative (FN)"

    def test_compare_importance_rankings(self, analyzer):
        """Test importance ranking comparison."""
        analyzer.initialize_shap_explainer()
        analyzer.calculate_shap_values(max_samples=20)

        comparison = analyzer.compare_importance_rankings(top_n=5)

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 5
        assert "feature" in comparison.columns
        assert "builtin_importance" in comparison.columns
        assert "builtin_rank" in comparison.columns
        assert "shap_importance" in comparison.columns
        assert "shap_rank" in comparison.columns
        assert "rank_difference" in comparison.columns

    def test_identify_discrepancies(self, analyzer):
        """Test discrepancy identification."""
        analyzer.initialize_shap_explainer()
        analyzer.calculate_shap_values(max_samples=20)

        discrepancies = analyzer.identify_discrepancies(
            threshold=0
        )  # Low threshold to ensure results

        assert isinstance(discrepancies, pd.DataFrame)
        # Check all discrepancies meet threshold
        if len(discrepancies) > 0:
            assert all(discrepancies["rank_difference"] >= 0)

    def test_plot_importance_comparison(self, analyzer, tmp_path):
        """Test importance comparison plotting."""
        analyzer.initialize_shap_explainer()
        analyzer.calculate_shap_values(max_samples=20)

        save_path = tmp_path / "comparison.png"

        fig = analyzer.plot_importance_comparison(top_n=5, save_path=str(save_path))

        assert isinstance(fig, plt.Figure)
        assert save_path.exists()
        plt.close(fig)


class TestRecommendationGenerator:
    """Test cases for RecommendationGenerator class."""

    @pytest.fixture
    def generator(self, analyzer):
        """Create a RecommendationGenerator instance."""
        analyzer.initialize_shap_explainer()
        analyzer.calculate_shap_values(max_samples=20)
        return RecommendationGenerator(analyzer)

    def test_initialization(self, generator, analyzer):
        """Test generator initialization."""
        assert generator.analyzer is analyzer
        assert generator.fraud_drivers is None

    def test_identify_fraud_drivers(self, generator):
        """Test fraud driver identification."""
        drivers = generator.identify_fraud_drivers(top_n=3)

        assert isinstance(drivers, pd.DataFrame)
        assert len(drivers) == 3
        assert "feature" in drivers.columns
        assert "mean_abs_shap" in drivers.columns
        assert "importance_score" in drivers.columns
        # Check sorted descending
        assert drivers["mean_abs_shap"].is_monotonic_decreasing

    def test_generate_recommendations(self, generator):
        """Test recommendation generation."""
        recommendations = generator.generate_recommendations()

        assert isinstance(recommendations, list)
        assert len(recommendations) == 5  # Default top_n

        for rec in recommendations:
            assert isinstance(rec, dict)
            assert "feature" in rec
            assert "importance" in rec
            assert "insight" in rec
            assert "action" in rec
            assert isinstance(rec["insight"], str)
            assert isinstance(rec["action"], str)
            assert len(rec["insight"]) > 0
            assert len(rec["action"]) > 0

    def test_generate_feature_recommendation(self, generator):
        """Test feature-specific recommendation generation."""
        # Test known feature
        rec = generator._generate_feature_recommendation("V14", 0.25)
        assert "V14" in rec["insight"] or "V14" in rec["action"]

        # Test unknown feature (generic recommendation)
        rec = generator._generate_feature_recommendation("unknown_feature", 0.15)
        assert "unknown_feature" in rec["insight"]

    def test_create_business_report(self, generator, tmp_path):
        """Test business report creation."""
        save_path = tmp_path / "report.md"

        report = generator.create_business_report(save_path=str(save_path))

        assert isinstance(report, str)
        assert len(report) > 0
        assert "# Fraud Detection Model" in report
        assert "Executive Summary" in report
        assert "Top Fraud Drivers" in report
        assert "Strategic Recommendations" in report
        assert save_path.exists()

        # Check file content
        with open(save_path, "r") as f:
            file_content = f.read()
            assert file_content == report

    def test_plot_fraud_driver_summary(self, generator, tmp_path):
        """Test fraud driver summary plotting."""
        save_path = tmp_path / "drivers.png"

        fig = generator.plot_fraud_driver_summary(save_path=str(save_path))

        assert isinstance(fig, plt.Figure)
        assert save_path.exists()
        plt.close(fig)

    def test_plot_fraud_driver_summary_without_drivers(self, generator, tmp_path):
        """Test plotting initializes drivers if not already done."""
        # Ensure fraud_drivers is None
        generator.fraud_drivers = None

        save_path = tmp_path / "drivers.png"
        fig = generator.plot_fraud_driver_summary(save_path=str(save_path))

        assert generator.fraud_drivers is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_explainability_workflow(self, trained_model, tmp_path):
        """Test complete explainability analysis workflow."""
        model, X_test, y_test = trained_model

        # Create analyzer
        analyzer = ExplainabilityAnalyzer(model, X_test, y_test)

        # Extract built-in importance
        importance = analyzer.extract_feature_importance()
        assert len(importance) > 0

        # Initialize SHAP
        analyzer.initialize_shap_explainer()

        # Calculate SHAP values
        shap_values = analyzer.calculate_shap_values(max_samples=10)
        assert shap_values is not None

        # Analyze prediction
        analysis = analyzer.analyze_prediction(0)
        assert analysis["sample_index"] == 0

        # Compare importance methods
        comparison = analyzer.compare_importance_rankings(top_n=5)
        assert len(comparison) == 5

        # Generate recommendations
        generator = RecommendationGenerator(analyzer)
        recommendations = generator.generate_recommendations()
        assert len(recommendations) > 0

        # Create business report
        report_path = tmp_path / "business_report.md"
        report = generator.create_business_report(save_path=str(report_path))
        assert report_path.exists()

    def test_lightgbm_compatibility(self):
        """Test compatibility with LightGBM models."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.Series(np.random.randint(0, 2, 100))

        # Split data
        split_idx = 70
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Train LightGBM
        model = LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
        model.fit(X_train, y_train)

        # Create analyzer
        analyzer = ExplainabilityAnalyzer(model, X_test, y_test)

        # Test methods
        importance = analyzer.extract_feature_importance()
        assert len(importance) > 0

        analyzer.initialize_shap_explainer()
        shap_values = analyzer.calculate_shap_values(max_samples=10)
        assert shap_values is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
