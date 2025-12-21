"""Unit tests for EDA_fraud module.

This module contains tests for the FraudDataEDA class to ensure
correct exploratory data analysis functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from src.EDA_fraud import FraudDataEDA


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after tests
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_fraud_data():
    """Create sample fraud data for testing."""
    np.random.seed(42)
    n_samples = 100

    data = pd.DataFrame(
        {
            "user_id": range(1, n_samples + 1),
            "signup_time": pd.date_range("2015-01-01", periods=n_samples, freq="h"),
            "purchase_time": pd.date_range("2015-01-02", periods=n_samples, freq="h"),
            "purchase_value": np.random.uniform(10, 500, n_samples),
            "device_id": [f"dev_{i%20}" for i in range(n_samples)],
            "source": np.random.choice(["SEO", "Ads", "Direct"], n_samples),
            "browser": np.random.choice(["Chrome", "Safari", "Firefox"], n_samples),
            "sex": np.random.choice(["M", "F"], n_samples),
            "age": np.random.randint(18, 70, n_samples),
            "ip_address": np.random.randint(1000000, 9999999, n_samples),
            "class": np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),  # 10% fraud
        }
    )
    return data


class TestFraudDataEDA:
    """Test cases for FraudDataEDA class."""

    def test_initialization(self, sample_fraud_data, temp_output_dir):
        """Test FraudDataEDA initialization."""
        eda = FraudDataEDA(
            sample_fraud_data, target_column="class", output_dir=temp_output_dir
        )

        assert eda.data.shape == sample_fraud_data.shape
        assert eda.target_column == "class"
        assert eda.output_dir == Path(temp_output_dir)
        assert eda.output_dir.exists()
        assert isinstance(eda.report, dict)

    def test_initialization_creates_output_dir(self, sample_fraud_data):
        """Test that initialization creates output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = Path(temp_dir) / "new_output"
            assert not new_dir.exists()

            eda = FraudDataEDA(sample_fraud_data, output_dir=str(new_dir))

            assert new_dir.exists()

    def test_univariate_analysis_numerical(self, sample_fraud_data, temp_output_dir):
        """Test univariate analysis for numerical features."""
        eda = FraudDataEDA(sample_fraud_data, output_dir=temp_output_dir)

        results = eda.univariate_analysis(
            numerical_features=["purchase_value", "age"],
            categorical_features=[],
            save_plots=True,
        )

        # Check results structure
        assert "numerical_stats" in results
        assert "categorical_stats" in results
        assert "plots_saved" in results

        # Check numerical stats
        assert "purchase_value" in results["numerical_stats"]
        assert "age" in results["numerical_stats"]

        # Check stats content
        pv_stats = results["numerical_stats"]["purchase_value"]
        assert "mean" in pv_stats
        assert "median" in pv_stats
        assert "std" in pv_stats
        assert "min" in pv_stats
        assert "max" in pv_stats
        assert "skewness" in pv_stats
        assert "kurtosis" in pv_stats

        # Check plot was saved
        assert len(results["plots_saved"]) > 0
        plot_path = Path(results["plots_saved"][0])
        assert plot_path.exists()

    def test_univariate_analysis_categorical(self, sample_fraud_data, temp_output_dir):
        """Test univariate analysis for categorical features."""
        eda = FraudDataEDA(sample_fraud_data, output_dir=temp_output_dir)

        results = eda.univariate_analysis(
            numerical_features=[],
            categorical_features=["source", "browser"],
            save_plots=True,
        )

        # Check categorical stats
        assert "source" in results["categorical_stats"]
        assert "browser" in results["categorical_stats"]

        # Check stats content
        source_stats = results["categorical_stats"]["source"]
        assert "unique_count" in source_stats
        assert "top_value" in source_stats
        assert "top_value_count" in source_stats
        assert "top_value_percentage" in source_stats

        # Check plot was saved
        assert len(results["plots_saved"]) > 0

    def test_univariate_analysis_auto_detect(self, sample_fraud_data, temp_output_dir):
        """Test auto-detection of feature types."""
        eda = FraudDataEDA(sample_fraud_data, output_dir=temp_output_dir)

        results = eda.univariate_analysis(save_plots=False)

        # Should auto-detect numerical and categorical features
        assert len(results["numerical_stats"]) > 0
        assert len(results["categorical_stats"]) > 0

        # Target column should be excluded from numerical features
        assert "class" not in results["numerical_stats"]

    def test_bivariate_analysis(self, sample_fraud_data, temp_output_dir):
        """Test bivariate analysis."""
        eda = FraudDataEDA(sample_fraud_data, output_dir=temp_output_dir)

        results = eda.bivariate_analysis(
            features=["purchase_value", "age", "source", "browser"], save_plots=True
        )

        # Check results structure
        assert "fraud_rates" in results
        assert "correlations" in results
        assert "plots_saved" in results

        # Check correlations for numerical features
        assert "purchase_value" in results["correlations"]
        assert "age" in results["correlations"]
        assert isinstance(results["correlations"]["purchase_value"], float)

        # Check fraud rates for categorical features
        assert "source" in results["fraud_rates"]
        assert "browser" in results["fraud_rates"]

        # Check plots were saved
        assert len(results["plots_saved"]) > 0

    def test_bivariate_analysis_all_features(self, sample_fraud_data, temp_output_dir):
        """Test bivariate analysis with all features."""
        eda = FraudDataEDA(sample_fraud_data, output_dir=temp_output_dir)

        results = eda.bivariate_analysis(save_plots=False)

        # Should analyze all non-target features
        assert len(results["correlations"]) > 0 or len(results["fraud_rates"]) > 0

    def test_analyze_class_imbalance(self, sample_fraud_data, temp_output_dir):
        """Test class imbalance analysis."""
        eda = FraudDataEDA(sample_fraud_data, output_dir=temp_output_dir)

        results = eda.analyze_class_imbalance(save_plot=True)

        # Check results structure
        assert "fraud_count" in results
        assert "legitimate_count" in results
        assert "fraud_percentage" in results
        assert "legitimate_percentage" in results
        assert "imbalance_ratio" in results
        assert "total_transactions" in results

        # Check values make sense
        assert results["fraud_count"] >= 0
        assert results["legitimate_count"] >= 0
        assert results["total_transactions"] == len(sample_fraud_data)
        assert (
            results["fraud_count"] + results["legitimate_count"]
            == results["total_transactions"]
        )
        assert 0 <= results["fraud_percentage"] <= 100
        assert 0 <= results["legitimate_percentage"] <= 100
        assert results["fraud_percentage"] + results[
            "legitimate_percentage"
        ] == pytest.approx(100, rel=1e-6)

        # Check plot was saved
        assert "plot_saved" in results
        plot_path = Path(results["plot_saved"])
        assert plot_path.exists()

    def test_analyze_class_imbalance_no_fraud(self, temp_output_dir):
        """Test class imbalance analysis when there's no fraud."""
        data = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "class": [0, 0, 0, 0, 0]})

        eda = FraudDataEDA(data, output_dir=temp_output_dir)
        results = eda.analyze_class_imbalance(save_plot=False)

        assert results["fraud_count"] == 0
        assert results["legitimate_count"] == 5
        assert results["fraud_percentage"] == 0.0

    def test_temporal_analysis(self, sample_fraud_data, temp_output_dir):
        """Test temporal pattern analysis."""
        eda = FraudDataEDA(sample_fraud_data, output_dir=temp_output_dir)

        results = eda.temporal_analysis(time_column="purchase_time", save_plot=True)

        # Check results structure
        assert "hourly_fraud_rate" in results
        assert "daily_fraud_rate" in results
        assert "plots_saved" in results

        # Check hourly stats
        hourly_stats = results["hourly_fraud_rate"]
        assert "fraud_count" in hourly_stats
        assert "total" in hourly_stats
        assert "fraud_rate" in hourly_stats

        # Check plot was saved
        assert len(results["plots_saved"]) > 0
        plot_path = Path(results["plots_saved"][0])
        assert plot_path.exists()

    def test_temporal_analysis_missing_column(self, sample_fraud_data, temp_output_dir):
        """Test temporal analysis with missing time column."""
        eda = FraudDataEDA(sample_fraud_data, output_dir=temp_output_dir)

        results = eda.temporal_analysis(
            time_column="nonexistent_column", save_plot=False
        )

        # Should return empty results without crashing
        assert isinstance(results, dict)
        assert "plots_saved" in results

    def test_temporal_analysis_converts_datetime(self, temp_output_dir):
        """Test that temporal analysis converts string dates to datetime."""
        data = pd.DataFrame(
            {
                "purchase_time": [
                    "2015-01-01 10:00:00",
                    "2015-01-01 14:00:00",
                    "2015-01-01 18:00:00",
                ],
                "class": [0, 1, 0],
            }
        )

        eda = FraudDataEDA(data, output_dir=temp_output_dir)
        results = eda.temporal_analysis(save_plot=False)

        # Should successfully process and return results
        assert "hourly_fraud_rate" in results

    def test_categorical_analysis(self, sample_fraud_data, temp_output_dir):
        """Test categorical features analysis."""
        eda = FraudDataEDA(sample_fraud_data, output_dir=temp_output_dir)

        results = eda.categorical_analysis(
            features=["source", "browser", "sex"], save_plot=False
        )

        # Check results for each feature
        assert "source" in results
        assert "browser" in results
        assert "sex" in results

        # Check structure of results
        source_results = results["source"]
        assert "fraud_count" in source_results
        assert "total" in source_results
        assert "fraud_rate" in source_results

    def test_categorical_analysis_auto_detect(self, sample_fraud_data, temp_output_dir):
        """Test auto-detection of categorical features."""
        eda = FraudDataEDA(sample_fraud_data, output_dir=temp_output_dir)

        results = eda.categorical_analysis(save_plot=False)

        # Should auto-detect categorical features
        assert len(results) > 0

    def test_categorical_analysis_missing_feature(
        self, sample_fraud_data, temp_output_dir
    ):
        """Test categorical analysis with non-existent feature."""
        eda = FraudDataEDA(sample_fraud_data, output_dir=temp_output_dir)

        results = eda.categorical_analysis(
            features=["nonexistent_feature"], save_plot=False
        )

        # Should handle gracefully
        assert isinstance(results, dict)

    def test_generate_eda_report_empty(self, sample_fraud_data, temp_output_dir):
        """Test generating EDA report without running analyses."""
        eda = FraudDataEDA(sample_fraud_data, output_dir=temp_output_dir)

        report = eda.generate_eda_report()

        # Should return empty report
        assert isinstance(report, dict)

    def test_generate_eda_report_complete(self, sample_fraud_data, temp_output_dir):
        """Test generating comprehensive EDA report."""
        eda = FraudDataEDA(sample_fraud_data, output_dir=temp_output_dir)

        # Run all analyses
        eda.univariate_analysis(save_plots=False)
        eda.bivariate_analysis(save_plots=False)
        eda.analyze_class_imbalance(save_plot=False)
        eda.temporal_analysis(save_plot=False)
        eda.categorical_analysis(save_plot=False)

        report = eda.generate_eda_report()

        # Check report contains all analyses
        assert "univariate_analysis" in report
        assert "bivariate_analysis" in report
        assert "class_imbalance" in report
        assert "temporal_analysis" in report
        assert "categorical_analysis" in report

    def test_report_persistence(self, sample_fraud_data, temp_output_dir):
        """Test that report is stored in the object."""
        eda = FraudDataEDA(sample_fraud_data, output_dir=temp_output_dir)

        eda.analyze_class_imbalance(save_plot=False)

        # Check report is stored
        assert "class_imbalance" in eda.report
        assert eda.report["class_imbalance"]["fraud_count"] >= 0


class TestFraudDataEDAIntegration:
    """Integration tests for complete EDA workflow."""

    def test_complete_eda_workflow(self, sample_fraud_data, temp_output_dir):
        """Test complete EDA workflow."""
        eda = FraudDataEDA(sample_fraud_data, output_dir=temp_output_dir)

        # Run all analyses
        univariate_results = eda.univariate_analysis(save_plots=True)
        bivariate_results = eda.bivariate_analysis(save_plots=True)
        imbalance_results = eda.analyze_class_imbalance(save_plot=True)
        temporal_results = eda.temporal_analysis(save_plot=True)
        categorical_results = eda.categorical_analysis(save_plot=True)

        # Generate final report
        report = eda.generate_eda_report()

        # Verify all results are present
        assert len(univariate_results) > 0
        assert len(bivariate_results) > 0
        assert len(imbalance_results) > 0
        assert len(temporal_results) > 0
        assert len(categorical_results) > 0
        assert len(report) == 5

        # Verify plots were created
        output_path = Path(temp_output_dir)
        plot_files = list(output_path.glob("*.png"))
        assert len(plot_files) > 0

    def test_eda_with_different_target_column(self, temp_output_dir):
        """Test EDA with a different target column name."""
        data = pd.DataFrame(
            {
                "feature1": np.random.randn(50),
                "feature2": np.random.choice(["A", "B"], 50),
                "fraud_label": np.random.choice([0, 1], 50),
            }
        )

        eda = FraudDataEDA(
            data, target_column="fraud_label", output_dir=temp_output_dir
        )

        results = eda.analyze_class_imbalance(save_plot=False)

        assert "fraud_count" in results
        assert results["total_transactions"] == 50
