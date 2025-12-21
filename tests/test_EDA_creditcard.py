"""Unit tests for EDA_creditcard module.

This module contains tests for the CreditCardEDA class to ensure
correct exploratory data analysis functionality for the credit card dataset.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path

from src.EDA_creditcard import CreditCardEDA


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_cc_data():
    """Create a small synthetic credit card dataset for testing."""
    np.random.seed(0)
    n = 120
    data = pd.DataFrame(
        {
            "Time": np.random.randint(0, 60_000, n),
            **{f"V{i}": np.random.randn(n) for i in range(1, 5)},  # small subset V1-V4
            "Amount": np.abs(np.random.randn(n) * 100),
            "Class": np.random.choice([0, 1], size=n, p=[0.98, 0.02]),
        }
    )
    return data


class TestCreditCardEDA:
    def test_initialization(self, sample_cc_data, temp_output_dir):
        eda = CreditCardEDA(
            sample_cc_data, target_column="Class", output_dir=temp_output_dir
        )
        assert eda.data.shape == sample_cc_data.shape
        assert eda.target_column == "Class"
        assert eda.output_dir == Path(temp_output_dir)
        assert eda.output_dir.exists()

    def test_pca_features_analysis(self, sample_cc_data, temp_output_dir):
        eda = CreditCardEDA(sample_cc_data, output_dir=temp_output_dir)
        results = eda.pca_features_analysis(save_plots=True)
        assert "stats" in results
        assert len(results["stats"]) > 0
        assert len(results.get("plots_saved", [])) >= 1
        assert Path(results["plots_saved"][0]).exists()

    def test_amount_analysis(self, sample_cc_data, temp_output_dir):
        eda = CreditCardEDA(sample_cc_data, output_dir=temp_output_dir)
        results = eda.amount_analysis(save_plots=True)
        assert "stats" in results
        assert "mean" in results["stats"]
        assert len(results.get("plots_saved", [])) == 1

    def test_time_analysis(self, sample_cc_data, temp_output_dir):
        eda = CreditCardEDA(sample_cc_data, output_dir=temp_output_dir)
        results = eda.time_analysis(save_plots=True)
        assert "hourly" in results
        assert len(results.get("plots_saved", [])) == 1

    def test_correlation_analysis(self, sample_cc_data, temp_output_dir):
        eda = CreditCardEDA(sample_cc_data, output_dir=temp_output_dir)
        results = eda.correlation_analysis(save_plots=True)
        assert "corr_matrix" in results
        assert len(results.get("plots_saved", [])) == 1

    def test_analyze_class_imbalance(self, sample_cc_data, temp_output_dir):
        eda = CreditCardEDA(sample_cc_data, output_dir=temp_output_dir)
        results = eda.analyze_class_imbalance(save_plot=True)
        assert "fraud_count" in results
        assert "legitimate_count" in results
        assert "imbalance_ratio" in results
        assert Path(results["plot_saved"]).exists()

    def test_generate_eda_report(self, sample_cc_data, temp_output_dir):
        eda = CreditCardEDA(sample_cc_data, output_dir=temp_output_dir)
        eda.pca_features_analysis(save_plots=False)
        eda.amount_analysis(save_plots=False)
        eda.time_analysis(save_plots=False)
        eda.correlation_analysis(save_plots=False)
        eda.analyze_class_imbalance(save_plot=False)
        report = eda.generate_eda_report()
        assert "pca_features_analysis" in report
        assert "amount_analysis" in report
        assert "time_analysis" in report
        assert "correlation_analysis" in report
        assert "class_imbalance" in report
