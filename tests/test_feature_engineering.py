"""Unit tests for feature_engineering module.

Tests the FeatureEngineer class covering temporal features, durations,
transaction velocity/frequency, aggregation, encoding, and scaling.
"""

import pytest
import pandas as pd
import numpy as np

from src.feature_engineering import FeatureEngineer


@pytest.fixture
def sample_fraud_df():
    np.random.seed(1)
    n = 50
    df = pd.DataFrame(
        {
            "user_id": np.random.choice(range(1, 6), size=n),
            "signup_time": pd.date_range("2015-01-01", periods=n, freq="h"),
            "purchase_time": pd.date_range("2015-01-02", periods=n, freq="h"),
            "purchase_value": np.random.uniform(10, 300, size=n),
            "browser": np.random.choice(["Chrome", "Safari", "Firefox"], size=n),
            "source": np.random.choice(["SEO", "Ads", "Direct"], size=n),
            "class": np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
        }
    )
    return df


class TestFeatureEngineer:
    def test_create_time_features(self, sample_fraud_df):
        fe = FeatureEngineer()
        out = fe.create_time_features(sample_fraud_df, time_col="purchase_time")
        assert "hour_of_day" in out.columns
        assert "day_of_week" in out.columns
        assert "day_name" in out.columns

    def test_time_since_signup(self, sample_fraud_df):
        fe = FeatureEngineer()
        out = fe.calculate_time_since_signup(sample_fraud_df)
        assert "time_since_signup_min" in out.columns
        # Check positive durations
        assert (out["time_since_signup_min"] >= 0).all()

    def test_transaction_frequency(self, sample_fraud_df):
        fe = FeatureEngineer()
        out = fe.calculate_transaction_frequency(sample_fraud_df)
        assert "transactions_per_user" in out.columns
        # Per-user counts should be consistent
        counts = out.groupby("user_id")["transactions_per_user"].nunique()
        assert (counts == 1).all()

    def test_transaction_velocity(self, sample_fraud_df):
        fe = FeatureEngineer()
        out = fe.calculate_transaction_velocity(sample_fraud_df, window_hours=24)
        col = "txn_velocity_24h"
        assert col in out.columns
        # Counts must be >=1 (the current transaction itself)
        assert (out[col] >= 1).all()

    def test_aggregated_features(self, sample_fraud_df):
        fe = FeatureEngineer()
        out = fe.create_aggregated_features(sample_fraud_df, value_col="purchase_value")
        assert "purchase_value_mean" in out.columns
        assert "purchase_value_std" in out.columns

    def test_encode_categorical_features(self, sample_fraud_df):
        fe = FeatureEngineer(target_column="class")
        out, enc = fe.encode_categorical_features(
            sample_fraud_df, ["browser", "source"]
        )
        # Encoder should create new columns
        created = [
            c
            for c in out.columns
            if c.startswith("browser_") or c.startswith("source_")
        ]
        assert len(created) > 0

    def test_scale_numerical_features(self, sample_fraud_df):
        fe = FeatureEngineer(target_column="class")
        out, scaler = fe.scale_numerical_features(sample_fraud_df, method="standard")
        # Check columns are scaled (mean ~ 0 for numerical columns)
        num_cols = sample_fraud_df.select_dtypes(include=[np.number]).columns.tolist()
        if "class" in num_cols:
            num_cols.remove("class")
        means = out[num_cols].mean().abs()
        # Allow small numerical tolerance
        assert (means < 1e-1).all()


class TestCreditCardFeatureEngineering:
    """Tests for credit card specific feature engineering."""

    @pytest.fixture
    def sample_creditcard_df(self):
        """Create sample credit card data with V1-V28 features."""
        np.random.seed(42)
        n = 100

        # Create PCA features V1-V28
        pca_features = {f"V{i}": np.random.randn(n) for i in range(1, 29)}

        df = pd.DataFrame(
            {
                "Time": np.random.randint(0, 172800, n),  # 0-48 hours in seconds
                **pca_features,
                "Amount": np.abs(np.random.randn(n) * 100),
                "Class": np.random.choice([0, 1], size=n, p=[0.998, 0.002]),
            }
        )

        return df

    def test_engineer_creditcard_features_basic(self, sample_creditcard_df):
        """Test basic credit card feature engineering."""
        fe = FeatureEngineer()
        result = fe.engineer_creditcard_features(sample_creditcard_df)

        # Check shape: should have 30 features + Class
        assert result.shape[1] == 31
        assert result.shape[0] == sample_creditcard_df.shape[0]

        # Check all V features present
        for i in range(1, 29):
            assert f"V{i}" in result.columns, f"V{i} missing"

        # Check other features
        assert "Amount" in result.columns
        assert "hours" in result.columns
        assert "Class" in result.columns

        # Check Time was removed
        assert "Time" not in result.columns

    def test_engineer_creditcard_features_hours_conversion(self, sample_creditcard_df):
        """Test Time to hours conversion."""
        fe = FeatureEngineer()
        result = fe.engineer_creditcard_features(sample_creditcard_df)

        # Check hours is float
        assert result["hours"].dtype == float

        # Check hours are reasonable (0-48 for our test data)
        assert result["hours"].min() >= 0
        assert result["hours"].max() <= 48

        # Verify conversion: hours = Time / 3600
        expected_hours = sample_creditcard_df["Time"] / 3600
        assert np.allclose(result["hours"], expected_hours, rtol=1e-5)

    def test_engineer_creditcard_features_preserve_pca(self, sample_creditcard_df):
        """Test that PCA features are preserved."""
        fe = FeatureEngineer()
        result = fe.engineer_creditcard_features(
            sample_creditcard_df, preserve_pca=True
        )

        # Count V features
        v_features = [c for c in result.columns if c.startswith("V")]
        assert len(v_features) == 28

        # Verify values are unchanged (not scaled)
        for v_feat in v_features:
            assert np.allclose(result[v_feat], sample_creditcard_df[v_feat], rtol=1e-5)

    def test_engineer_creditcard_features_without_scaling(self, sample_creditcard_df):
        """Test feature engineering without scaling."""
        fe = FeatureEngineer()
        result = fe.engineer_creditcard_features(
            sample_creditcard_df, scale_features=False
        )

        # Values should not be standardized
        # Check that means are not close to 0 (would indicate scaling)
        non_zero_means = result[["Amount", "hours"]].mean().abs()
        assert (non_zero_means > 0.1).any()

    def test_engineer_creditcard_features_missing_time(self):
        """Test error handling when Time column is missing."""
        fe = FeatureEngineer()
        df_no_time = pd.DataFrame(
            {"V1": [1, 2, 3], "Amount": [10, 20, 30], "Class": [0, 1, 0]}
        )

        with pytest.raises(ValueError, match="'Time' column is required"):
            fe.engineer_creditcard_features(df_no_time)

    def test_engineer_creditcard_features_missing_pca(self):
        """Test error handling when PCA features are missing."""
        fe = FeatureEngineer()
        df_no_pca = pd.DataFrame(
            {"Time": [100, 200, 300], "Amount": [10, 20, 30], "Class": [0, 1, 0]}
        )

        with pytest.raises(ValueError, match="Missing PCA features"):
            fe.engineer_creditcard_features(df_no_pca, preserve_pca=True)

    def test_engineer_creditcard_features_without_class(self, sample_creditcard_df):
        """Test with dataset that doesn't have Class column."""
        fe = FeatureEngineer()
        df_no_class = sample_creditcard_df.drop("Class", axis=1)

        result = fe.engineer_creditcard_features(df_no_class)

        # Should have 30 features (no Class)
        assert result.shape[1] == 30
        assert "Class" not in result.columns

        # Should still have all V features, Amount, hours
        v_count = sum(1 for c in result.columns if c.startswith("V"))
        assert v_count == 28
        assert "Amount" in result.columns
        assert "hours" in result.columns

    def test_engineer_creditcard_features_feature_count(self, sample_creditcard_df):
        """Test that exact feature count is maintained."""
        fe = FeatureEngineer()
        result = fe.engineer_creditcard_features(sample_creditcard_df)

        # Count features (excluding Class)
        features = [c for c in result.columns if c != "Class"]
        assert len(features) == 30, f"Expected 30 features, got {len(features)}"

        # Break down: 28 V features + Amount + hours
        v_features = [c for c in features if c.startswith("V")]
        assert len(v_features) == 28
        assert "Amount" in features
        assert "hours" in features
