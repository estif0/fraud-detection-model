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
