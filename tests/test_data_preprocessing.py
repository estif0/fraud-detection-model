"""Unit tests for data preprocessing module.

This module contains tests for the DataLoader class and other preprocessing
components to ensure correct data loading and processing functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from src.data_preprocessing import DataLoader


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after tests
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_processed_dir():
    """Create a temporary directory for processed data files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after tests
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_fraud_data():
    """Create sample fraud data for testing."""
    data = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 5],
            "signup_time": ["2015-01-01 00:00:00"] * 5,
            "purchase_time": ["2015-01-02 00:00:00"] * 5,
            "purchase_value": [100, 200, 150, 300, 50],
            "device_id": ["dev1", "dev2", "dev3", "dev4", "dev5"],
            "source": ["SEO", "Ads", "SEO", "Direct", "Ads"],
            "browser": ["Chrome", "Safari", "Chrome", "Firefox", "Chrome"],
            "sex": ["M", "F", "M", "F", "M"],
            "age": [25, 30, 35, 40, 28],
            "ip_address": [123456789, 987654321, 111222333, 444555666, 777888999],
            "class": [0, 1, 0, 0, 1],
        }
    )
    return data


@pytest.fixture
def sample_ip_mapping():
    """Create sample IP mapping data for testing."""
    data = pd.DataFrame(
        {
            "lower_bound_ip_address": [100000000, 200000000, 300000000],
            "upper_bound_ip_address": [199999999, 299999999, 399999999],
            "country": ["USA", "UK", "Germany"],
        }
    )
    return data


@pytest.fixture
def sample_creditcard_data():
    """Create sample credit card data for testing."""
    data = pd.DataFrame(
        {
            "Time": [0, 100, 200, 300, 400],
            "V1": np.random.randn(5),
            "V2": np.random.randn(5),
            "V3": np.random.randn(5),
            "Amount": [100.0, 200.5, 50.0, 300.0, 25.5],
            "Class": [0, 1, 0, 0, 1],
        }
    )
    return data


class TestDataLoader:
    """Test cases for DataLoader class."""

    def test_initialization(self, temp_data_dir, temp_processed_dir):
        """Test DataLoader initialization."""
        loader = DataLoader(data_dir=temp_data_dir, processed_dir=temp_processed_dir)

        assert loader.data_dir == Path(temp_data_dir)
        assert loader.processed_dir == Path(temp_processed_dir)
        assert loader.processed_dir.exists()

    def test_load_fraud_data_success(
        self, temp_data_dir, temp_processed_dir, sample_fraud_data
    ):
        """Test successful loading of fraud data."""
        # Save sample data to temp directory
        fraud_file = Path(temp_data_dir) / "Fraud_Data.csv"
        sample_fraud_data.to_csv(fraud_file, index=False)

        # Load data
        loader = DataLoader(data_dir=temp_data_dir, processed_dir=temp_processed_dir)
        loaded_data = loader.load_fraud_data()

        # Assertions
        assert isinstance(loaded_data, pd.DataFrame)
        assert loaded_data.shape == sample_fraud_data.shape
        assert list(loaded_data.columns) == list(sample_fraud_data.columns)
        assert loaded_data["class"].sum() == 2  # Two fraud cases in sample data

    def test_load_fraud_data_file_not_found(self, temp_data_dir, temp_processed_dir):
        """Test loading fraud data when file doesn't exist."""
        loader = DataLoader(data_dir=temp_data_dir, processed_dir=temp_processed_dir)

        with pytest.raises(FileNotFoundError):
            loader.load_fraud_data()

    def test_load_ip_mapping_success(
        self, temp_data_dir, temp_processed_dir, sample_ip_mapping
    ):
        """Test successful loading of IP mapping data."""
        # Save sample data to temp directory
        ip_file = Path(temp_data_dir) / "IpAddress_to_Country.csv"
        sample_ip_mapping.to_csv(ip_file, index=False)

        # Load data
        loader = DataLoader(data_dir=temp_data_dir, processed_dir=temp_processed_dir)
        loaded_data = loader.load_ip_mapping()

        # Assertions
        assert isinstance(loaded_data, pd.DataFrame)
        assert loaded_data.shape == sample_ip_mapping.shape
        assert "country" in loaded_data.columns
        assert loaded_data["country"].nunique() == 3

    def test_load_ip_mapping_file_not_found(self, temp_data_dir, temp_processed_dir):
        """Test loading IP mapping when file doesn't exist."""
        loader = DataLoader(data_dir=temp_data_dir, processed_dir=temp_processed_dir)

        with pytest.raises(FileNotFoundError):
            loader.load_ip_mapping()

    def test_load_creditcard_data_success(
        self, temp_data_dir, temp_processed_dir, sample_creditcard_data
    ):
        """Test successful loading of credit card data."""
        # Save sample data to temp directory
        cc_file = Path(temp_data_dir) / "creditcard.csv"
        sample_creditcard_data.to_csv(cc_file, index=False)

        # Load data
        loader = DataLoader(data_dir=temp_data_dir, processed_dir=temp_processed_dir)
        loaded_data = loader.load_creditcard_data()

        # Assertions
        assert isinstance(loaded_data, pd.DataFrame)
        assert loaded_data.shape == sample_creditcard_data.shape
        assert "Class" in loaded_data.columns
        assert "Amount" in loaded_data.columns
        assert loaded_data["Class"].sum() == 2

    def test_load_creditcard_data_file_not_found(
        self, temp_data_dir, temp_processed_dir
    ):
        """Test loading credit card data when file doesn't exist."""
        loader = DataLoader(data_dir=temp_data_dir, processed_dir=temp_processed_dir)

        with pytest.raises(FileNotFoundError):
            loader.load_creditcard_data()

    def test_save_processed_data_success(
        self, temp_data_dir, temp_processed_dir, sample_fraud_data
    ):
        """Test successful saving of processed data."""
        loader = DataLoader(data_dir=temp_data_dir, processed_dir=temp_processed_dir)

        # Save data
        output_filename = "test_output.csv"
        loader.save_processed_data(sample_fraud_data, output_filename)

        # Verify file exists
        output_path = Path(temp_processed_dir) / output_filename
        assert output_path.exists()

        # Load and verify content
        saved_data = pd.read_csv(output_path)
        assert saved_data.shape == sample_fraud_data.shape

    def test_save_processed_data_invalid_filename(
        self, temp_data_dir, temp_processed_dir, sample_fraud_data
    ):
        """Test saving data with invalid filename (no .csv extension)."""
        loader = DataLoader(data_dir=temp_data_dir, processed_dir=temp_processed_dir)

        with pytest.raises(ValueError):
            loader.save_processed_data(sample_fraud_data, "invalid_name.txt")

    def test_save_processed_data_with_index(
        self, temp_data_dir, temp_processed_dir, sample_fraud_data
    ):
        """Test saving data with index included."""
        loader = DataLoader(data_dir=temp_data_dir, processed_dir=temp_processed_dir)

        output_filename = "test_with_index.csv"
        loader.save_processed_data(sample_fraud_data, output_filename, index=True)

        # Load and verify index was saved
        output_path = Path(temp_processed_dir) / output_filename
        saved_data = pd.read_csv(output_path)

        # When index is saved, it becomes the first column
        assert saved_data.shape[1] == sample_fraud_data.shape[1] + 1

    def test_get_data_info(self, temp_data_dir, temp_processed_dir, sample_fraud_data):
        """Test getting data information."""
        loader = DataLoader(data_dir=temp_data_dir, processed_dir=temp_processed_dir)

        info = loader.get_data_info(sample_fraud_data)

        # Assertions
        assert info["shape"] == sample_fraud_data.shape
        assert info["columns"] == sample_fraud_data.columns.tolist()
        assert isinstance(info["dtypes"], dict)
        assert isinstance(info["missing"], dict)
        assert isinstance(info["memory_usage"], float)
        assert info["memory_usage"] > 0

    def test_get_data_info_with_missing_values(self, temp_data_dir, temp_processed_dir):
        """Test getting data info when data has missing values."""
        data_with_missing = pd.DataFrame(
            {
                "col1": [1, 2, None, 4],
                "col2": ["a", None, "c", "d"],
                "col3": [1.0, 2.0, 3.0, None],
            }
        )

        loader = DataLoader(data_dir=temp_data_dir, processed_dir=temp_processed_dir)
        info = loader.get_data_info(data_with_missing)

        # Check missing values are correctly counted
        assert info["missing"]["col1"] == 1
        assert info["missing"]["col2"] == 1
        assert info["missing"]["col3"] == 1


class TestDataLoaderIntegration:
    """Integration tests for DataLoader with multiple operations."""

    def test_load_and_save_workflow(
        self, temp_data_dir, temp_processed_dir, sample_fraud_data
    ):
        """Test complete load and save workflow."""
        # Setup
        fraud_file = Path(temp_data_dir) / "Fraud_Data.csv"
        sample_fraud_data.to_csv(fraud_file, index=False)

        loader = DataLoader(data_dir=temp_data_dir, processed_dir=temp_processed_dir)

        # Load data
        loaded_data = loader.load_fraud_data()

        # Get info
        info = loader.get_data_info(loaded_data)
        assert info["shape"] == sample_fraud_data.shape

        # Save processed data
        loader.save_processed_data(loaded_data, "processed_fraud.csv")

        # Verify saved file
        output_path = Path(temp_processed_dir) / "processed_fraud.csv"
        assert output_path.exists()
