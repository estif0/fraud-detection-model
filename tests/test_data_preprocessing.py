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
from src.data_preprocessing import DataLoader, DataCleaner


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


class TestDataCleaner:
    """Test cases for DataCleaner class."""

    def test_initialization(self, sample_fraud_data):
        """Test DataCleaner initialization."""
        cleaner = DataCleaner(sample_fraud_data)

        assert cleaner.original_shape == sample_fraud_data.shape
        assert len(cleaner.cleaning_log) > 0
        # Ensure original data is not modified
        assert cleaner.data.shape == sample_fraud_data.shape

    def test_check_missing_values_none(self, sample_fraud_data):
        """Test checking missing values when there are none."""
        cleaner = DataCleaner(sample_fraud_data)
        missing_df = cleaner.check_missing_values()

        # Should return empty DataFrame when no missing values
        assert len(missing_df) == 0

    def test_check_missing_values_present(self):
        """Test checking missing values when they exist."""
        data = pd.DataFrame(
            {
                "col1": [1, 2, None, 4, 5],
                "col2": ["a", None, "c", None, "e"],
                "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )

        cleaner = DataCleaner(data)
        missing_df = cleaner.check_missing_values()

        assert len(missing_df) == 2  # col1 and col2 have missing values
        assert "col1" in missing_df["Column"].values
        assert "col2" in missing_df["Column"].values
        assert (
            missing_df[missing_df["Column"] == "col1"]["Missing_Count"].values[0] == 1
        )
        assert (
            missing_df[missing_df["Column"] == "col2"]["Missing_Count"].values[0] == 2
        )

    def test_handle_missing_values_drop(self):
        """Test dropping rows with missing values."""
        data = pd.DataFrame(
            {"col1": [1, 2, None, 4, 5], "col2": ["a", "b", "c", "d", "e"]}
        )

        cleaner = DataCleaner(data)
        cleaned = cleaner.handle_missing_values(strategy="drop")

        assert len(cleaned) == 4  # One row dropped
        assert cleaned["col1"].isnull().sum() == 0

    def test_handle_missing_values_fill(self):
        """Test filling missing values with specified values."""
        data = pd.DataFrame(
            {"col1": [1, 2, None, 4, 5], "col2": ["a", None, "c", "d", "e"]}
        )

        cleaner = DataCleaner(data)
        cleaned = cleaner.handle_missing_values(
            strategy="fill", fill_value={"col1": 0, "col2": "Unknown"}
        )

        assert cleaned["col1"].isnull().sum() == 0
        assert cleaned["col2"].isnull().sum() == 0
        assert cleaned.loc[2, "col1"] == 0
        assert cleaned.loc[1, "col2"] == "Unknown"

    def test_handle_missing_values_mean(self):
        """Test filling missing values with mean."""
        data = pd.DataFrame(
            {"col1": [1.0, 2.0, None, 4.0, 5.0], "col2": [10, 20, None, 40, 50]}
        )

        cleaner = DataCleaner(data)
        cleaned = cleaner.handle_missing_values(strategy="mean")

        assert cleaned["col1"].isnull().sum() == 0
        assert cleaned["col2"].isnull().sum() == 0
        # Mean of [1, 2, 4, 5] = 3.0
        assert cleaned.loc[2, "col1"] == 3.0
        # Mean of [10, 20, 40, 50] = 30.0
        assert cleaned.loc[2, "col2"] == 30.0

    def test_handle_missing_values_median(self):
        """Test filling missing values with median."""
        data = pd.DataFrame(
            {"col1": [1.0, 2.0, None, 4.0, 10.0], "col2": [10, 20, None, 40, 100]}
        )

        cleaner = DataCleaner(data)
        cleaned = cleaner.handle_missing_values(strategy="median")

        assert cleaned["col1"].isnull().sum() == 0
        assert cleaned["col2"].isnull().sum() == 0
        # Median of [1, 2, 4, 10] = 3.0
        assert cleaned.loc[2, "col1"] == 3.0
        # Median of [10, 20, 40, 100] = 30.0
        assert cleaned.loc[2, "col2"] == 30.0

    def test_handle_missing_values_mode(self):
        """Test filling missing values with mode."""
        data = pd.DataFrame(
            {"col1": ["a", "a", None, "b", "a"], "col2": ["x", None, "y", "x", "x"]}
        )

        cleaner = DataCleaner(data)
        cleaned = cleaner.handle_missing_values(strategy="mode")

        assert cleaned["col1"].isnull().sum() == 0
        assert cleaned["col2"].isnull().sum() == 0
        assert cleaned.loc[2, "col1"] == "a"  # Mode is 'a'
        assert cleaned.loc[1, "col2"] == "x"  # Mode is 'x'

    def test_handle_missing_values_drop_columns(self):
        """Test dropping columns with high missing percentage."""
        data = pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [None, None, None, None, 5],  # 80% missing
                "col3": [1, None, 3, 4, 5],  # 20% missing
            }
        )

        cleaner = DataCleaner(data)
        cleaned = cleaner.handle_missing_values(strategy="drop_columns", threshold=0.5)

        assert "col1" in cleaned.columns
        assert "col2" not in cleaned.columns  # Dropped (80% > 50%)
        assert "col3" in cleaned.columns

    def test_handle_missing_values_invalid_strategy(self, sample_fraud_data):
        """Test handling missing values with invalid strategy."""
        cleaner = DataCleaner(sample_fraud_data)

        with pytest.raises(ValueError):
            cleaner.handle_missing_values(strategy="invalid_strategy")

    def test_handle_missing_values_fill_without_values(self):
        """Test fill strategy without providing fill_value."""
        data = pd.DataFrame({"col1": [1, None, 3]})
        cleaner = DataCleaner(data)

        with pytest.raises(ValueError):
            cleaner.handle_missing_values(strategy="fill")

    def test_remove_duplicates_all_columns(self):
        """Test removing duplicate rows based on all columns."""
        data = pd.DataFrame({"col1": [1, 2, 1, 3], "col2": ["a", "b", "a", "c"]})

        cleaner = DataCleaner(data)
        cleaned = cleaner.remove_duplicates()

        assert len(cleaned) == 3  # One duplicate removed

    def test_remove_duplicates_subset(self):
        """Test removing duplicates based on subset of columns."""
        data = pd.DataFrame(
            {"user_id": [1, 2, 1, 3], "transaction": ["tx1", "tx2", "tx3", "tx4"]}
        )

        cleaner = DataCleaner(data)
        cleaned = cleaner.remove_duplicates(subset=["user_id"])

        assert len(cleaned) == 3  # One duplicate user_id removed

    def test_remove_duplicates_keep_last(self):
        """Test removing duplicates keeping last occurrence."""
        data = pd.DataFrame({"col1": [1, 2, 1, 3], "col2": ["a", "b", "c", "d"]})

        cleaner = DataCleaner(data)
        cleaned = cleaner.remove_duplicates(subset=["col1"], keep="last")

        assert len(cleaned) == 3
        # Should keep the row with col2='c' (last occurrence of col1=1)
        assert "c" in cleaned["col2"].values
        assert cleaned[cleaned["col1"] == 1]["col2"].values[0] == "c"

    def test_remove_duplicates_none_found(self, sample_fraud_data):
        """Test removing duplicates when none exist."""
        cleaner = DataCleaner(sample_fraud_data)
        cleaned = cleaner.remove_duplicates()

        assert len(cleaned) == len(sample_fraud_data)

    def test_validate_data_types_no_expected(self, sample_fraud_data):
        """Test validating data types without expected types."""
        cleaner = DataCleaner(sample_fraud_data)
        result = cleaner.validate_data_types()

        assert "current_types" in result
        assert len(result["current_types"]) > 0

    def test_validate_data_types_with_conversion(self):
        """Test validating and converting data types."""
        data = pd.DataFrame({"col1": ["1", "2", "3"], "col2": ["1.5", "2.5", "3.5"]})

        cleaner = DataCleaner(data)
        result = cleaner.validate_data_types(
            expected_types={"col1": "int", "col2": "float"}, convert=True
        )

        assert "converted" in result
        assert "col1" in result["converted"]
        assert "col2" in result["converted"]
        assert (
            cleaner.data["col1"].dtype == np.int64
            or cleaner.data["col1"].dtype == np.int32
        )
        assert cleaner.data["col2"].dtype == np.float64

    def test_validate_data_types_without_conversion(self):
        """Test validating data types without conversion."""
        data = pd.DataFrame({"col1": ["1", "2", "3"]})

        cleaner = DataCleaner(data)
        result = cleaner.validate_data_types(
            expected_types={"col1": "int"}, convert=False
        )

        assert "mismatches" in result
        assert "col1" in result["mismatches"]
        # Type should not have changed
        assert cleaner.data["col1"].dtype == object

    def test_validate_data_types_column_not_found(self, sample_fraud_data):
        """Test validating data types with non-existent column."""
        cleaner = DataCleaner(sample_fraud_data)
        result = cleaner.validate_data_types(expected_types={"nonexistent_col": "int"})

        # Should not raise error, just skip the column
        assert "current_types" in result

    def test_generate_cleaning_report(self):
        """Test generating cleaning report."""
        data = pd.DataFrame(
            {"col1": [1, 2, None, 4, 5], "col2": ["a", "b", "c", "d", "e"]}
        )

        cleaner = DataCleaner(data)
        cleaner.handle_missing_values(strategy="drop")
        report = cleaner.generate_cleaning_report()

        assert "original_shape" in report
        assert "current_shape" in report
        assert "rows_removed" in report
        assert "columns_removed" in report
        assert "cleaning_operations" in report
        assert "missing_values_remaining" in report
        assert "duplicates_remaining" in report

        assert report["original_shape"] == (5, 2)
        assert report["current_shape"] == (4, 2)
        assert report["rows_removed"] == 1
        assert report["columns_removed"] == 0
        assert report["missing_values_remaining"] == 0

    def test_get_cleaned_data(self, sample_fraud_data):
        """Test getting cleaned data."""
        cleaner = DataCleaner(sample_fraud_data)
        cleaned = cleaner.get_cleaned_data()

        assert isinstance(cleaned, pd.DataFrame)
        assert cleaned.shape == sample_fraud_data.shape
        # Ensure it's a copy
        assert cleaned is not cleaner.data


class TestDataCleanerIntegration:
    """Integration tests for DataCleaner with multiple operations."""

    def test_complete_cleaning_workflow(self):
        """Test complete cleaning workflow with multiple operations."""
        # Create messy data
        data = pd.DataFrame(
            {
                "user_id": [1, 2, 1, 3, 4, None],
                "age": ["25", "30", "25", "35", "40", "28"],
                "amount": [100.0, 200.0, 100.0, None, 300.0, 150.0],
                "category": ["A", "B", "A", "C", "B", "A"],
            }
        )

        cleaner = DataCleaner(data)

        # Check missing values
        missing = cleaner.check_missing_values()
        assert len(missing) == 2  # user_id and amount have missing values

        # Handle missing values
        cleaner.handle_missing_values(
            strategy="fill", fill_value={"user_id": 0, "amount": 0.0}
        )

        # Remove duplicates
        cleaner.remove_duplicates(subset=["user_id", "age", "category"])

        # Validate data types
        cleaner.validate_data_types(
            expected_types={"user_id": "int", "age": "int", "amount": "float"},
            convert=True,
        )

        # Generate report
        report = cleaner.generate_cleaning_report()

        # Get cleaned data
        cleaned = cleaner.get_cleaned_data()

        # Assertions
        assert cleaned["user_id"].isnull().sum() == 0
        assert cleaned["amount"].isnull().sum() == 0
        assert len(cleaned) < len(data)  # Duplicates removed
        assert cleaned["age"].dtype in [np.int64, np.int32]
        assert len(report["cleaning_operations"]) > 0
