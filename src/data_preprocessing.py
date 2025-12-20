"""Data preprocessing module for fraud detection project.

This module contains classes for loading, cleaning, and preprocessing
transaction data from multiple sources including e-commerce fraud data
and credit card transaction data.

Classes:
    DataLoader: Handles loading data from various CSV sources.
    DataCleaner: Manages data cleaning and validation operations.
    IPMapper: Maps IP addresses to countries using range-based lookup.
    ImbalanceHandler: Handles class imbalance using various sampling techniques.
"""

import os
from typing import Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path


class DataLoader:
    """Load and manage transaction datasets for fraud detection.

    This class provides methods to load fraud data, IP mapping data,
    and credit card transaction data from CSV files. It also handles
    saving processed data to specified output directories.

    Attributes:
        data_dir (Path): Path to the directory containing raw data files.
        processed_dir (Path): Path to the directory for saving processed data.

    Example:
        >>> loader = DataLoader(data_dir='data/raw', processed_dir='data/processed')
        >>> fraud_data = loader.load_fraud_data()
        >>> print(fraud_data.shape)
    """

    def __init__(
        self, data_dir: str = "data/raw", processed_dir: str = "data/processed"
    ):
        """Initialize the DataLoader with specified directories.

        Args:
            data_dir (str): Path to directory containing raw data files.
                Defaults to 'data/raw'.
            processed_dir (str): Path to directory for saving processed data.
                Defaults to 'data/processed'.
        """
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)

        # Create processed directory if it doesn't exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_fraud_data(self, filename: str = "Fraud_Data.csv") -> pd.DataFrame:
        """Load e-commerce fraud transaction data.

        Loads the Fraud_Data.csv file containing e-commerce transaction
        information including user details, purchase information, and
        fraud labels.

        Args:
            filename (str): Name of the fraud data CSV file.
                Defaults to 'Fraud_Data.csv'.

        Returns:
            pd.DataFrame: DataFrame containing fraud transaction data with columns:
                - user_id: Unique identifier for users
                - signup_time: User signup timestamp
                - purchase_time: Purchase timestamp
                - purchase_value: Transaction amount
                - device_id: Device identifier
                - source: Traffic source
                - browser: Browser used
                - sex: User gender
                - age: User age
                - ip_address: IP address (as float)
                - class: Fraud label (1=fraud, 0=legitimate)

        Raises:
            FileNotFoundError: If the specified file does not exist.
            pd.errors.EmptyDataError: If the file is empty.

        Example:
            >>> loader = DataLoader()
            >>> fraud_data = loader.load_fraud_data()
            >>> print(fraud_data.columns.tolist())
        """
        file_path = self.data_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Fraud data file not found at: {file_path}")

        # Load data with appropriate data types
        fraud_data = pd.read_csv(file_path)

        print(
            f"✓ Loaded fraud data: {fraud_data.shape[0]} rows, {fraud_data.shape[1]} columns"
        )
        print(f"  Fraud cases: {fraud_data['class'].sum()}")
        print(f"  Legitimate cases: {(fraud_data['class'] == 0).sum()}")

        return fraud_data

    def load_ip_mapping(
        self, filename: str = "IpAddress_to_Country.csv"
    ) -> pd.DataFrame:
        """Load IP address to country mapping data.

        Loads the mapping file that associates IP address ranges with
        countries for geolocation analysis.

        Args:
            filename (str): Name of the IP mapping CSV file.
                Defaults to 'IpAddress_to_Country.csv'.

        Returns:
            pd.DataFrame: DataFrame containing IP mapping data with columns:
                - lower_bound_ip_address: Lower bound of IP range (as integer)
                - upper_bound_ip_address: Upper bound of IP range (as integer)
                - country: Country name

        Raises:
            FileNotFoundError: If the specified file does not exist.

        Example:
            >>> loader = DataLoader()
            >>> ip_mapping = loader.load_ip_mapping()
            >>> print(ip_mapping.head())
        """
        file_path = self.data_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"IP mapping file not found at: {file_path}")

        # Load IP mapping data
        ip_mapping = pd.read_csv(file_path)

        print(f"✓ Loaded IP mapping data: {ip_mapping.shape[0]} IP ranges")
        print(f"  Unique countries: {ip_mapping['country'].nunique()}")

        return ip_mapping

    def load_creditcard_data(self, filename: str = "creditcard.csv") -> pd.DataFrame:
        """Load credit card transaction data.

        Loads the credit card dataset containing anonymized PCA-transformed
        features and transaction information for fraud detection.

        Args:
            filename (str): Name of the credit card data CSV file.
                Defaults to 'creditcard.csv'.

        Returns:
            pd.DataFrame: DataFrame containing credit card transaction data with columns:
                - Time: Seconds elapsed since first transaction
                - V1-V28: PCA-transformed anonymized features
                - Amount: Transaction amount
                - Class: Fraud label (1=fraud, 0=legitimate)

        Raises:
            FileNotFoundError: If the specified file does not exist.

        Example:
            >>> loader = DataLoader()
            >>> cc_data = loader.load_creditcard_data()
            >>> print(f"Features: {cc_data.shape[1]}")
        """
        file_path = self.data_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Credit card data file not found at: {file_path}")

        # Load credit card data
        cc_data = pd.read_csv(file_path)

        print(
            f"✓ Loaded credit card data: {cc_data.shape[0]} rows, {cc_data.shape[1]} columns"
        )
        print(f"  Fraud cases: {cc_data['Class'].sum()}")
        print(f"  Legitimate cases: {(cc_data['Class'] == 0).sum()}")

        return cc_data

    def save_processed_data(
        self, data: pd.DataFrame, filename: str, index: bool = False
    ) -> None:
        """Save processed data to the processed directory.

        Saves a DataFrame to the specified filename in the processed
        data directory.

        Args:
            data (pd.DataFrame): DataFrame to save.
            filename (str): Name for the output CSV file (should end with .csv).
            index (bool): Whether to save the DataFrame index. Defaults to False.

        Raises:
            ValueError: If filename doesn't end with .csv.

        Example:
            >>> loader = DataLoader()
            >>> processed_data = pd.DataFrame({'col1': [1, 2, 3]})
            >>> loader.save_processed_data(processed_data, 'cleaned_data.csv')
        """
        if not filename.endswith(".csv"):
            raise ValueError("Filename must end with .csv")

        output_path = self.processed_dir / filename

        # Save to CSV
        data.to_csv(output_path, index=index)

        print(f"✓ Saved processed data to: {output_path}")
        print(f"  Shape: {data.shape[0]} rows, {data.shape[1]} columns")

    def get_data_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get summary information about a dataset.

        Args:
            data (pd.DataFrame): DataFrame to analyze.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - shape: Tuple of (rows, columns)
                - columns: List of column names
                - dtypes: Dictionary of column data types
                - missing: Dictionary of missing value counts per column
                - memory_usage: Total memory usage in MB

        Example:
            >>> loader = DataLoader()
            >>> fraud_data = loader.load_fraud_data()
            >>> info = loader.get_data_info(fraud_data)
            >>> print(info['shape'])
        """
        info = {
            "shape": data.shape,
            "columns": data.columns.tolist(),
            "dtypes": data.dtypes.to_dict(),
            "missing": data.isnull().sum().to_dict(),
            "memory_usage": data.memory_usage(deep=True).sum() / 1024**2,  # MB
        }

        return info


class DataCleaner:
    """Clean and validate transaction data for fraud detection.

    This class provides methods to identify and handle data quality issues
    including missing values, duplicates, invalid data types, and outliers.
    All cleaning decisions are logged and can be included in a cleaning report.

    Attributes:
        data (pd.DataFrame): The dataset to clean.
        cleaning_log (list): Log of all cleaning operations performed.
        original_shape (tuple): Original shape of the dataset before cleaning.

    Example:
        >>> cleaner = DataCleaner(fraud_data)
        >>> cleaner.check_missing_values()
        >>> cleaned_data = cleaner.handle_missing_values()
        >>> report = cleaner.generate_cleaning_report()
    """

    def __init__(self, data: pd.DataFrame):
        """Initialize the DataCleaner with a dataset.

        Args:
            data (pd.DataFrame): The dataset to clean.
        """
        self.data = data.copy()  # Work with a copy to preserve original
        self.cleaning_log = []
        self.original_shape = data.shape

        self.cleaning_log.append(
            f"Initialized DataCleaner with data shape: {self.original_shape}"
        )

    def check_missing_values(self) -> pd.DataFrame:
        """Identify missing values in the dataset.

        Analyzes each column for missing values and returns a summary
        DataFrame with counts and percentages.

        Returns:
            pd.DataFrame: Summary of missing values with columns:
                - Column: Column name
                - Missing_Count: Number of missing values
                - Missing_Percentage: Percentage of missing values
                - Data_Type: Column data type

        Example:
            >>> cleaner = DataCleaner(fraud_data)
            >>> missing_summary = cleaner.check_missing_values()
            >>> print(missing_summary)
        """
        missing_count = self.data.isnull().sum()
        missing_percentage = (missing_count / len(self.data)) * 100

        missing_df = pd.DataFrame(
            {
                "Column": missing_count.index,
                "Missing_Count": missing_count.values,
                "Missing_Percentage": missing_percentage.values,
                "Data_Type": self.data.dtypes.values,
            }
        )

        # Filter to show only columns with missing values
        missing_df = missing_df[missing_df["Missing_Count"] > 0].sort_values(
            "Missing_Count", ascending=False
        )

        if len(missing_df) > 0:
            self.cleaning_log.append(
                f"Found missing values in {len(missing_df)} columns"
            )
            print(f"⚠ Missing values found in {len(missing_df)} columns:")
            print(missing_df.to_string(index=False))
        else:
            self.cleaning_log.append("No missing values found")
            print("✓ No missing values found")

        return missing_df

    def handle_missing_values(
        self,
        strategy: str = "drop",
        fill_value: Optional[Dict[str, Any]] = None,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """Handle missing values using specified strategy.

        Implements different strategies for handling missing values:
        - 'drop': Drop rows with any missing values
        - 'drop_columns': Drop columns with missing percentage above threshold
        - 'fill': Fill missing values with specified values
        - 'forward_fill': Forward fill missing values
        - 'mean': Fill numerical columns with mean
        - 'median': Fill numerical columns with median
        - 'mode': Fill categorical columns with mode

        Args:
            strategy (str): Strategy to use for handling missing values.
                Defaults to 'drop'.
            fill_value (dict, optional): Dictionary mapping column names to fill values.
                Used when strategy='fill'.
            threshold (float): Threshold for dropping columns (0-1).
                Used when strategy='drop_columns'. Defaults to 0.5 (50%).

        Returns:
            pd.DataFrame: Cleaned dataset with missing values handled.

        Raises:
            ValueError: If strategy is invalid or fill_value is missing for 'fill' strategy.

        Example:
            >>> cleaner = DataCleaner(fraud_data)
            >>> # Drop rows with missing values
            >>> cleaned = cleaner.handle_missing_values(strategy='drop')
            >>> # Fill with specific values
            >>> cleaned = cleaner.handle_missing_values(
            ...     strategy='fill',
            ...     fill_value={'age': 0, 'source': 'Unknown'}
            ... )
        """
        rows_before = len(self.data)
        cols_before = len(self.data.columns)

        if strategy == "drop":
            # Drop rows with any missing values
            self.data = self.data.dropna()
            rows_dropped = rows_before - len(self.data)
            self.cleaning_log.append(f"Dropped {rows_dropped} rows with missing values")
            print(f"✓ Dropped {rows_dropped} rows with missing values")

        elif strategy == "drop_columns":
            # Drop columns with missing percentage above threshold
            missing_pct = self.data.isnull().sum() / len(self.data)
            cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()

            if cols_to_drop:
                self.data = self.data.drop(columns=cols_to_drop)
                self.cleaning_log.append(
                    f"Dropped {len(cols_to_drop)} columns with >{threshold*100}% missing: {cols_to_drop}"
                )
                print(f"✓ Dropped {len(cols_to_drop)} columns: {cols_to_drop}")
            else:
                print(f"✓ No columns exceed {threshold*100}% missing threshold")

        elif strategy == "fill":
            if fill_value is None:
                raise ValueError("fill_value must be provided when strategy='fill'")

            for col, value in fill_value.items():
                if col in self.data.columns:
                    missing_count = self.data[col].isnull().sum()
                    self.data[col] = self.data[col].fillna(value)
                    self.cleaning_log.append(
                        f"Filled {missing_count} missing values in '{col}' with {value}"
                    )
                    print(
                        f"✓ Filled {missing_count} missing values in '{col}' with {value}"
                    )

        elif strategy == "forward_fill":
            self.data = self.data.fillna(method="ffill")
            self.cleaning_log.append("Applied forward fill to all columns")
            print("✓ Applied forward fill")

        elif strategy == "mean":
            # Fill numerical columns with mean
            numerical_cols = self.data.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if self.data[col].isnull().sum() > 0:
                    mean_value = self.data[col].mean()
                    missing_count = self.data[col].isnull().sum()
                    self.data[col] = self.data[col].fillna(mean_value)
                    self.cleaning_log.append(
                        f"Filled {missing_count} missing values in '{col}' with mean: {mean_value:.2f}"
                    )
                    print(f"✓ Filled '{col}' with mean: {mean_value:.2f}")

        elif strategy == "median":
            # Fill numerical columns with median
            numerical_cols = self.data.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if self.data[col].isnull().sum() > 0:
                    median_value = self.data[col].median()
                    missing_count = self.data[col].isnull().sum()
                    self.data[col] = self.data[col].fillna(median_value)
                    self.cleaning_log.append(
                        f"Filled {missing_count} missing values in '{col}' with median: {median_value:.2f}"
                    )
                    print(f"✓ Filled '{col}' with median: {median_value:.2f}")

        elif strategy == "mode":
            # Fill categorical columns with mode
            categorical_cols = self.data.select_dtypes(include=["object"]).columns
            for col in categorical_cols:
                if self.data[col].isnull().sum() > 0:
                    mode_value = self.data[col].mode()[0]
                    missing_count = self.data[col].isnull().sum()
                    self.data[col] = self.data[col].fillna(mode_value)
                    self.cleaning_log.append(
                        f"Filled {missing_count} missing values in '{col}' with mode: {mode_value}"
                    )
                    print(f"✓ Filled '{col}' with mode: {mode_value}")

        else:
            raise ValueError(
                f"Invalid strategy: {strategy}. "
                "Choose from: 'drop', 'drop_columns', 'fill', 'forward_fill', 'mean', 'median', 'mode'"
            )

        return self.data

    def remove_duplicates(
        self, subset: Optional[list] = None, keep: str = "first"
    ) -> pd.DataFrame:
        """Remove duplicate rows from the dataset.

        Args:
            subset (list, optional): Column labels to consider for identifying duplicates.
                If None, uses all columns.
            keep (str): Which duplicate to keep. Options:
                - 'first': Keep first occurrence (default)
                - 'last': Keep last occurrence
                - False: Drop all duplicates

        Returns:
            pd.DataFrame: Dataset with duplicates removed.

        Example:
            >>> cleaner = DataCleaner(fraud_data)
            >>> cleaned = cleaner.remove_duplicates()
            >>> # Remove duplicates based on user_id only
            >>> cleaned = cleaner.remove_duplicates(subset=['user_id'])
        """
        rows_before = len(self.data)

        self.data = self.data.drop_duplicates(subset=subset, keep=keep)

        rows_dropped = rows_before - len(self.data)

        if rows_dropped > 0:
            self.cleaning_log.append(
                f"Removed {rows_dropped} duplicate rows "
                f"(subset={subset}, keep={keep})"
            )
            print(f"✓ Removed {rows_dropped} duplicate rows")
        else:
            self.cleaning_log.append("No duplicate rows found")
            print("✓ No duplicate rows found")

        return self.data

    def validate_data_types(
        self, expected_types: Optional[Dict[str, str]] = None, convert: bool = True
    ) -> Dict[str, Any]:
        """Validate and optionally convert column data types.

        Checks if columns have expected data types and can automatically
        convert them if specified.

        Args:
            expected_types (dict, optional): Dictionary mapping column names to
                expected data types (e.g., {'age': 'int', 'purchase_value': 'float'}).
                If None, returns current data types.
            convert (bool): Whether to automatically convert to expected types.
                Defaults to True.

        Returns:
            dict: Dictionary with validation results:
                - current_types: Current data types
                - mismatches: Columns with type mismatches
                - converted: Columns that were converted (if convert=True)

        Raises:
            ValueError: If conversion fails for a column.

        Example:
            >>> cleaner = DataCleaner(fraud_data)
            >>> expected = {'user_id': 'int', 'age': 'int', 'purchase_value': 'float'}
            >>> result = cleaner.validate_data_types(expected, convert=True)
        """
        current_types = self.data.dtypes.to_dict()

        if expected_types is None:
            print("Current data types:")
            for col, dtype in current_types.items():
                print(f"  {col}: {dtype}")
            return {"current_types": current_types}

        mismatches = {}
        converted = []

        for col, expected_type in expected_types.items():
            if col not in self.data.columns:
                print(f"⚠ Column '{col}' not found in dataset")
                continue

            current_type = str(self.data[col].dtype)

            # Normalize type names for comparison
            expected_normalized = expected_type.lower()
            current_normalized = current_type.lower()

            # Check for type mismatch
            is_mismatch = False
            if "int" in expected_normalized and "int" not in current_normalized:
                is_mismatch = True
            elif "float" in expected_normalized and "float" not in current_normalized:
                is_mismatch = True
            elif "object" in expected_normalized and "object" not in current_normalized:
                is_mismatch = True
            elif (
                "datetime" in expected_normalized
                and "datetime" not in current_normalized
            ):
                is_mismatch = True

            if is_mismatch:
                mismatches[col] = {"current": current_type, "expected": expected_type}

                if convert:
                    try:
                        # Attempt conversion
                        if "int" in expected_normalized:
                            self.data[col] = self.data[col].astype(int)
                        elif "float" in expected_normalized:
                            self.data[col] = self.data[col].astype(float)
                        elif "datetime" in expected_normalized:
                            self.data[col] = pd.to_datetime(self.data[col])
                        elif (
                            "object" in expected_normalized
                            or "str" in expected_normalized
                        ):
                            self.data[col] = self.data[col].astype(str)

                        converted.append(col)
                        self.cleaning_log.append(
                            f"Converted '{col}' from {current_type} to {expected_type}"
                        )
                        print(f"✓ Converted '{col}': {current_type} → {expected_type}")

                    except Exception as e:
                        error_msg = (
                            f"Failed to convert '{col}' to {expected_type}: {str(e)}"
                        )
                        self.cleaning_log.append(error_msg)
                        print(f"✗ {error_msg}")

        if not mismatches:
            print("✓ All data types match expected types")

        return {
            "current_types": current_types,
            "mismatches": mismatches,
            "converted": converted,
        }

    def generate_cleaning_report(self) -> Dict[str, Any]:
        """Generate a comprehensive cleaning report.

        Creates a detailed report of all cleaning operations performed,
        including before/after statistics and a complete log of changes.

        Returns:
            dict: Dictionary containing:
                - original_shape: Original dataset dimensions
                - current_shape: Current dataset dimensions
                - rows_removed: Number of rows removed
                - columns_removed: Number of columns removed
                - cleaning_operations: List of all operations performed
                - missing_values_remaining: Count of remaining missing values
                - duplicates_remaining: Count of remaining duplicates

        Example:
            >>> cleaner = DataCleaner(fraud_data)
            >>> cleaner.check_missing_values()
            >>> cleaner.remove_duplicates()
            >>> report = cleaner.generate_cleaning_report()
            >>> print(report['rows_removed'])
        """
        current_shape = self.data.shape

        report = {
            "original_shape": self.original_shape,
            "current_shape": current_shape,
            "rows_removed": self.original_shape[0] - current_shape[0],
            "columns_removed": self.original_shape[1] - current_shape[1],
            "cleaning_operations": self.cleaning_log,
            "missing_values_remaining": self.data.isnull().sum().sum(),
            "duplicates_remaining": self.data.duplicated().sum(),
        }

        print("\n" + "=" * 60)
        print("DATA CLEANING REPORT")
        print("=" * 60)
        print(f"Original shape: {report['original_shape']}")
        print(f"Current shape:  {report['current_shape']}")
        print(f"Rows removed:   {report['rows_removed']}")
        print(f"Columns removed: {report['columns_removed']}")
        print(f"\nMissing values remaining: {report['missing_values_remaining']}")
        print(f"Duplicates remaining:     {report['duplicates_remaining']}")
        print(f"\nCleaning operations performed: {len(report['cleaning_operations'])}")
        print("\nOperation log:")
        for i, operation in enumerate(report["cleaning_operations"], 1):
            print(f"  {i}. {operation}")
        print("=" * 60 + "\n")

        return report

    def get_cleaned_data(self) -> pd.DataFrame:
        """Return the cleaned dataset.

        Returns:
            pd.DataFrame: The cleaned dataset.

        Example:
            >>> cleaner = DataCleaner(fraud_data)
            >>> cleaner.remove_duplicates()
            >>> cleaned_data = cleaner.get_cleaned_data()
        """
        return self.data.copy()
