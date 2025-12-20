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
