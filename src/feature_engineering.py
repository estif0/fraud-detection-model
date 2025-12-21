"""Feature engineering module for fraud detection project.

This module provides a `FeatureEngineer` class to create temporal and
behavioral features, encode categorical variables, and scale numerical
features. It is designed to be imported and used in notebooks and
training scripts to ensure modular, reproducible pipelines.

Classes:
    FeatureEngineer: Create and transform features for modeling.
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler


@dataclass
class FeatureEngineer:
    """Feature engineering utilities for fraud datasets.

    Attributes:
        datetime_columns (Dict[str, str]): Mapping of columns to parse as datetime.
        id_column (str): User identifier column.
        target_column (str): Target variable column name.

    Example:
        >>> fe = FeatureEngineer(datetime_columns={'signup_time': 'signup_time', 'purchase_time': 'purchase_time'})
        >>> df = fe.create_time_features(df)
        >>> df = fe.calculate_time_since_signup(df)
        >>> df = fe.calculate_transaction_frequency(df)
        >>> X_enc, enc = fe.encode_categorical_features(df, ['browser', 'source'])
        >>> X_scaled, scaler = fe.scale_numerical_features(X_enc, method='standard')
    """

    datetime_columns: Dict[str, str] = None
    id_column: str = "user_id"
    target_column: Optional[str] = None

    def _ensure_datetime(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col])
        return df

    def create_time_features(
        self, df: pd.DataFrame, time_col: str = "purchase_time"
    ) -> pd.DataFrame:
        """Create temporal features like hour_of_day and day_of_week.

        Args:
            df (pd.DataFrame): Input dataframe.
            time_col (str): Timestamp column to derive features from.

        Returns:
            pd.DataFrame: Dataframe with new temporal features.
        """
        df = df.copy()
        if time_col not in df.columns:
            return df
        df = self._ensure_datetime(df, time_col)
        df["hour_of_day"] = df[time_col].dt.hour
        df["day_of_week"] = df[time_col].dt.dayofweek
        df["day_name"] = df[time_col].dt.day_name()
        return df

    def calculate_time_since_signup(
        self,
        df: pd.DataFrame,
        signup_col: str = "signup_time",
        purchase_col: str = "purchase_time",
    ) -> pd.DataFrame:
        """Calculate duration between signup and purchase in minutes.

        Args:
            df (pd.DataFrame): Input dataframe.
            signup_col (str): Signup timestamp column.
            purchase_col (str): Purchase timestamp column.

        Returns:
            pd.DataFrame: Dataframe with `time_since_signup_min`.
        """
        df = df.copy()
        if signup_col in df.columns and purchase_col in df.columns:
            df = self._ensure_datetime(df, signup_col)
            df = self._ensure_datetime(df, purchase_col)
            delta = (df[purchase_col] - df[signup_col]).dt.total_seconds() / 60.0
            df["time_since_signup_min"] = delta
        return df

    def calculate_transaction_velocity(
        self, df: pd.DataFrame, window_hours: int = 24, time_col: str = "purchase_time"
    ) -> pd.DataFrame:
        """Compute rolling transaction counts per user over a time window.

        A simple approximation computes per-user counts in the last `window_hours`.
        Requires `user_id` and a datetime `purchase_time` column.

        Args:
            df (pd.DataFrame): Input dataframe sorted by time.
            window_hours (int): Rolling window size in hours.
            time_col (str): Timestamp column name.

        Returns:
            pd.DataFrame: Dataframe with `txn_velocity_{window_hours}h` per row.
        """
        df = df.copy()
        if self.id_column not in df.columns or time_col not in df.columns:
            return df
        df = self._ensure_datetime(df, time_col)
        df = df.sort_values([self.id_column, time_col])

        # For each user, compute count of transactions in past window_hours
        velocity_col = f"txn_velocity_{window_hours}h"
        df[velocity_col] = 0
        window = pd.Timedelta(hours=window_hours)
        for uid, grp in df.groupby(self.id_column):
            times = grp[time_col]
            counts = []
            for t in times:
                counts.append(((times >= t - window) & (times <= t)).sum())
            df.loc[grp.index, velocity_col] = counts
        return df

    def calculate_transaction_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-user transaction frequency (total transactions per user).

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with `transactions_per_user` feature.
        """
        df = df.copy()
        if self.id_column not in df.columns:
            return df
        counts = df.groupby(self.id_column).size().rename("transactions_per_user")
        df = df.merge(counts, left_on=self.id_column, right_index=True, how="left")
        return df

    def create_aggregated_features(
        self, df: pd.DataFrame, value_col: str = "purchase_value"
    ) -> pd.DataFrame:
        """Create simple aggregated user-level statistics for a value column.

        Args:
            df (pd.DataFrame): Input dataframe.
            value_col (str): Column to aggregate.

        Returns:
            pd.DataFrame: Dataframe with per-user mean and std of `value_col`.
        """
        df = df.copy()
        if self.id_column not in df.columns or value_col not in df.columns:
            return df
        agg = (
            df.groupby(self.id_column)[value_col]
            .agg(["mean", "std"])
            .rename(columns={"mean": f"{value_col}_mean", "std": f"{value_col}_std"})
        )
        df = df.merge(agg, left_on=self.id_column, right_index=True, how="left")
        return df

    def encode_categorical_features(
        self, df: pd.DataFrame, categorical_cols: List[str]
    ) -> Tuple[pd.DataFrame, OneHotEncoder]:
        """Apply one-hot encoding to categorical columns.

        Args:
            df (pd.DataFrame): Input dataframe.
            categorical_cols (List[str]): Categorical columns to encode.

        Returns:
            Tuple[pd.DataFrame, OneHotEncoder]: Transformed dataframe and fitted encoder.
        """
        df = df.copy()
        present = [c for c in categorical_cols if c in df.columns]
        if not present:
            return df, OneHotEncoder(handle_unknown="ignore")

        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded = enc.fit_transform(df[present])
        encoded_df = pd.DataFrame(
            encoded, columns=enc.get_feature_names_out(present), index=df.index
        )
        df = pd.concat([df.drop(columns=present), encoded_df], axis=1)
        return df, enc

    def scale_numerical_features(
        self,
        df: pd.DataFrame,
        method: str = "standard",
        columns: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, Any]:
        """Scale numerical features using StandardScaler or MinMaxScaler.

        Args:
            df (pd.DataFrame): Input dataframe.
            method (str): 'standard' or 'minmax'.
            columns (Optional[List[str]]): Columns to scale; if None, auto-detect numericals.

        Returns:
            Tuple[pd.DataFrame, Any]: Transformed dataframe and fitted scaler.
        """
        df = df.copy()
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if self.target_column and self.target_column in columns:
                columns.remove(self.target_column)
        if not columns:
            return df, None

        scaler = StandardScaler() if method == "standard" else MinMaxScaler()
        df[columns] = scaler.fit_transform(df[columns])
        return df, scaler
