"""Run end-to-end feature engineering pipeline for fraud dataset.

This script loads a cleaned fraud dataset from `data/processed/`, builds
temporal and behavioral features, encodes categoricals, scales numericals,
and saves the engineered dataset.

Usage example:
    python scripts/run_feature_engineering.py \
        --input data/processed/cleaned_fraud.csv \
        --output data/processed/engineered_fraud.csv \
        --scaler standard \
        --categoricals browser source sex
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd

# Add project root to Python path for src imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.feature_engineering import FeatureEngineer


def run_feature_engineering(args: argparse.Namespace) -> None:
    logging.info("Loading cleaned dataset: %s", args.input)
    df = pd.read_csv(args.input)

    fe = FeatureEngineer(target_column="class")

    logging.info("Creating temporal features")
    df = fe.create_time_features(df, time_col="purchase_time")
    df = fe.calculate_time_since_signup(
        df, signup_col="signup_time", purchase_col="purchase_time"
    )

    logging.info("Computing transaction frequency and velocity")
    df = fe.calculate_transaction_frequency(df)
    df = fe.calculate_transaction_velocity(
        df, window_hours=args.velocity_window, time_col="purchase_time"
    )

    logging.info("Creating aggregated user-level features")
    if "purchase_value" in df.columns:
        df = fe.create_aggregated_features(df, value_col="purchase_value")

    # Encode categoricals
    cats: List[str] = args.categoricals
    logging.info("Encoding categoricals: %s", ", ".join(cats) if cats else "(none)")
    df_enc, _ = fe.encode_categorical_features(df, cats)

    # Scale numericals
    logging.info("Scaling numericals with %s scaler", args.scaler)
    df_scaled, _ = fe.scale_numerical_features(df_enc, method=args.scaler)

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_scaled.to_csv(output_path, index=False)
    logging.info("Saved engineered dataset to: %s", output_path)


def build_argparser() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run feature engineering for fraud dataset")
    p.add_argument(
        "--input",
        default="data/processed/cleaned_fraud.csv",
        help="Input cleaned fraud CSV path",
    )
    p.add_argument(
        "--output",
        default="data/processed/engineered_fraud.csv",
        help="Output engineered CSV path",
    )
    p.add_argument(
        "--velocity-window", type=int, default=24, help="Velocity window in hours"
    )
    p.add_argument(
        "--categoricals",
        nargs="*",
        default=["browser", "source", "sex"],
        help="Categorical columns to encode",
    )
    p.add_argument(
        "--scaler",
        choices=["standard", "minmax"],
        default="standard",
        help="Scaling method for numerical features",
    )
    p.add_argument("--log-level", default="INFO", help="Logging level")
    args = p.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )
    return args


if __name__ == "__main__":
    run_feature_engineering(build_argparser())
