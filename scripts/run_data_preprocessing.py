"""Run end-to-end data preprocessing pipeline.

This script loads datasets, applies cleaning, optional IP-to-country mapping,
handles basic exports, and saves outputs to `data/processed/`.

Usage examples:
    python scripts/run_data_preprocessing.py --dataset fraud --clean-strategy drop
    python scripts/run_data_preprocessing.py --dataset fraud --ip-map --clean-strategy fill \
        --fill "age=0" --fill "source=Unknown" --output cleaned_fraud.csv

    python scripts/run_data_preprocessing.py --dataset creditcard --clean-strategy median \
        --output cleaned_creditcard.csv
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd

# Add project root to Python path for src imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src.data_preprocessing import DataLoader, DataCleaner, IPMapper


def parse_fill_args(items: list[str]) -> Dict[str, Any]:
    """Parse key=value pairs from CLI into a dict."""
    result: Dict[str, Any] = {}
    for it in items:
        if "=" not in it:
            continue
        k, v = it.split("=", 1)
        # try numeric conversion
        try:
            if "." in v:
                result[k] = float(v)
            else:
                result[k] = int(v)
        except ValueError:
            result[k] = v
    return result


def run_preprocessing(args: argparse.Namespace) -> None:
    logging.info("Initializing DataLoader")
    loader = DataLoader(data_dir=args.data_dir, processed_dir=args.processed_dir)

    if args.dataset == "fraud":
        logging.info("Loading Fraud_Data.csv")
        df = loader.load_fraud_data()
    elif args.dataset == "creditcard":
        logging.info("Loading creditcard.csv")
        df = loader.load_creditcard_data()
    else:
        raise ValueError("Unsupported dataset. Choose 'fraud' or 'creditcard'.")

    logging.info("Starting cleaning with strategy=%s", args.clean_strategy)
    cleaner = DataCleaner(df)
    cleaner.check_missing_values()

    if args.clean_strategy == "fill":
        fill_map = parse_fill_args(args.fill)
        df_clean = cleaner.handle_missing_values(strategy="fill", fill_value=fill_map)
    elif args.clean_strategy in {
        "drop",
        "drop_columns",
        "forward_fill",
        "mean",
        "median",
        "mode",
    }:
        extra = {}
        if args.clean_strategy == "drop_columns":
            extra["threshold"] = args.drop_threshold
        df_clean = cleaner.handle_missing_values(strategy=args.clean_strategy, **extra)
    else:
        raise ValueError("Invalid clean strategy")

    df_clean = cleaner.remove_duplicates()

    # Validate a few common types if present
    expected = {}
    if args.dataset == "fraud":
        expected = {"user_id": "int", "age": "int", "purchase_value": "float"}
    elif args.dataset == "creditcard":
        expected = {"Amount": "float"}
    cleaner.validate_data_types(expected_types=expected, convert=True)
    cleaner.generate_cleaning_report()

    # Optional IP mapping for fraud dataset
    if args.dataset == "fraud" and args.ip_map:
        logging.info("Applying IP-to-country mapping")
        ip_map_path = Path(args.data_dir) / "IpAddress_to_Country.csv"
        ip_df = (
            loader.load_ip_mapping()
            if ip_map_path.exists()
            else loader.load_ip_mapping()
        )
        mapper = IPMapper(ip_df)
        df_clean = mapper.map_ip_to_country(df_clean, ip_column="ip_address")

    # Save output
    output_name = args.output or (
        "cleaned_fraud.csv" if args.dataset == "fraud" else "cleaned_creditcard.csv"
    )
    loader.save_processed_data(df_clean, output_name)


def build_argparser() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run data preprocessing pipeline")
    p.add_argument(
        "--dataset",
        choices=["fraud", "creditcard"],
        required=True,
        help="Dataset to process",
    )
    p.add_argument("--data-dir", default="data/raw", help="Raw data directory")
    p.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Processed data output directory",
    )
    p.add_argument(
        "--clean-strategy",
        default="drop",
        help="Missing data strategy: drop, drop_columns, fill, forward_fill, mean, median, mode",
    )
    p.add_argument(
        "--drop-threshold",
        type=float,
        default=0.5,
        help="Threshold for drop_columns (0-1)",
    )
    p.add_argument(
        "--fill", action="append", default=[], help="Fill pairs key=value (repeatable)"
    )
    p.add_argument(
        "--ip-map", action="store_true", help="Apply IP mapping for fraud dataset"
    )
    p.add_argument("--output", help="Output CSV filename")
    p.add_argument("--log-level", default="INFO", help="Logging level")
    args = p.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )
    return args


if __name__ == "__main__":
    run_preprocessing(build_argparser())
