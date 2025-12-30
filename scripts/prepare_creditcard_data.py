#!/usr/bin/env python3
"""Prepare credit card data with proper feature engineering.

This script fixes the data preparation pipeline by preserving all V1-V28
PCA features during feature engineering, train-test splitting, scaling,
and SMOTE resampling.

Usage:
    python scripts/prepare_creditcard_data.py

Output:
    - data/processed/cc_train_scaled_full.csv (30 features)
    - data/processed/cc_test_scaled_full.csv (30 features)
    - data/processed/cc_train_smote_full.csv (30 features, balanced)
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from src.data_preprocessing import DataLoader
from src.feature_engineering import FeatureEngineer


def main():
    """Main pipeline to prepare credit card data with all features."""

    print("=" * 80)
    print("CREDIT CARD DATA PREPARATION PIPELINE")
    print("=" * 80)
    print()

    # Configuration
    input_file = "data/processed/cleaned_creditcard.csv"
    output_dir = Path("data/processed")
    test_size = 0.2
    random_state = 42

    # Step 1: Load cleaned data
    print("Step 1: Loading cleaned credit card data...")
    print("-" * 80)

    if not Path(input_file).exists():
        # Try loading from raw data
        print(f"  {input_file} not found. Loading from raw data...")
        loader = DataLoader()
        cc_data = loader.load_creditcard_data()
    else:
        cc_data = pd.read_csv(input_file)
        print(f"  ✓ Loaded from {input_file}")

    print(f"  Shape: {cc_data.shape}")
    print(f"  Columns: {list(cc_data.columns)[:5]}... (showing first 5)")
    print()

    # Validate we have all required features
    required_pca = [f"V{i}" for i in range(1, 29)]
    missing_pca = [f for f in required_pca if f not in cc_data.columns]

    if missing_pca:
        print(f"❌ ERROR: Missing PCA features: {missing_pca}")
        print("  The input data must contain V1-V28 features.")
        return 1

    print(f"  ✓ All V1-V28 PCA features present")
    print()

    # Step 2: Feature Engineering (preserve V1-V28)
    print("Step 2: Feature engineering (preserving V1-V28)...")
    print("-" * 80)

    fe = FeatureEngineer()
    cc_engineered = fe.engineer_creditcard_features(
        cc_data, preserve_pca=True, scale_features=False  # We'll scale after split
    )

    print()

    # Verify feature count
    feature_count = len(cc_engineered.columns) - 1  # Exclude Class
    if feature_count != 30:
        print(f"⚠️  WARNING: Expected 30 features, got {feature_count}")
    else:
        print(f"  ✓ Feature count verified: {feature_count} features")

    print()

    # Step 3: Train-Test Split (Stratified)
    print("Step 3: Train-test split (stratified, 80/20)...")
    print("-" * 80)

    X = cc_engineered.drop("Class", axis=1)
    y = cc_engineered["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"  Train set: {X_train.shape[0]} samples")
    print(f"    Fraud: {y_train.sum()} ({y_train.mean():.2%})")
    print(f"    Legitimate: {(y_train == 0).sum()} ({(y_train == 0).mean():.2%})")
    print()
    print(f"  Test set: {X_test.shape[0]} samples")
    print(f"    Fraud: {y_test.sum()} ({y_test.mean():.2%})")
    print(f"    Legitimate: {(y_test == 0).sum()} ({(y_test == 0).mean():.2%})")
    print()

    # Step 4: Feature Scaling
    print("Step 4: Feature scaling (StandardScaler)...")
    print("-" * 80)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    print(f"  ✓ Features scaled using StandardScaler")
    print(f"  ✓ Scaler fitted on training data only")
    print()

    # Step 5: Save scaled train/test sets
    print("Step 5: Saving scaled train/test sets...")
    print("-" * 80)

    train_scaled_path = output_dir / "cc_train_scaled_full.csv"
    test_scaled_path = output_dir / "cc_test_scaled_full.csv"

    train_scaled = pd.concat([X_train_scaled, y_train], axis=1)
    test_scaled = pd.concat([X_test_scaled, y_test], axis=1)

    train_scaled.to_csv(train_scaled_path, index=False)
    test_scaled.to_csv(test_scaled_path, index=False)

    print(f"  ✓ Saved: {train_scaled_path}")
    print(f"    Shape: {train_scaled.shape}")
    print(f"  ✓ Saved: {test_scaled_path}")
    print(f"    Shape: {test_scaled.shape}")
    print()

    # Step 6: Apply SMOTE to training data
    print("Step 6: Applying SMOTE to training data...")
    print("-" * 80)

    print(f"  Before SMOTE:")
    print(f"    Fraud: {y_train.sum()} ({y_train.mean():.2%})")
    print(f"    Legitimate: {(y_train == 0).sum()}")
    print(f"    Ratio: 1:{int((y_train == 0).sum() / y_train.sum())}")
    print()

    smote = SMOTE(random_state=random_state, n_jobs=-1)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    print(f"  After SMOTE:")
    print(f"    Fraud: {y_train_smote.sum()} ({y_train_smote.mean():.2%})")
    print(f"    Legitimate: {(y_train_smote == 0).sum()}")
    print(f"    Ratio: 1:{int((y_train_smote == 0).sum() / y_train_smote.sum())}")
    print(f"  ✓ Dataset balanced to 1:1 ratio")
    print()

    # Step 7: Save SMOTE-resampled training data
    print("Step 7: Saving SMOTE-resampled training data...")
    print("-" * 80)

    train_smote_path = output_dir / "cc_train_smote_full.csv"

    # Convert back to DataFrame
    X_train_smote_df = pd.DataFrame(X_train_smote, columns=X_train_scaled.columns)
    train_smote = pd.concat(
        [X_train_smote_df, pd.Series(y_train_smote, name="Class")], axis=1
    )

    train_smote.to_csv(train_smote_path, index=False)

    print(f"  ✓ Saved: {train_smote_path}")
    print(f"    Shape: {train_smote.shape}")
    print()

    # Step 8: Final Validation
    print("Step 8: Final validation...")
    print("-" * 80)

    # Verify all files
    files_to_check = [train_scaled_path, test_scaled_path, train_smote_path]
    all_valid = True

    for file_path in files_to_check:
        df = pd.read_csv(file_path)
        feature_cols = [c for c in df.columns if c != "Class"]

        # Check feature count
        if len(feature_cols) != 30:
            print(
                f"  ❌ {file_path.name}: Wrong feature count ({len(feature_cols)} != 30)"
            )
            all_valid = False
        else:
            print(f"  ✓ {file_path.name}: {len(feature_cols)} features")

        # Check V1-V28 present
        pca_present = sum(1 for c in feature_cols if c.startswith("V"))
        if pca_present != 28:
            print(f"    ⚠️  Only {pca_present}/28 PCA features found")
            all_valid = False
        else:
            print(f"    ✓ All 28 PCA features (V1-V28) present")

        # Check target
        if "Class" not in df.columns:
            print(f"    ❌ Target column 'Class' missing")
            all_valid = False
        else:
            print(f"    ✓ Target column 'Class' present")

    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    if all_valid:
        print("✅ SUCCESS! All files generated correctly with 30 features.")
        print()
        print("Generated Files:")
        print(f"  1. {train_scaled_path.name}")
        print(f"     - {train_scaled.shape[0]:,} samples, 30 features + target")
        print(f"     - Imbalanced (original distribution)")
        print()
        print(f"  2. {test_scaled_path.name}")
        print(f"     - {test_scaled.shape[0]:,} samples, 30 features + target")
        print(f"     - Imbalanced (realistic evaluation)")
        print()
        print(f"  3. {train_smote_path.name}")
        print(f"     - {train_smote.shape[0]:,} samples, 30 features + target")
        print(f"     - Balanced 1:1 (for training)")
        print()
        print("Next Steps:")
        print("  1. Update modeling notebook to use these new files")
        print("  2. Retrain models on 30 features")
        print("  3. Compare performance with old 2-feature model")
        print()
        return 0
    else:
        print("❌ VALIDATION FAILED! Check the warnings above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
