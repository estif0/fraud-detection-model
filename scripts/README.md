# Scripts

Executable scripts to run the preprocessing and feature engineering pipelines.

## run_data_preprocessing.py

- Purpose: Load raw datasets, clean them, optionally map IP addresses to countries, and save processed outputs.
- Datasets: fraud (Fraud_Data.csv), creditcard (creditcard.csv)

Usage:

```bash
# Fraud dataset: drop rows with missing values
python scripts/run_data_preprocessing.py --dataset fraud --clean-strategy drop --output cleaned_fraud.csv

# Fraud dataset: fill specific columns and apply IP mapping
python scripts/run_data_preprocessing.py --dataset fraud --clean-strategy fill \
  --fill age=0 --fill source=Unknown --ip-map --output cleaned_fraud_with_country.csv

# Credit card dataset: median imputation
python scripts/run_data_preprocessing.py --dataset creditcard --clean-strategy median --output cleaned_creditcard.csv
```

Options:
- --clean-strategy: drop, drop_columns, fill, forward_fill, mean, median, mode
- --fill: key=value pairs (repeatable) for fill strategy
- --ip-map: apply IP-to-country mapping (fraud dataset only)
- --data-dir: raw data directory (default: data/raw)
- --processed-dir: processed output directory (default: data/processed)
- --output: output CSV filename

## run_feature_engineering.py

- Purpose: Build temporal and behavioral features, encode categoricals, scale numericals, and save engineered dataset.
- Input: Cleaned fraud CSV (default: data/processed/cleaned_fraud.csv)

Usage:

```bash
python scripts/run_feature_engineering.py \
  --input data/processed/cleaned_fraud.csv \
  --output data/processed/engineered_fraud.csv \
  --scaler standard \
  --categoricals browser source sex
```

Options:
- --velocity-window: window in hours for transaction velocity (default: 24)
- --categoricals: list of categorical columns to encode (default: browser source sex)
- --scaler: standard or minmax

## Notes

- Scripts import modular code from src/ and follow OOP architecture.
- Ensure raw data files exist under data/raw/ before running.
- Outputs are saved under data/processed/ by default.
