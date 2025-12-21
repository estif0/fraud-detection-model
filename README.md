# Fraud Detection Model 🔍💳

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange.svg)](https://scikit-learn.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Advanced machine learning solution for detecting fraudulent transactions in e-commerce and bank credit card systems using ensemble methods, feature engineering, and model explainability techniques.

## 📋 Table of Contents

- [Business Problem](#business-problem)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Methodology](#methodology)
- [Results](#results)
- [Technologies](#technologies)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Business Problem

**Client:** Adey Innovations Inc. (Financial Technology Sector)

Fraudulent transactions pose significant financial risks to both e-commerce platforms and banking institutions. This project develops robust fraud detection models that:

- **Minimize False Positives:** Reduce customer friction from legitimate transactions being flagged
- **Maximize Fraud Detection:** Catch fraudulent transactions before financial loss occurs
- **Provide Explainability:** Understand which features drive fraud predictions for actionable insights
- **Handle Class Imbalance:** Work effectively with highly imbalanced datasets (fraud is rare)

**Impact:** Effective fraud detection can save millions in losses while maintaining excellent customer experience.

## 📊 Datasets

### 1. E-commerce Fraud Data (`Fraud_Data.csv`)
- **151,112 transactions** with 11 features
- **Target:** `class` (1=fraud, 0=legitimate)
- **Features:** user_id, signup_time, purchase_time, purchase_value, device_id, source, browser, sex, age, ip_address
- **Challenge:** Class imbalance (~6% fraud rate)

### 2. Credit Card Transactions (`creditcard.csv`)
- **284,807 transactions** with 31 features
- **Target:** `Class` (1=fraud, 0=legitimate)
- **Features:** Time, Amount, V1-V28 (PCA-transformed features)
- **Challenge:** Extreme class imbalance (~0.17% fraud rate)

### 3. IP Geolocation Data (`IpAddress_to_Country.csv`)
- IP address range to country mapping
- Used for geolocation-based fraud pattern analysis

## 📁 Project Structure

```
fraud-detection-model/
│
├── data/
│   ├── raw/                    # Original datasets (not in git)
│   └── processed/              # Cleaned and transformed data (not in git)
│
├── src/                        # Source code modules
│   ├── data_preprocessing.py   # Data loading, cleaning, IP mapping, imbalance handling
│   ├── feature_engineering.py  # Feature creation, encoding, scaling
│   ├── EDA_fraud.py           # EDA for e-commerce fraud data
│   └── EDA_creditcard.py      # EDA for credit card data
│
├── notebooks/                  # Jupyter notebooks for analysis
│   ├── 01-eda-fraud-data.ipynb
│   └── 02-eda-creditcard.ipynb
│
├── tests/                      # Unit tests (pytest)
│   ├── test_data_preprocessing.py
│   ├── test_feature_engineering.py
│   ├── test_EDA_fraud.py
│   └── test_EDA_creditcard.py
│
├── scripts/                    # Executable pipeline scripts
│   ├── run_data_preprocessing.py
│   └── run_feature_engineering.py
│
├── reports/                    # Analysis reports and visualizations
│   └── images/                # Generated plots and charts
│
├── models/                     # Saved trained models (Task 2)
│
├── docs/                       # Project documentation
│   └── local/                 # Local development docs
│
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/fraud-detection-model.git
cd fraud-detection-model
```

2. **Create and activate virtual environment**
```bash
# Linux/Mac
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Place datasets in `data/raw/`**
```
data/raw/
├── Fraud_Data.csv
├── creditcard.csv
└── IpAddress_to_Country.csv
```

## 🏃 Quick Start

### Option 1: Using Jupyter Notebooks

```bash
jupyter notebook
# Navigate to notebooks/ and open 01-eda-fraud-data.ipynb
```

### Option 2: Using Command-Line Scripts

**Data Preprocessing:**
```bash
# Clean fraud data with IP mapping
python scripts/run_data_preprocessing.py \
    --dataset fraud \
    --clean-strategy drop \
    --ip-map \
    --output cleaned_fraud_with_country.csv

# Clean credit card data
python scripts/run_data_preprocessing.py \
    --dataset creditcard \
    --clean-strategy median \
    --output cleaned_creditcard.csv
```

**Feature Engineering:**
```bash
python scripts/run_feature_engineering.py \
    --input data/processed/cleaned_fraud.csv \
    --output data/processed/engineered_fraud.csv \
    --scaler standard
```

### Option 3: Using Python API

```python
from src.data_preprocessing import DataLoader, DataCleaner, IPMapper
from src.feature_engineering import FeatureEngineer
from src.EDA_fraud import FraudDataEDA

# Load and clean data
loader = DataLoader(data_dir='data/raw')
fraud_data = loader.load_fraud_data()

cleaner = DataCleaner(fraud_data)
cleaned_data = cleaner.handle_missing_values(strategy='drop')

# Perform EDA
eda = FraudDataEDA(cleaned_data, output_dir='reports/images')
eda.univariate_analysis()
eda.bivariate_analysis()
eda.analyze_class_imbalance()

# Engineer features
fe = FeatureEngineer()
data = fe.create_time_features(cleaned_data)
data = fe.calculate_transaction_frequency(data)
data_scaled, scaler = fe.scale_numerical_features(data)
```

## 🔬 Methodology

### Task 1: Data Analysis & Preprocessing ✅ **COMPLETE**

1. **Data Cleaning**
   - Missing value analysis and handling (drop, impute, fill)
   - Duplicate removal
   - Data type validation and correction
   - Output: `cleaned_fraud.csv`, `cleaned_creditcard.csv`

2. **Exploratory Data Analysis (EDA)**
   - Univariate analysis: Feature distributions
   - Bivariate analysis: Feature-target relationships
   - Class imbalance quantification
   - Temporal pattern analysis
   - Output: 10 visualizations in `reports/images/`

3. **Geolocation Analysis**
   - IP address to integer conversion
   - Country mapping using range-based lookup
   - Fraud pattern analysis by geography
   - Output: `cleaned_fraud_with_country.csv`

4. **Feature Engineering**
   - **Time features:** hour_of_day, day_of_week, time_since_signup
   - **Behavioral features:** transaction_velocity, transaction_frequency
   - **Aggregations:** per-user statistics (mean, std)
   - **Encoding:** One-hot encoding for categoricals
   - **Scaling:** StandardScaler and MinMaxScaler

5. **Class Imbalance Handling**
   - SMOTE oversampling (synthetic minority samples)
   - Random undersampling (reduce majority class)
   - SMOTE+ENN combined strategy
   - Output: `cc_train_smote_resampled.csv`

### Task 2: Model Building & Training 🚧 **IN PROGRESS**

- Stratified train-test split
- Baseline models (Logistic Regression)
- Ensemble models (Random Forest, XGBoost, LightGBM)
- Cross-validation (Stratified K-Fold, k=5)
- Hyperparameter tuning (GridSearchCV)
- Model comparison using appropriate metrics

### Task 3: Model Explainability 📋 **PLANNED**

- Feature importance extraction
- SHAP global analysis (summary plots)
- SHAP local analysis (force plots)
- Business recommendations

## 📈 Results

### Task 1 Highlights

**Data Quality:**
- ✅ Both datasets cleaned and validated
- ✅ Zero missing values after preprocessing
- ✅ All data types corrected

**EDA Insights:**
- **Fraud Rate:** E-commerce ~6%, Credit Card ~0.17%
- **Imbalance Ratio:** 1:15 (e-commerce), 1:577 (credit card)
- **Key Features:** Transaction timing, user behavior, transaction amounts show strong fraud signals
- **Geographic Patterns:** Certain countries show higher fraud rates

**Feature Engineering:**
- Created 10+ new engineered features
- Applied SMOTE resampling: balanced 1:1 ratio in training data
- Scaled features for model compatibility

**Visualizations Generated:**
- 10 comprehensive plots covering univariate, bivariate, temporal, and imbalance analysis
- All saved to `reports/images/`

### Model Performance (Task 2 - Coming Soon)
*Results will be updated after Task 2 completion*

## 🛠 Technologies

**Core:**
- Python 3.8+
- pandas 2.x - Data manipulation
- numpy - Numerical computing
- scikit-learn 1.3.2 - Machine learning

**Visualization:**
- matplotlib - Static plotting
- seaborn - Statistical visualizations
- plotly - Interactive charts

**Modeling:**
- imbalanced-learn 0.11.0 - SMOTE and sampling techniques
- xgboost - Gradient boosting
- lightgbm - Gradient boosting
- shap - Model explainability

**Development:**
- jupyter - Interactive notebooks
- pytest - Unit testing
- black - Code formatting
- flake8 - Linting

## 🧪 Testing

Run all tests:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

Run specific test file:
```bash
pytest tests/test_data_preprocessing.py -v
```

## 📝 Project Timeline

- **Dec 21, 2025:** ✅ Task 1 (Data Analysis & Preprocessing) - COMPLETE
- **Dec 28, 2025:** 🚧 Task 2 (Model Building & Training) - IN PROGRESS
- **Dec 30, 2025:** 📋 Task 3 (Model Explainability) - PLANNED

## 👥 Contributing

This is an academic project for Adey Innovations Inc. For questions or suggestions, please contact the project maintainer.

## 📄 License

See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Adey Innovations Inc. for the business problem and guidance
- Dataset sources: E-commerce platform logs and credit card transaction database
- 10 Academy Data Science Program

---

**Note:** Raw data files are not included in the repository due to size and privacy concerns. Place datasets in `data/raw/` before running analyses.
