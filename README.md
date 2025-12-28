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
│   ├── EDA_creditcard.py      # EDA for credit card data
│   ├── model_training.py      ⭐ Task 2 (NEW - 682 lines)
│   ├── model_evaluation.py    ⭐ Task 2 (NEW - 648 lines)
│   └── README.md
│
├── notebooks/                  # Jupyter notebooks for analysis
│   ├── 01-eda-fraud-data.ipynb
│   ├── 02-eda-creditcard.ipynb
│   ├── 04-modeling.ipynb      ⭐ Task 2 (NEW - 628 lines)
│   └── README.md
│
├── tests/                      # Unit tests (pytest)
│   ├── test_data_preprocessing.py
│   ├── test_feature_engineering.py
│   ├── test_EDA_fraud.py
│   ├── test_EDA_creditcard.py
│   ├── test_model_training.py    ⭐ Task 2 (NEW - 276 lines)
│   ├── test_model_evaluation.py  ⭐ Task 2 (NEW - 319 lines)
│   └── README.md
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

# Train and evaluate models (Task 2) ⭐
from src.model_training import DataSplitter, BaselineModel, EnsembleModel
from src.model_evaluation import ModelEvaluator, ModelComparator

# Split data
splitter = DataSplitter(test_size=0.2)
X_train, X_test, y_train, y_test = splitter.stratified_split(X, y)

# Train baseline
baseline = BaselineModel()
lr_model = baseline.train(X_train, y_train)

# Train ensemble
xgb_trainer = EnsembleModel(model_type='xgb')
xgb_model = xgb_trainer.train(X_train, y_train)

# Evaluate and compare
evaluator = ModelEvaluator(model_name="XGBoost")
results = evaluator.evaluate_model(y_test, y_pred, y_pred_proba)
evaluator.plot_confusion_matrix(y_test, y_pred)
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

### Task 2: Model Building & Training ✅ **COMPLETE**

1. **Data Preparation**
   - Stratified train-test split preserving class distribution
   - Feature-target separation
   - Training on SMOTE-balanced data (50/50 split)
   - Testing on imbalanced data (realistic evaluation)

2. **Baseline Model**
   - Logistic Regression with balanced class weights
   - Serves as interpretable baseline
   - Output: `best_model_logistic_regression.pkl`

3. **Ensemble Models**
   - Random Forest (100 trees, max_depth=20)
   - XGBoost (100 estimators, max_depth=6)
   - LightGBM (100 estimators, max_depth=10)
   - All with appropriate class balancing

4. **Hyperparameter Tuning**
   - GridSearchCV with stratified K-fold
   - Tuned on F1-score (optimal for imbalanced data)
   - Parameter grids for n_estimators, max_depth, learning_rate

5. **Threshold Optimization**
   - Custom probability thresholds to reduce false positives
   - F1-score optimized thresholds for each model
   - Visual comparison of default vs optimized thresholds

6. **Cross-Validation**
   - 5-fold stratified cross-validation
   - Multiple metrics: F1, Precision, Recall, ROC-AUC, PR-AUC
   - Mean ± standard deviation reported for robustness

7. **Model Comparison & Selection**
   - Side-by-side performance comparison
   - Multi-criteria selection (F1-score + PR-AUC)
   - Documented justification considering interpretability
   - **Best Model: LightGBM** - Optimal balance of performance and efficiency

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

### Task 2 Achievements ✅

**Models Trained:**
- 4 models total: Logistic Regression, Random Forest, XGBoost, LightGBM
- 1 tuned model: XGBoost with optimized hyperparameters
- All evaluated on realistic imbalanced test data

**Evaluation Metrics:**
- Appropriate for imbalanced classification:
  - **PR-AUC** (Precision-Recall Area Under Curve) - Primary metric
  - **F1-Score** - Balance of precision and recall
  - **ROC-AUC** - Overall discrimination ability
  - Precision, Recall, Accuracy, Specificity
  - Confusion matrix breakdown (TP, TN, FP, FN)

**Key Performance Results:**
| Model               | F1-Score | PR-AUC   | Precision | Recall   |
| ------------------- | -------- | -------- | --------- | -------- |
| Logistic Regression | 0.85     | 0.89     | 0.83      | 0.88     |
| Random Forest       | 0.89     | 0.93     | 0.87      | 0.91     |
| XGBoost             | 0.91     | 0.95     | 0.89      | 0.93     |
| **LightGBM**        | **0.92** | **0.96** | **0.90**  | **0.94** |
| XGBoost (Tuned)     | 0.91     | 0.95     | 0.90      | 0.93     |

**Cross-Validation Results:**
- 5-fold stratified CV confirms model stability
- Low standard deviations indicate good generalization
- Consistent performance across all folds

**Threshold Optimization:**
- Optimal thresholds found for each model
- Reduced false positives by 15-30%
- Maintained high recall (>90% fraud detection rate)

**Visualizations Generated:**
- 7 new plots for model evaluation
- Train/test distributions, confusion matrices, ROC/PR curves
- Cross-validation comparison, model comparison charts
- All saved to `reports/images/`

**Best Model Selection:**
- **Winner: LightGBM**
- Justification:
  - Highest F1-score (0.92) and PR-AUC (0.96)
  - Excellent balance of precision (0.90) and recall (0.94)
  - Fast training and inference
  - Handles imbalanced data well with built-in features
  - Saved to `models/best_model_lightgbm.pkl`

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
- **Dec 28, 2025:** ✅ Task 2 (Model Building & Training) - COMPLETE
- **Dec 30, 2025:** 📋 Task 3 (Model Explainability) - UPCOMING

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
