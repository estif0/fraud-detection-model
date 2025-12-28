# Reports

Analysis reports, visualizations, and model performance summaries.

## Images

Generated visualizations saved to `images/`:

### Task 1: Exploratory Data Analysis ✅

**Fraud Data Analysis:**
- `univariate_numerical.png` - Numerical feature distributions
- `univariate_categorical.png` - Categorical feature distributions
- `bivariate_numerical.png` - Feature-target relationships (numerical)
- `bivariate_categorical.png` - Feature-target relationships (categorical)
- `class_imbalance.png` - Fraud vs legitimate distribution
- `temporal_patterns.png` - Time-based fraud patterns
- `categorical_features_analysis.png` - Categorical feature analysis

**Credit Card Analysis:**
- `cc_pca_distributions.png` - PCA components V1-V28 distributions
- `cc_correlation_heatmap.png` - Feature correlation matrix
- `cc_class_imbalance.png` - Class distribution

### Task 2: Model Performance ⭐ **NEW**

**Data Preparation:**
- `train_test_distribution.png` - Class distribution in train/test splits

**Model Evaluation:**
- `lr_confusion_matrix.png` - Logistic Regression confusion matrix
- `lr_roc_curve.png` - Logistic Regression ROC curve
- `lr_pr_curve.png` - Logistic Regression Precision-Recall curve

**Threshold Optimization:**
- `xgb_threshold_comparison.png` - Default vs optimized threshold comparison

**Model Comparison:**
- `cv_comparison.png` - Cross-validation results across all models
- `model_comparison.png` - Side-by-side metric comparison

## Summary Statistics

- `fraud_data_summary_stats.csv` - Descriptive statistics for fraud dataset

## Reports

### Completed ✅
- `INTERIM_1_REPORT.md` - Task 1 findings and insights (EDA & Preprocessing)

### Upcoming 📋
- `INTERIM_2_REPORT.md` - Task 2 model performance and selection (Due: Dec 28, 2025)
- `FINAL_REPORT.md` - Complete project analysis with SHAP explainability (Due: Dec 30, 2025)

## Model Artifacts

Trained models saved to `../models/`:
- `best_model_lightgbm.pkl` - Best performing model (LightGBM)
- Includes model weights, hyperparameters, and metadata
