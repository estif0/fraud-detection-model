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

### Task 2: Model Performance

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

### Task 3: Model Explainability ⭐ **NEW**

**Built-in Feature Importance:**
- `feature_importance_builtin.png` - Top 10 features from model
- `importance_comparison.png` - Built-in vs SHAP importance comparison

**SHAP Global Analysis:**
- `shap_summary_plot.png` - SHAP beeswarm plot (global importance)
- `shap_bar_plot.png` - Mean absolute SHAP values bar chart
- `shap_dependence_V14.png` - Feature interaction plot for V14
- `fraud_drivers_summary.png` - Top fraud drivers visualization

**SHAP Local Analysis:**
- `shap_force_tp.html` - Interactive force plot for True Positive
- `shap_waterfall_tp.png` - Waterfall plot showing feature contributions
- `explainability_confusion_matrix.png` - Model performance baseline

**Comparison Data:**
- `feature_importance_comparison.csv` - Side-by-side importance rankings

## Summary Statistics

- `fraud_data_summary_stats.csv` - Descriptive statistics for fraud dataset

## Reports

### Completed ✅
- `INTERIM_1_REPORT.md` - Task 1 findings and insights (EDA & Preprocessing)
- `BUSINESS_INSIGHTS_REPORT.md` - Task 3 explainability analysis and recommendations
- Various completion reports in `../docs/local/`:
  - `TASK_2_COMPLETION_REPORT.md`
  - `TASK_3_COMPLETION_REPORT.md`
  - `CODEBASE_EVALUATION_REPORT.md`

## Model Artifacts

Trained models saved to `../models/`:
- `best_model_xgboost_tuned.pkl` - Best performing model (XGBoost with tuned hyperparameters)
- Includes model weights, hyperparameters, training metadata, and timestamp

## Summary Data

- `fraud_data_summary_stats.csv` - Descriptive statistics for fraud dataset
- `feature_importance_comparison.csv` - Built-in vs SHAP importance rankings
