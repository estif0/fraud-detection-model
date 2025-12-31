"""Generate missing SHAP force plots for FP and FN cases."""

import sys, os
sys.path.append(os.path.abspath(".."))
import pandas as pd
import numpy as np
import joblib, shap
import matplotlib.pyplot as plt

model_data = joblib.load("../models/best_model_xgboost_tuned.pkl")
model = model_data["model"] if isinstance(model_data, dict) else model_data
test_df = pd.read_csv("../data/processed/cc_test_scaled_full.csv")
X_test = test_df.drop("Class", axis=1)
y_test = test_df["Class"]
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

fp_indices = np.where((y_test == 0) & (y_pred == 1))[0]
fn_indices = np.where((y_test == 1) & (y_pred == 0))[0]

print(f"FP: {len(fp_indices)}, FN: {len(fn_indices)}")

fp_idx = fp_indices[np.argmin(np.abs(y_pred_proba[fp_indices] - 0.5))]
fn_idx = fn_indices[np.argmax(y_pred_proba[fn_indices])]

print(f"FP idx={fp_idx}, prob={y_pred_proba[fp_idx]:.4f}")
print(f"FN idx={fn_idx}, prob={y_pred_proba[fn_idx]:.4f}")

explainer = shap.TreeExplainer(model)

print("Computing FP SHAP...")
fp_vals = explainer.shap_values(X_test.iloc[[fp_idx]])
if isinstance(fp_vals, list):
    fp_vals = fp_vals[1]
exp_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

shap.save_html("../reports/images/shap_force_fp.html", 
               shap.force_plot(exp_val, fp_vals[0], X_test.iloc[fp_idx], matplotlib=False, show=False))
print("Saved shap_force_fp.html")

plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap.Explanation(values=fp_vals[0], base_values=exp_val, 
                                     data=X_test.iloc[fp_idx].values, feature_names=list(X_test.columns)), show=False)
plt.tight_layout()
plt.savefig("../reports/images/shap_waterfall_fp.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved shap_waterfall_fp.png")

print("Computing FN SHAP...")
fn_vals = explainer.shap_values(X_test.iloc[[fn_idx]])
if isinstance(fn_vals, list):
    fn_vals = fn_vals[1]

shap.save_html("../reports/images/shap_force_fn.html",
               shap.force_plot(exp_val, fn_vals[0], X_test.iloc[fn_idx], matplotlib=False, show=False))
print("Saved shap_force_fn.html")

plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap.Explanation(values=fn_vals[0], base_values=exp_val,
                                     data=X_test.iloc[fn_idx].values, feature_names=list(X_test.columns)), show=False)
plt.tight_layout()
plt.savefig("../reports/images/shap_waterfall_fn.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved shap_waterfall_fn.png")

print("\n✓ All plots generated!")
