"""Exploratory Data Analysis module for credit card transaction data.

This module provides comprehensive EDA utilities for the `creditcard.csv`
dataset including PCA features analysis, amount/time patterns, correlation
analysis, and class distribution visualization. It follows the same OOP
architecture and documentation standards used across the project.

Classes:
    CreditCardEDA: Comprehensive EDA for bank credit card transactions.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


class CreditCardEDA:
    """Perform EDA on credit card transaction dataset.

    This class analyzes anonymized PCA features (V1-V28), transaction `Amount`,
    temporal `Time` patterns, correlation structure, and class imbalance.

    Attributes:
        data (pd.DataFrame): The credit card dataset.
        target_column (str): Target column name (defaults to 'Class').
        output_dir (Path): Directory to save visualizations.
        report (dict): Stores results from analyses.

    Example:
        >>> eda = CreditCardEDA(cc_data, target_column='Class', output_dir='reports/images')
        >>> eda.pca_features_analysis()
        >>> eda.amount_analysis()
        >>> eda.time_analysis()
        >>> eda.correlation_analysis()
        >>> eda.analyze_class_imbalance()
        >>> summary = eda.generate_eda_report()
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_column: str = "Class",
        output_dir: str = "reports/images",
    ) -> None:
        """Initialize the EDA analyzer.

        Args:
            data (pd.DataFrame): Credit card dataset.
            target_column (str): Target column name. Defaults to 'Class'.
            output_dir (str): Directory to save plots. Defaults to 'reports/images'.
        """
        self.data = data.copy()
        self.target_column = target_column
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report: Dict[str, Any] = {}

        print("✓ CreditCardEDA initialized")
        print(f"  Dataset shape: {self.data.shape}")
        print(f"  Target column: {self.target_column}")
        print(f"  Output directory: {self.output_dir}")

    def _get_pca_feature_names(self) -> List[str]:
        """Detect PCA feature columns V1-V28 present in the dataset.

        Returns:
            List[str]: List of PCA feature column names.
        """
        return [c for c in self.data.columns if c.startswith("V") and c[1:].isdigit()]

    def pca_features_analysis(self, save_plots: bool = True) -> Dict[str, Any]:
        """Analyze distribution of PCA features V1-V28.

        Creates histograms for PCA components and basic summary statistics.

        Args:
            save_plots (bool): Whether to save generated plots.

        Returns:
            dict: Statistics and saved plot paths.
        """
        features = self._get_pca_feature_names()
        results: Dict[str, Any] = {"stats": {}, "plots_saved": []}

        print("\n" + "=" * 60)
        print("PCA FEATURES ANALYSIS")
        print("=" * 60)

        for f in features:
            s = self.data[f]
            results["stats"][f] = {
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "max": float(s.max()),
                "skew": float(s.skew()),
                "kurtosis": float(s.kurtosis()),
            }

        print(f"Analyzed {len(features)} PCA features")

        if save_plots and features:
            n = len(features)
            n_cols = 4
            n_rows = int(np.ceil(n / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows))
            axes = axes.flatten()
            for i, f in enumerate(features):
                ax = axes[i]
                self.data[f].hist(bins=40, ax=ax, color="steelblue", edgecolor="black")
                ax.set_title(f"{f}")
                ax.grid(alpha=0.3)
            for i in range(len(features), len(axes)):
                axes[i].set_visible(False)
            plt.tight_layout()
            path = self.output_dir / "cc_pca_distributions.png"
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()
            results["plots_saved"].append(str(path))
            print(f"✓ Saved PCA distributions: {path}")

        self.report["pca_features_analysis"] = results
        return results

    def amount_analysis(self, save_plots: bool = True) -> Dict[str, Any]:
        """Analyze transaction amount patterns and relation to fraud.

        Args:
            save_plots (bool): Whether to save plots.

        Returns:
            dict: Summary stats and plot paths.
        """
        results: Dict[str, Any] = {"stats": {}, "plots_saved": []}
        if "Amount" not in self.data.columns:
            print("⚠ 'Amount' column not found")
            self.report["amount_analysis"] = results
            return results

        s = self.data["Amount"]
        results["stats"] = {
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
        }

        if save_plots:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            s.hist(bins=50, ax=axes[0], color="#4e79a7", edgecolor="black")
            axes[0].set_title("Transaction Amount Distribution")
            axes[0].set_xlabel("Amount")
            axes[0].set_ylabel("Frequency")
            axes[0].grid(alpha=0.3)

            if self.target_column in self.data.columns:
                # Box plot amount vs fraud
                self.data.boxplot(column="Amount", by=self.target_column, ax=axes[1])
                axes[1].set_title("Amount by Fraud Status")
                axes[1].set_xlabel("Fraud (0=No, 1=Yes)")
                axes[1].set_ylabel("Amount")
                plt.suptitle("")

            plt.tight_layout()
            path = self.output_dir / "cc_amount_analysis.png"
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()
            results["plots_saved"].append(str(path))
            print(f"✓ Saved amount analysis plot: {path}")

        self.report["amount_analysis"] = results
        return results

    def time_analysis(self, save_plots: bool = True) -> Dict[str, Any]:
        """Analyze temporal patterns based on the `Time` column.

        `Time` is seconds elapsed since the first transaction. We derive hour
        and day buckets to explore patterns.

        Args:
            save_plots (bool): Whether to save plots.

        Returns:
            dict: Temporal stats and plot paths.
        """
        results: Dict[str, Any] = {"plots_saved": []}
        if "Time" not in self.data.columns:
            print("⚠ 'Time' column not found")
            self.report["time_analysis"] = results
            return results

        df = self.data.copy()
        df["hours"] = (df["Time"] / 3600).astype(int)
        hourly = (
            df.groupby("hours")[self.target_column]
            .agg(["sum", "count", "mean"])
            .rename(
                columns={"sum": "fraud_count", "count": "total", "mean": "fraud_rate"}
            )
        )
        results["hourly"] = hourly.to_dict()

        if save_plots:
            fig, ax = plt.subplots(figsize=(12, 4))
            hourly["fraud_rate"].plot(ax=ax, color="#f28e2b", marker="o")
            ax.set_title("Fraud Rate by Hour (relative)")
            ax.set_xlabel("Hours since first transaction")
            ax.set_ylabel("Fraud Rate")
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: f"{y*100:.1f}%")
            )
            path = self.output_dir / "cc_time_patterns.png"
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()
            results["plots_saved"].append(str(path))
            print(f"✓ Saved time analysis plot: {path}")

        self.report["time_analysis"] = results
        return results

    def correlation_analysis(self, save_plots: bool = True) -> Dict[str, Any]:
        """Compute correlations among numerical features and with target.

        Args:
            save_plots (bool): Whether to save heatmap.

        Returns:
            dict: Correlation matrices and plot paths.
        """
        num_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        corr = self.data[num_cols].corr()
        results = {"corr_matrix": corr, "plots_saved": []}

        if save_plots:
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr, cmap="coolwarm", center=0)
            plt.title("Correlation Heatmap (Credit Card)")
            path = self.output_dir / "cc_correlation_heatmap.png"
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()
            results["plots_saved"].append(str(path))
            print(f"✓ Saved correlation heatmap: {path}")

        self.report["correlation_analysis"] = results
        return results

    def analyze_class_imbalance(self, save_plot: bool = True) -> Dict[str, Any]:
        """Analyze and visualize class distribution for the dataset.

        Args:
            save_plot (bool): Whether to save charts.

        Returns:
            dict: Counts, percentages, ratio, and plot path.
        """
        counts = self.data[self.target_column].value_counts()
        legit = int(counts.get(0, 0))
        fraud = int(counts.get(1, 0))
        total = legit + fraud
        fraud_pct = (fraud / total * 100) if total else 0.0
        legit_pct = (legit / total * 100) if total else 0.0
        ratio = (legit / fraud) if fraud else float("inf")

        results = {
            "fraud_count": fraud,
            "legitimate_count": legit,
            "fraud_percentage": fraud_pct,
            "legitimate_percentage": legit_pct,
            "imbalance_ratio": ratio,
            "total_transactions": total,
        }

        if save_plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            pd.Series({"Legitimate": legit, "Fraud": fraud}).plot(
                kind="bar", ax=ax1, color=["#2ecc71", "#e74c3c"]
            )
            ax1.set_title("Class Distribution (Count)")
            ax1.grid(axis="y", alpha=0.3)
            for c in ax1.containers:
                ax1.bar_label(c, fmt="%d")

            ax2.pie(
                [legit, fraud],
                labels=["Legitimate", "Fraud"],
                autopct="%1.2f%%",
                startangle=90,
                colors=["#2ecc71", "#e74c3c"],
                explode=(0, 0.1),
            )
            ax2.set_title("Class Distribution (Percentage)")

            plt.tight_layout()
            path = self.output_dir / "cc_class_imbalance.png"
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()
            results["plot_saved"] = str(path)
            print(f"✓ Saved class imbalance plot: {path}")

        self.report["class_imbalance"] = results
        return results

    def generate_eda_report(self) -> Dict[str, Any]:
        """Generate a summary report for all executed analyses."""
        print("\n" + "=" * 60)
        print("CREDIT CARD EDA REPORT SUMMARY")
        print("=" * 60)
        print(f"Dataset: {self.data.shape[0]:,} rows × {self.data.shape[1]} columns")
        print(f"Target column: {self.target_column}")
        print(f"Output directory: {self.output_dir}")

        for k in self.report.keys():
            print(f"  ✓ {k.replace('_', ' ').title()}")

        total_plots = sum(
            len(v.get("plots_saved", []))
            for v in self.report.values()
            if isinstance(v, dict)
        )
        print(f"\nVisualizations created: {total_plots}")
        print("=" * 60 + "\n")
        return self.report.copy()
