"""Exploratory Data Analysis module for e-commerce fraud data.

This module contains classes for performing comprehensive exploratory data
analysis on the fraud transaction dataset, including univariate analysis,
bivariate analysis, class imbalance analysis, and temporal pattern analysis.

Classes:
    FraudDataEDA: Comprehensive EDA for e-commerce fraud data.
"""

from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


class FraudDataEDA:
    """Perform comprehensive EDA on e-commerce fraud transaction data.

    This class provides methods to analyze fraud patterns, explore feature
    distributions, identify relationships between features and fraud labels,
    and visualize insights from the data.

    Attributes:
        data (pd.DataFrame): The fraud transaction dataset.
        target_column (str): Name of the target variable column.
        output_dir (Path): Directory for saving visualizations.
        report (dict): Dictionary storing analysis results.

    Example:
        >>> eda = FraudDataEDA(fraud_data, target_column='class',
        ...                    output_dir='reports/images')
        >>> eda.univariate_analysis()
        >>> eda.bivariate_analysis()
        >>> report = eda.generate_eda_report()
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_column: str = "class",
        output_dir: str = "reports/images",
    ):
        """Initialize the FraudDataEDA analyzer.

        Args:
            data (pd.DataFrame): The fraud transaction dataset.
            target_column (str): Name of the target variable column.
                Defaults to 'class'.
            output_dir (str): Directory path for saving visualizations.
                Defaults to 'reports/images'.
        """
        self.data = data.copy()
        self.target_column = target_column
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report = {}

        print(f"✓ FraudDataEDA initialized")
        print(f"  Dataset shape: {self.data.shape}")
        print(f"  Target column: {self.target_column}")
        print(f"  Output directory: {self.output_dir}")

    def univariate_analysis(
        self,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        save_plots: bool = True,
    ) -> Dict[str, Any]:
        """Analyze individual features (univariate analysis).

        Creates distribution plots for numerical features and count plots
        for categorical features. Provides summary statistics for each feature.

        Args:
            numerical_features (list, optional): List of numerical feature names.
                If None, auto-detects numerical columns.
            categorical_features (list, optional): List of categorical feature names.
                If None, auto-detects categorical columns.
            save_plots (bool): Whether to save plots to output directory.
                Defaults to True.

        Returns:
            dict: Dictionary containing:
                - numerical_stats: Statistics for numerical features
                - categorical_stats: Statistics for categorical features
                - plots_saved: List of saved plot filenames

        Example:
            >>> eda = FraudDataEDA(fraud_data)
            >>> results = eda.univariate_analysis()
            >>> print(results['numerical_stats'])
        """
        results = {"numerical_stats": {}, "categorical_stats": {}, "plots_saved": []}

        # Auto-detect feature types if not provided
        if numerical_features is None:
            numerical_features = self.data.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            # Remove target column
            if self.target_column in numerical_features:
                numerical_features.remove(self.target_column)

        if categorical_features is None:
            categorical_features = self.data.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

        print(f"\n{'='*60}")
        print("UNIVARIATE ANALYSIS")
        print(f"{'='*60}")

        # Analyze numerical features
        if numerical_features:
            print(f"\nAnalyzing {len(numerical_features)} numerical features...")

            for feature in numerical_features:
                stats = {
                    "mean": self.data[feature].mean(),
                    "median": self.data[feature].median(),
                    "std": self.data[feature].std(),
                    "min": self.data[feature].min(),
                    "max": self.data[feature].max(),
                    "skewness": self.data[feature].skew(),
                    "kurtosis": self.data[feature].kurtosis(),
                }
                results["numerical_stats"][feature] = stats

                print(f"\n  {feature}:")
                print(f"    Mean: {stats['mean']:.2f}")
                print(f"    Median: {stats['median']:.2f}")
                print(f"    Std: {stats['std']:.2f}")
                print(f"    Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
                print(f"    Skewness: {stats['skewness']:.2f}")

            # Create distribution plots
            if save_plots:
                n_features = len(numerical_features)
                n_cols = min(3, n_features)
                n_rows = (n_features + n_cols - 1) // n_cols

                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
                if n_features == 1:
                    axes = [axes]
                else:
                    axes = axes.flatten()

                for idx, feature in enumerate(numerical_features):
                    ax = axes[idx]
                    self.data[feature].hist(bins=50, ax=ax, edgecolor="black")
                    ax.set_title(f"Distribution of {feature}", fontweight="bold")
                    ax.set_xlabel(feature)
                    ax.set_ylabel("Frequency")
                    ax.grid(alpha=0.3)

                # Hide unused subplots
                for idx in range(len(numerical_features), len(axes)):
                    axes[idx].set_visible(False)

                plt.tight_layout()
                filename = "univariate_numerical.png"
                filepath = self.output_dir / filename
                plt.savefig(filepath, dpi=300, bbox_inches="tight")
                plt.close()
                results["plots_saved"].append(str(filepath))
                print(f"\n✓ Saved numerical features plot: {filepath}")

        # Analyze categorical features
        if categorical_features:
            print(f"\nAnalyzing {len(categorical_features)} categorical features...")

            for feature in categorical_features:
                value_counts = self.data[feature].value_counts()
                stats = {
                    "unique_count": self.data[feature].nunique(),
                    "top_value": value_counts.index[0],
                    "top_value_count": value_counts.values[0],
                    "top_value_percentage": (value_counts.values[0] / len(self.data))
                    * 100,
                }
                results["categorical_stats"][feature] = stats

                print(f"\n  {feature}:")
                print(f"    Unique values: {stats['unique_count']}")
                print(
                    f"    Most common: {stats['top_value']} ({stats['top_value_count']} occurrences, "
                    f"{stats['top_value_percentage']:.1f}%)"
                )

            # Create count plots
            if save_plots:
                n_features = len(categorical_features)
                n_cols = min(2, n_features)
                n_rows = (n_features + n_cols - 1) // n_cols

                fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
                if n_features == 1:
                    axes = [axes]
                else:
                    axes = axes.flatten()

                for idx, feature in enumerate(categorical_features):
                    ax = axes[idx]
                    value_counts = self.data[feature].value_counts().head(10)
                    value_counts.plot(kind="bar", ax=ax, color="steelblue")
                    ax.set_title(f"Distribution of {feature}", fontweight="bold")
                    ax.set_xlabel(feature)
                    ax.set_ylabel("Count")
                    ax.tick_params(axis="x", rotation=45)
                    ax.grid(axis="y", alpha=0.3)

                # Hide unused subplots
                for idx in range(len(categorical_features), len(axes)):
                    axes[idx].set_visible(False)

                plt.tight_layout()
                filename = "univariate_categorical.png"
                filepath = self.output_dir / filename
                plt.savefig(filepath, dpi=300, bbox_inches="tight")
                plt.close()
                results["plots_saved"].append(str(filepath))
                print(f"\n✓ Saved categorical features plot: {filepath}")

        self.report["univariate_analysis"] = results
        print(f"\n{'='*60}\n")
        return results

    def bivariate_analysis(
        self, features: Optional[List[str]] = None, save_plots: bool = True
    ) -> Dict[str, Any]:
        """Analyze relationships between features and target variable.

        Examines how each feature relates to the fraud label using
        appropriate visualizations (box plots for numerical, count plots
        for categorical).

        Args:
            features (list, optional): List of features to analyze.
                If None, analyzes all features.
            save_plots (bool): Whether to save plots. Defaults to True.

        Returns:
            dict: Dictionary containing:
                - fraud_rates: Fraud rates for different feature segments
                - correlations: Correlations with target variable
                - plots_saved: List of saved plot filenames

        Example:
            >>> eda = FraudDataEDA(fraud_data)
            >>> results = eda.bivariate_analysis()
            >>> print(results['fraud_rates'])
        """
        results = {"fraud_rates": {}, "correlations": {}, "plots_saved": []}

        if features is None:
            features = [col for col in self.data.columns if col != self.target_column]

        print(f"\n{'='*60}")
        print("BIVARIATE ANALYSIS")
        print(f"{'='*60}")

        # Separate numerical and categorical features
        numerical_features = [
            f
            for f in features
            if f in self.data.select_dtypes(include=[np.number]).columns
        ]
        categorical_features = [
            f
            for f in features
            if f in self.data.select_dtypes(include=["object", "category"]).columns
        ]

        # Analyze numerical features
        if numerical_features:
            print(
                f"\nAnalyzing {len(numerical_features)} numerical features vs fraud..."
            )

            # Calculate correlations
            for feature in numerical_features:
                corr = self.data[feature].corr(self.data[self.target_column])
                results["correlations"][feature] = corr
                print(f"  {feature}: correlation = {corr:.4f}")

            # Create box plots
            if save_plots and len(numerical_features) > 0:
                n_features = min(
                    6, len(numerical_features)
                )  # Limit to 6 for readability
                n_cols = 3
                n_rows = (n_features + n_cols - 1) // n_cols

                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
                axes = axes.flatten() if n_features > 1 else [axes]

                for idx, feature in enumerate(numerical_features[:n_features]):
                    ax = axes[idx]
                    self.data.boxplot(column=feature, by=self.target_column, ax=ax)
                    ax.set_title(f"{feature} by Fraud Status")
                    ax.set_xlabel("Fraud (0=No, 1=Yes)")
                    ax.set_ylabel(feature)
                    plt.sca(ax)
                    plt.xticks([1, 2], ["No", "Yes"])

                # Hide unused subplots
                for idx in range(n_features, len(axes)):
                    axes[idx].set_visible(False)

                plt.suptitle("")  # Remove default title
                plt.tight_layout()
                filename = "bivariate_numerical.png"
                filepath = self.output_dir / filename
                plt.savefig(filepath, dpi=300, bbox_inches="tight")
                plt.close()
                results["plots_saved"].append(str(filepath))
                print(f"\n✓ Saved numerical bivariate plot: {filepath}")

        # Analyze categorical features
        if categorical_features:
            print(
                f"\nAnalyzing {len(categorical_features)} categorical features vs fraud..."
            )

            for feature in categorical_features:
                fraud_by_category = (
                    self.data.groupby(feature)[self.target_column]
                    .agg(["sum", "count", "mean"])
                    .rename(
                        columns={
                            "sum": "fraud_count",
                            "count": "total",
                            "mean": "fraud_rate",
                        }
                    )
                )
                fraud_by_category = fraud_by_category.sort_values(
                    "fraud_rate", ascending=False
                )

                results["fraud_rates"][feature] = fraud_by_category.to_dict()

                print(f"\n  {feature} - Top 5 fraud rates:")
                for idx, (cat, row) in enumerate(fraud_by_category.head().iterrows()):
                    print(
                        f"    {cat}: {row['fraud_rate']*100:.2f}% "
                        f"({int(row['fraud_count'])}/{int(row['total'])})"
                    )

            # Create stacked bar plots
            if save_plots and len(categorical_features) > 0:
                n_features = min(4, len(categorical_features))
                n_cols = 2
                n_rows = (n_features + n_cols - 1) // n_cols

                fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
                axes = axes.flatten() if n_features > 1 else [axes]

                for idx, feature in enumerate(categorical_features[:n_features]):
                    ax = axes[idx]

                    # Get top categories by frequency
                    top_categories = self.data[feature].value_counts().head(10).index
                    data_subset = self.data[self.data[feature].isin(top_categories)]

                    fraud_counts = pd.crosstab(
                        data_subset[feature], data_subset[self.target_column]
                    )
                    fraud_counts.plot(
                        kind="bar", stacked=True, ax=ax, color=["#2ecc71", "#e74c3c"]
                    )
                    ax.set_title(f"{feature} vs Fraud Status", fontweight="bold")
                    ax.set_xlabel(feature)
                    ax.set_ylabel("Count")
                    ax.legend(["Legitimate", "Fraud"], loc="upper right")
                    ax.tick_params(axis="x", rotation=45)
                    ax.grid(axis="y", alpha=0.3)

                # Hide unused subplots
                for idx in range(n_features, len(axes)):
                    axes[idx].set_visible(False)

                plt.tight_layout()
                filename = "bivariate_categorical.png"
                filepath = self.output_dir / filename
                plt.savefig(filepath, dpi=300, bbox_inches="tight")
                plt.close()
                results["plots_saved"].append(str(filepath))
                print(f"\n✓ Saved categorical bivariate plot: {filepath}")

        self.report["bivariate_analysis"] = results
        print(f"\n{'='*60}\n")
        return results

    def analyze_class_imbalance(self, save_plot: bool = True) -> Dict[str, Any]:
        """Analyze and visualize class imbalance in the dataset.

        Calculates fraud vs legitimate transaction ratios and creates
        visualization of class distribution.

        Args:
            save_plot (bool): Whether to save the plot. Defaults to True.

        Returns:
            dict: Dictionary containing:
                - fraud_count: Number of fraud cases
                - legitimate_count: Number of legitimate cases
                - fraud_percentage: Percentage of fraud cases
                - imbalance_ratio: Ratio of legitimate to fraud cases
                - plot_saved: Path to saved plot (if save_plot=True)

        Example:
            >>> eda = FraudDataEDA(fraud_data)
            >>> imbalance_info = eda.analyze_class_imbalance()
            >>> print(f"Imbalance ratio: {imbalance_info['imbalance_ratio']:.2f}:1")
        """
        print(f"\n{'='*60}")
        print("CLASS IMBALANCE ANALYSIS")
        print(f"{'='*60}\n")

        class_counts = self.data[self.target_column].value_counts()
        legitimate_count = class_counts[0]
        fraud_count = class_counts[1] if 1 in class_counts else 0

        total = len(self.data)
        fraud_percentage = (fraud_count / total) * 100
        legitimate_percentage = (legitimate_count / total) * 100
        imbalance_ratio = (
            legitimate_count / fraud_count if fraud_count > 0 else float("inf")
        )

        results = {
            "fraud_count": int(fraud_count),
            "legitimate_count": int(legitimate_count),
            "fraud_percentage": fraud_percentage,
            "legitimate_percentage": legitimate_percentage,
            "imbalance_ratio": imbalance_ratio,
            "total_transactions": total,
        }

        print(f"Total transactions: {total:,}")
        print(f"\nClass Distribution:")
        print(f"  Legitimate (0): {legitimate_count:,} ({legitimate_percentage:.2f}%)")
        print(f"  Fraud (1):      {fraud_count:,} ({fraud_percentage:.2f}%)")
        print(f"\nImbalance Ratio: {imbalance_ratio:.2f}:1 (Legitimate:Fraud)")

        # Create visualization
        if save_plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Bar plot
            class_counts.plot(kind="bar", ax=ax1, color=["#2ecc71", "#e74c3c"])
            ax1.set_title(
                "Fraud Class Distribution (Count)", fontsize=14, fontweight="bold"
            )
            ax1.set_xlabel("Class (0=Legitimate, 1=Fraud)")
            ax1.set_ylabel("Count")
            ax1.set_xticklabels(["Legitimate", "Fraud"], rotation=0)
            ax1.grid(axis="y", alpha=0.3)

            # Add count labels
            for container in ax1.containers:
                ax1.bar_label(container, fmt="%d")

            # Pie chart
            ax2.pie(
                [legitimate_count, fraud_count],
                labels=["Legitimate", "Fraud"],
                autopct="%1.2f%%",
                startangle=90,
                colors=["#2ecc71", "#e74c3c"],
                explode=(0, 0.1),
            )
            ax2.set_title(
                "Fraud Class Distribution (Percentage)", fontsize=14, fontweight="bold"
            )

            plt.tight_layout()
            filename = "class_imbalance.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close()
            results["plot_saved"] = str(filepath)
            print(f"\n✓ Saved class imbalance plot: {filepath}")

        self.report["class_imbalance"] = results
        print(f"\n{'='*60}\n")
        return results

    def temporal_analysis(
        self, time_column: str = "purchase_time", save_plot: bool = True
    ) -> Dict[str, Any]:
        """Analyze temporal patterns in fraud transactions.

        Examines fraud rates over time, identifying patterns by hour,
        day of week, or other temporal segments.

        Args:
            time_column (str): Name of the timestamp column.
                Defaults to 'purchase_time'.
            save_plot (bool): Whether to save plots. Defaults to True.

        Returns:
            dict: Dictionary containing:
                - hourly_fraud_rate: Fraud rate by hour of day
                - daily_fraud_rate: Fraud rate by day (if applicable)
                - plots_saved: List of saved plot filenames

        Example:
            >>> eda = FraudDataEDA(fraud_data)
            >>> temporal_info = eda.temporal_analysis('purchase_time')
        """
        print(f"\n{'='*60}")
        print("TEMPORAL ANALYSIS")
        print(f"{'='*60}\n")

        results = {"plots_saved": []}

        if time_column not in self.data.columns:
            print(f"⚠ Time column '{time_column}' not found in dataset")
            return results

        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(self.data[time_column]):
            self.data[time_column] = pd.to_datetime(self.data[time_column])

        # Extract temporal features
        self.data["hour"] = self.data[time_column].dt.hour
        self.data["day_of_week"] = self.data[time_column].dt.dayofweek
        self.data["day_name"] = self.data[time_column].dt.day_name()

        # Analyze by hour
        hourly_stats = (
            self.data.groupby("hour")[self.target_column]
            .agg(["sum", "count", "mean"])
            .rename(
                columns={"sum": "fraud_count", "count": "total", "mean": "fraud_rate"}
            )
        )

        results["hourly_fraud_rate"] = hourly_stats.to_dict()

        print("Fraud Rate by Hour of Day:")
        print("  Hour  Fraud Rate  Total Transactions")
        for hour, row in hourly_stats.iterrows():
            print(
                f"  {hour:02d}:00  {row['fraud_rate']*100:6.2f}%   {int(row['total']):8,}"
            )

        # Analyze by day of week
        daily_stats = (
            self.data.groupby("day_name")[self.target_column]
            .agg(["sum", "count", "mean"])
            .rename(
                columns={"sum": "fraud_count", "count": "total", "mean": "fraud_rate"}
            )
        )

        # Reorder days
        day_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        daily_stats = daily_stats.reindex(
            [d for d in day_order if d in daily_stats.index]
        )

        results["daily_fraud_rate"] = daily_stats.to_dict()

        print("\nFraud Rate by Day of Week:")
        print("  Day        Fraud Rate  Total Transactions")
        for day, row in daily_stats.iterrows():
            print(f"  {day:9s}  {row['fraud_rate']*100:6.2f}%   {int(row['total']):8,}")

        # Create visualizations
        if save_plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # Hourly pattern
            hourly_stats["fraud_rate"].plot(
                kind="line", ax=ax1, marker="o", color="steelblue", linewidth=2
            )
            ax1.set_title("Fraud Rate by Hour of Day", fontsize=14, fontweight="bold")
            ax1.set_xlabel("Hour")
            ax1.set_ylabel("Fraud Rate")
            ax1.set_xticks(range(0, 24, 2))
            ax1.grid(True, alpha=0.3)
            ax1.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: f"{y*100:.1f}%")
            )

            # Daily pattern
            daily_stats["fraud_rate"].plot(kind="bar", ax=ax2, color="coral")
            ax2.set_title("Fraud Rate by Day of Week", fontsize=14, fontweight="bold")
            ax2.set_xlabel("Day")
            ax2.set_ylabel("Fraud Rate")
            ax2.tick_params(axis="x", rotation=45)
            ax2.grid(axis="y", alpha=0.3)
            ax2.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: f"{y*100:.1f}%")
            )

            plt.tight_layout()
            filename = "temporal_patterns.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close()
            results["plots_saved"].append(str(filepath))
            print(f"\n✓ Saved temporal patterns plot: {filepath}")

        self.report["temporal_analysis"] = results
        print(f"\n{'='*60}\n")
        return results

    def categorical_analysis(
        self, features: Optional[List[str]] = None, save_plot: bool = True
    ) -> Dict[str, Any]:
        """Detailed analysis of categorical features.

        Analyzes distribution and fraud rates for categorical features
        like browser, source, and sex.

        Args:
            features (list, optional): List of categorical features to analyze.
                If None, analyzes all categorical features.
            save_plot (bool): Whether to save plots. Defaults to True.

        Returns:
            dict: Dictionary containing statistics for each categorical feature.

        Example:
            >>> eda = FraudDataEDA(fraud_data)
            >>> cat_stats = eda.categorical_analysis(['browser', 'source'])
        """
        print(f"\n{'='*60}")
        print("CATEGORICAL FEATURES ANALYSIS")
        print(f"{'='*60}\n")

        results = {}

        if features is None:
            features = self.data.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

        for feature in features:
            if feature not in self.data.columns:
                print(f"⚠ Feature '{feature}' not found in dataset")
                continue

            print(f"\nAnalyzing {feature}:")
            print(f"  Unique values: {self.data[feature].nunique()}")

            # Calculate fraud rates by category
            fraud_by_cat = (
                self.data.groupby(feature)[self.target_column]
                .agg(["sum", "count", "mean"])
                .rename(
                    columns={
                        "sum": "fraud_count",
                        "count": "total",
                        "mean": "fraud_rate",
                    }
                )
            )
            fraud_by_cat = fraud_by_cat.sort_values("fraud_rate", ascending=False)

            results[feature] = fraud_by_cat.to_dict()

            print(f"\n  Top 5 categories by fraud rate:")
            for cat, row in fraud_by_cat.head().iterrows():
                print(
                    f"    {cat}: {row['fraud_rate']*100:.2f}% "
                    f"({int(row['fraud_count'])}/{int(row['total'])} transactions)"
                )

        self.report["categorical_analysis"] = results
        print(f"\n{'='*60}\n")

        if save_plot:
            n_features = min(4, len(features))
            n_cols = 2
            n_rows = (n_features + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
            axes = axes.flatten() if n_features > 1 else [axes]

            for idx, feature in enumerate(features[:n_features]):
                ax = axes[idx]

                # Get top categories by frequency
                top_categories = self.data[feature].value_counts().head(10).index
                data_subset = self.data[self.data[feature].isin(top_categories)]

                fraud_counts = pd.crosstab(
                    data_subset[feature], data_subset[self.target_column]
                )
                fraud_counts.plot(
                    kind="bar", stacked=True, ax=ax, color=["#2ecc71", "#e74c3c"]
                )
                ax.set_title(f"{feature} vs Fraud Status", fontweight="bold")
                ax.set_xlabel(feature)
                ax.set_ylabel("Count")
                ax.legend(["Legitimate", "Fraud"], loc="upper right")
                ax.tick_params(axis="x", rotation=45)
                ax.grid(axis="y", alpha=0.3)

            # Hide unused subplots
            for idx in range(n_features, len(axes)):
                axes[idx].set_visible(False)

            plt.tight_layout()
            filename = "categorical_features_analysis.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"\n✓ Saved categorical features analysis plot: {filepath}")
            results["plot_saved"] = str(filepath)
        self.report["categorical_analysis"] = results
        print(f"\n{'='*60}\n")
        return results

    def generate_eda_report(self) -> Dict[str, Any]:
        """Generate comprehensive EDA report.

        Compiles all analysis results into a single comprehensive report
        with summary statistics and key insights.

        Returns:
            dict: Complete EDA report with all analysis results.

        Example:
            >>> eda = FraudDataEDA(fraud_data)
            >>> eda.univariate_analysis()
            >>> eda.bivariate_analysis()
            >>> eda.analyze_class_imbalance()
            >>> report = eda.generate_eda_report()
        """
        print(f"\n{'='*60}")
        print("EDA REPORT SUMMARY")
        print(f"{'='*60}\n")

        print(f"Dataset: {self.data.shape[0]:,} rows × {self.data.shape[1]} columns")
        print(f"Target column: {self.target_column}")
        print(f"Output directory: {self.output_dir}")

        print(f"\nAnalyses completed:")
        for analysis_name in self.report.keys():
            print(f"  ✓ {analysis_name.replace('_', ' ').title()}")

        if "class_imbalance" in self.report:
            imb = self.report["class_imbalance"]
            print(f"\nClass Imbalance:")
            print(f"  Fraud: {imb['fraud_count']:,} ({imb['fraud_percentage']:.2f}%)")
            print(
                f"  Legitimate: {imb['legitimate_count']:,} ({imb['legitimate_percentage']:.2f}%)"
            )
            print(f"  Imbalance Ratio: {imb['imbalance_ratio']:.2f}:1")

        # Count total plots saved
        total_plots = sum(
            len(analysis.get("plots_saved", []))
            for analysis in self.report.values()
            if isinstance(analysis, dict)
        )
        print(f"\nVisualizations created: {total_plots}")

        print(f"\n{'='*60}\n")

        return self.report.copy()
