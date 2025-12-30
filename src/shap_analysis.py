"""Module for model explainability analysis using SHAP.

This module contains classes for analyzing and explaining machine learning
model predictions using SHAP (SHapley Additive exPlanations) values and
built-in feature importance methods.

Classes:
    ExplainabilityAnalyzer: Extract and visualize feature importance
    RecommendationGenerator: Generate business recommendations from insights
"""

from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore")


class ExplainabilityAnalyzer:
    """Analyzer for model explainability using SHAP and feature importance.

    This class provides methods to extract built-in feature importance,
    calculate SHAP values, and create visualizations for both global
    and local model interpretability.

    Attributes:
        model: Trained machine learning model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        feature_names (List[str]): Names of features
        shap_values: Calculated SHAP values
        explainer: SHAP explainer object
    """

    def __init__(self, model, X_test: pd.DataFrame, y_test: pd.Series):
        """Initialize the ExplainabilityAnalyzer.

        Args:
            model: Trained sklearn/lightgbm/xgboost model
            X_test (pd.DataFrame): Test features for analysis
            y_test (pd.Series): Test labels for analysis
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = list(X_test.columns)
        self.shap_values = None
        self.explainer = None

    def extract_feature_importance(self) -> pd.DataFrame:
        """Extract built-in feature importance from the model.

        Returns:
            pd.DataFrame: Feature importance with columns ['feature', 'importance']

        Raises:
            AttributeError: If model doesn't have feature_importances_ attribute

        Example:
            >>> analyzer = ExplainabilityAnalyzer(model, X_test, y_test)
            >>> importance_df = analyzer.extract_feature_importance()
            >>> print(importance_df.head())
        """
        if not hasattr(self.model, "feature_importances_"):
            raise AttributeError("Model does not have feature_importances_ attribute")

        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": self.model.feature_importances_,
            }
        )

        # Sort by importance descending
        importance_df = importance_df.sort_values(
            "importance", ascending=False
        ).reset_index(drop=True)

        return importance_df

    def plot_feature_importance(
        self, top_n: int = 10, save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot top N features by built-in importance.

        Args:
            top_n (int): Number of top features to display (default: 10)
            save_path (str, optional): Path to save the plot

        Returns:
            plt.Figure: Matplotlib figure object

        Example:
            >>> fig = analyzer.plot_feature_importance(top_n=10,
            ...                                       save_path='reports/images/feature_importance.png')
        """
        importance_df = self.extract_feature_importance()
        top_features = importance_df.head(top_n)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Create horizontal bar plot
        bars = ax.barh(range(len(top_features)), top_features["importance"])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features["feature"])
        ax.set_xlabel("Feature Importance", fontsize=12)
        ax.set_ylabel("Features", fontsize=12)
        ax.set_title(
            f"Top {top_n} Features by Built-in Importance",
            fontsize=14,
            fontweight="bold",
        )

        # Color bars by value
        norm = plt.Normalize(
            vmin=top_features["importance"].min(), vmax=top_features["importance"].max()
        )
        colors = plt.cm.viridis(norm(top_features["importance"]))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        # Invert y-axis to show most important at top
        ax.invert_yaxis()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✓ Feature importance plot saved to {save_path}")

        return fig

    def compare_importance_methods(self) -> pd.DataFrame:
        """Compare different feature importance metrics from the model.

        For tree-based models, this compares gain-based and split-based importance.

        Returns:
            pd.DataFrame: Comparison of importance metrics

        Example:
            >>> comparison_df = analyzer.compare_importance_methods()
        """
        importance_df = self.extract_feature_importance()

        # For LightGBM and XGBoost, we can get different importance types
        if isinstance(self.model, (LGBMClassifier, XGBClassifier)):
            try:
                # Get gain-based importance (default)
                gain_importance = self.model.feature_importances_

                # For LightGBM, try to get split-based importance
                if isinstance(self.model, LGBMClassifier):
                    split_importance = self.model.booster_.feature_importance(
                        importance_type="split"
                    )
                elif isinstance(self.model, XGBClassifier):
                    split_importance = self.model.get_booster().get_score(
                        importance_type="weight"
                    )
                    # Convert to array matching feature order
                    split_importance = [
                        split_importance.get(f"f{i}", 0)
                        for i in range(len(self.feature_names))
                    ]

                importance_df["gain_importance"] = gain_importance
                importance_df["split_importance"] = split_importance

            except Exception as e:
                print(f"Could not extract multiple importance types: {e}")

        return importance_df

    def initialize_shap_explainer(self, background_samples: int = 100):
        """Initialize SHAP explainer for the model.

        Uses TreeExplainer for tree-based models (faster) and KernelExplainer
        for other model types.

        Args:
            background_samples (int): Number of background samples for KernelExplainer

        Example:
            >>> analyzer.initialize_shap_explainer(background_samples=100)
        """
        print("Initializing SHAP explainer...")

        # Use TreeExplainer for tree-based models (much faster)
        if isinstance(self.model, XGBClassifier):
            # For XGBoost, use KernelExplainer with a wrapper function
            # Sample background data for efficiency
            background = shap.sample(
                self.X_test, min(background_samples, len(self.X_test))
            )

            # Create a wrapper function that doesn't have feature_names_in_
            def model_predict(X):
                return self.model.predict_proba(X)

            self.explainer = shap.KernelExplainer(model_predict, background)
            print(
                f"✓ KernelExplainer initialized for XGBoost with {len(background)} background samples"
            )
        elif isinstance(self.model, (RandomForestClassifier, LGBMClassifier)):
            self.explainer = shap.TreeExplainer(self.model)
            print("✓ TreeExplainer initialized (fast)")
        else:
            # Use KernelExplainer for other models (slower)
            # Sample background data for efficiency
            background = shap.sample(self.X_test, background_samples)
            self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
            print(
                f"✓ KernelExplainer initialized with {background_samples} background samples"
            )

    def calculate_shap_values(self, max_samples: Optional[int] = None) -> np.ndarray:
        """Calculate SHAP values for test data.

        Args:
            max_samples (int, optional): Limit analysis to first N samples for speed

        Returns:
            np.ndarray: SHAP values for predictions

        Raises:
            ValueError: If explainer not initialized

        Example:
            >>> analyzer.initialize_shap_explainer()
            >>> shap_values = analyzer.calculate_shap_values(max_samples=1000)
        """
        if self.explainer is None:
            raise ValueError(
                "Explainer not initialized. Call initialize_shap_explainer() first."
            )

        print("Calculating SHAP values...")

        # Limit samples if specified
        X_analyze = (
            self.X_test if max_samples is None else self.X_test.iloc[:max_samples]
        )

        # Calculate SHAP values
        self.shap_values = self.explainer.shap_values(X_analyze)

        # For binary classification, shap_values might be a list [class0, class1]
        # or a 2D array with shape (n_samples, n_features, n_classes)
        # We want values for positive class (fraud = 1)
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]
        elif len(self.shap_values.shape) == 3:
            # 3D array: (n_samples, n_features, n_classes)
            self.shap_values = self.shap_values[:, :, 1]
        elif len(self.shap_values.shape) == 2 and self.shap_values.shape[0] != len(
            X_analyze
        ):
            # 2D array but wrong shape, likely (n_features, n_classes)
            # This shouldn't happen but handle it just in case
            if self.shap_values.shape[1] == 2:
                self.shap_values = self.shap_values[:, 1]

        # Ensure we have the right shape: (n_samples, n_features)
        expected_shape = (len(X_analyze), len(self.feature_names))
        if self.shap_values.shape != expected_shape:
            print(
                f"Warning: SHAP values shape {self.shap_values.shape} doesn't match expected {expected_shape}"
            )

        print(
            f"✓ SHAP values calculated for {len(X_analyze)} samples (shape: {self.shap_values.shape})"
        )

        return self.shap_values

    def plot_shap_summary(
        self, max_display: int = 10, save_path: Optional[str] = None
    ) -> None:
        """Create SHAP summary plot showing global feature importance.

        Args:
            max_display (int): Maximum number of features to display
            save_path (str, optional): Path to save the plot

        Raises:
            ValueError: If SHAP values not calculated

        Example:
            >>> analyzer.plot_shap_summary(max_display=10,
            ...                           save_path='reports/images/shap_summary.png')
        """
        if self.shap_values is None:
            raise ValueError(
                "SHAP values not calculated. Call calculate_shap_values() first."
            )

        plt.figure(figsize=(10, 8))

        # Create summary plot
        shap.summary_plot(
            self.shap_values,
            (
                self.X_test
                if len(self.shap_values) == len(self.X_test)
                else self.X_test.iloc[: len(self.shap_values)]
            ),
            max_display=max_display,
            show=False,
        )

        plt.title(
            "SHAP Summary Plot - Global Feature Importance",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✓ SHAP summary plot saved to {save_path}")

        plt.show()

    def plot_shap_beeswarm(
        self, max_display: int = 10, save_path: Optional[str] = None
    ) -> None:
        """Create SHAP beeswarm plot (newer version of summary plot).

        Args:
            max_display (int): Maximum number of features to display
            save_path (str, optional): Path to save the plot

        Example:
            >>> analyzer.plot_shap_beeswarm(max_display=10)
        """
        if self.shap_values is None:
            raise ValueError(
                "SHAP values not calculated. Call calculate_shap_values() first."
            )

        plt.figure(figsize=(10, 8))

        X_analyze = (
            self.X_test
            if len(self.shap_values) == len(self.X_test)
            else self.X_test.iloc[: len(self.shap_values)]
        )

        # Create Explanation object for newer SHAP API
        explanation = shap.Explanation(
            values=self.shap_values,
            base_values=np.full(
                len(self.shap_values),
                (
                    self.explainer.expected_value
                    if hasattr(self.explainer, "expected_value")
                    else 0
                ),
            ),
            data=X_analyze.values,
            feature_names=self.feature_names,
        )

        shap.plots.beeswarm(explanation, max_display=max_display, show=False)
        plt.title("SHAP Beeswarm Plot", fontsize=14, fontweight="bold", pad=20)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✓ SHAP beeswarm plot saved to {save_path}")

        plt.show()

    def plot_shap_bar(
        self, max_display: int = 10, save_path: Optional[str] = None
    ) -> None:
        """Create SHAP bar plot showing mean absolute SHAP values.

        Args:
            max_display (int): Maximum number of features to display
            save_path (str, optional): Path to save the plot

        Example:
            >>> analyzer.plot_shap_bar(max_display=10)
        """
        if self.shap_values is None:
            raise ValueError(
                "SHAP values not calculated. Call calculate_shap_values() first."
            )

        plt.figure(figsize=(10, 6))

        X_analyze = (
            self.X_test
            if len(self.shap_values) == len(self.X_test)
            else self.X_test.iloc[: len(self.shap_values)]
        )

        # Create Explanation object
        # Handle expected_value for binary classification (could be array or scalar)
        expected_val = 0
        if hasattr(self.explainer, "expected_value"):
            if isinstance(self.explainer.expected_value, (list, np.ndarray)):
                # For binary classification, use positive class expected value
                expected_val = (
                    self.explainer.expected_value[-1]
                    if len(self.explainer.expected_value) > 1
                    else self.explainer.expected_value[0]
                )
            else:
                expected_val = self.explainer.expected_value

        explanation = shap.Explanation(
            values=self.shap_values,
            base_values=np.full(len(self.shap_values), expected_val),
            data=X_analyze.values,
            feature_names=self.feature_names,
        )

        shap.plots.bar(explanation, max_display=max_display, show=False)
        plt.title(
            "SHAP Bar Plot - Mean Absolute Impact", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✓ SHAP bar plot saved to {save_path}")

        plt.show()

    def plot_shap_force_plot(
        self, sample_index: int, save_path: Optional[str] = None
    ) -> None:
        """Create SHAP force plot for a single prediction.

        Args:
            sample_index (int): Index of sample to analyze
            save_path (str, optional): Path to save the plot (as HTML)

        Example:
            >>> analyzer.plot_shap_force_plot(sample_index=0,
            ...                              save_path='reports/images/force_plot_tp.html')
        """
        if self.shap_values is None:
            raise ValueError(
                "SHAP values not calculated. Call calculate_shap_values() first."
            )

        X_analyze = (
            self.X_test
            if len(self.shap_values) == len(self.X_test)
            else self.X_test.iloc[: len(self.shap_values)]
        )

        # Get expected value (base value)
        expected_value = self.explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[1]  # For binary classification

        # Create force plot
        force_plot = shap.force_plot(
            expected_value,
            self.shap_values[sample_index],
            X_analyze.iloc[sample_index],
            matplotlib=False,  # Use interactive JS plot
        )

        if save_path:
            shap.save_html(save_path, force_plot)
            print(f"✓ SHAP force plot saved to {save_path}")

        return force_plot

    def plot_shap_waterfall(
        self, sample_index: int, save_path: Optional[str] = None
    ) -> None:
        """Create SHAP waterfall plot for a single prediction.

        Args:
            sample_index (int): Index of sample to analyze
            save_path (str, optional): Path to save the plot

        Example:
            >>> analyzer.plot_shap_waterfall(sample_index=0,
            ...                             save_path='reports/images/waterfall_tp.png')
        """
        if self.shap_values is None:
            raise ValueError(
                "SHAP values not calculated. Call calculate_shap_values() first."
            )

        X_analyze = (
            self.X_test
            if len(self.shap_values) == len(self.X_test)
            else self.X_test.iloc[: len(self.shap_values)]
        )

        plt.figure(figsize=(10, 8))

        # Get expected value
        expected_value = self.explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[1]

        # Create Explanation object for single sample
        explanation = shap.Explanation(
            values=self.shap_values[sample_index],
            base_values=expected_value,
            data=X_analyze.iloc[sample_index].values,
            feature_names=self.feature_names,
        )

        shap.plots.waterfall(explanation, show=False)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✓ SHAP waterfall plot saved to {save_path}")

        plt.show()

    def analyze_prediction(self, sample_index: int) -> Dict[str, Any]:
        """Analyze a specific prediction with detailed breakdown.

        Args:
            sample_index (int): Index of sample to analyze

        Returns:
            Dict: Analysis including features, SHAP values, prediction, and label

        Example:
            >>> analysis = analyzer.analyze_prediction(sample_index=0)
            >>> print(f"Prediction: {analysis['prediction']}, Actual: {analysis['actual_label']}")
        """
        if self.shap_values is None:
            raise ValueError(
                "SHAP values not calculated. Call calculate_shap_values() first."
            )

        X_analyze = (
            self.X_test
            if len(self.shap_values) == len(self.X_test)
            else self.X_test.iloc[: len(self.shap_values)]
        )
        y_analyze = (
            self.y_test
            if len(self.shap_values) == len(self.y_test)
            else self.y_test.iloc[: len(self.shap_values)]
        )

        # Get sample data
        sample_features = X_analyze.iloc[sample_index]
        sample_shap = self.shap_values[sample_index]

        # Get prediction
        prediction_proba = self.model.predict_proba(
            sample_features.values.reshape(1, -1)
        )[0]
        prediction = 1 if prediction_proba[1] >= 0.5 else 0

        # Get actual label
        actual_label = y_analyze.iloc[sample_index]

        # Create feature contribution dataframe
        contributions = pd.DataFrame(
            {
                "feature": self.feature_names,
                "value": sample_features.values,
                "shap_value": sample_shap,
            }
        )
        contributions = contributions.sort_values(
            "shap_value", key=abs, ascending=False
        )

        analysis = {
            "sample_index": sample_index,
            "prediction": prediction,
            "prediction_proba": prediction_proba[1],
            "actual_label": actual_label,
            "is_correct": prediction == actual_label,
            "prediction_type": self._get_prediction_type(prediction, actual_label),
            "feature_contributions": contributions,
            "top_5_contributors": contributions.head(5),
        }

        return analysis

    def _get_prediction_type(self, prediction: int, actual: int) -> str:
        """Determine prediction type (TP, TN, FP, FN).

        Args:
            prediction (int): Predicted label
            actual (int): Actual label

        Returns:
            str: Prediction type
        """
        if prediction == 1 and actual == 1:
            return "True Positive (TP)"
        elif prediction == 0 and actual == 0:
            return "True Negative (TN)"
        elif prediction == 1 and actual == 0:
            return "False Positive (FP)"
        else:
            return "False Negative (FN)"

    def compare_importance_rankings(self, top_n: int = 10) -> pd.DataFrame:
        """Compare SHAP importance with built-in feature importance.

        Args:
            top_n (int): Number of top features to compare

        Returns:
            pd.DataFrame: Comparison of rankings

        Example:
            >>> comparison = analyzer.compare_importance_rankings(top_n=10)
        """
        if self.shap_values is None:
            raise ValueError(
                "SHAP values not calculated. Call calculate_shap_values() first."
            )

        # Get built-in importance
        builtin_importance = self.extract_feature_importance()
        builtin_importance = builtin_importance.rename(
            columns={"importance": "builtin_importance"}
        )
        builtin_importance["builtin_rank"] = range(1, len(builtin_importance) + 1)

        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        shap_importance = pd.DataFrame(
            {"feature": self.feature_names, "shap_importance": mean_abs_shap}
        )
        shap_importance = shap_importance.sort_values(
            "shap_importance", ascending=False
        ).reset_index(drop=True)
        shap_importance["shap_rank"] = range(1, len(shap_importance) + 1)

        # Merge both importance measures
        comparison = pd.merge(
            builtin_importance[["feature", "builtin_importance", "builtin_rank"]],
            shap_importance[["feature", "shap_importance", "shap_rank"]],
            on="feature",
        )

        # Calculate rank difference
        comparison["rank_difference"] = abs(
            comparison["builtin_rank"] - comparison["shap_rank"]
        )

        # Sort by SHAP importance
        comparison = comparison.sort_values(
            "shap_importance", ascending=False
        ).reset_index(drop=True)

        return comparison.head(top_n)

    def identify_discrepancies(self, threshold: int = 5) -> pd.DataFrame:
        """Identify features with large discrepancies between importance methods.

        Args:
            threshold (int): Rank difference threshold to flag discrepancies

        Returns:
            pd.DataFrame: Features with significant rank differences

        Example:
            >>> discrepancies = analyzer.identify_discrepancies(threshold=5)
        """
        comparison = self.compare_importance_rankings(top_n=len(self.feature_names))
        discrepancies = comparison[comparison["rank_difference"] >= threshold]

        return discrepancies.sort_values("rank_difference", ascending=False)

    def plot_importance_comparison(
        self, top_n: int = 10, save_path: Optional[str] = None
    ) -> plt.Figure:
        """Create side-by-side comparison of built-in and SHAP importance.

        Args:
            top_n (int): Number of top features to display
            save_path (str, optional): Path to save the plot

        Returns:
            plt.Figure: Matplotlib figure object

        Example:
            >>> fig = analyzer.plot_importance_comparison(top_n=10)
        """
        comparison = self.compare_importance_rankings(top_n=top_n)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Built-in importance
        axes[0].barh(
            range(len(comparison)),
            comparison["builtin_importance"],
            color="steelblue",
            alpha=0.8,
        )
        axes[0].set_yticks(range(len(comparison)))
        axes[0].set_yticklabels(comparison["feature"])
        axes[0].set_xlabel("Built-in Feature Importance", fontsize=12)
        axes[0].set_title("Built-in Feature Importance", fontsize=14, fontweight="bold")
        axes[0].invert_yaxis()

        # Plot 2: SHAP importance
        axes[1].barh(
            range(len(comparison)),
            comparison["shap_importance"],
            color="coral",
            alpha=0.8,
        )
        axes[1].set_yticks(range(len(comparison)))
        axes[1].set_yticklabels(comparison["feature"])
        axes[1].set_xlabel("Mean Absolute SHAP Value", fontsize=12)
        axes[1].set_title("SHAP Feature Importance", fontsize=14, fontweight="bold")
        axes[1].invert_yaxis()

        plt.suptitle(
            f"Feature Importance Comparison: Top {top_n} Features",
            fontsize=16,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✓ Importance comparison plot saved to {save_path}")

        return fig


class RecommendationGenerator:
    """Generate business recommendations from model explainability insights.

    This class analyzes SHAP values and feature importance to identify
    key fraud drivers and generate actionable business recommendations.

    Attributes:
        analyzer (ExplainabilityAnalyzer): Explainability analyzer object
        fraud_drivers (pd.DataFrame): Top fraud-driving features
    """

    def __init__(self, analyzer: ExplainabilityAnalyzer):
        """Initialize the RecommendationGenerator.

        Args:
            analyzer (ExplainabilityAnalyzer): Initialized analyzer with SHAP values
        """
        self.analyzer = analyzer
        self.fraud_drivers = None

    def identify_fraud_drivers(self, top_n: int = 5) -> pd.DataFrame:
        """Identify top N fraud-driving features from SHAP analysis.

        Args:
            top_n (int): Number of top drivers to identify

        Returns:
            pd.DataFrame: Top fraud drivers with SHAP importance

        Example:
            >>> generator = RecommendationGenerator(analyzer)
            >>> drivers = generator.identify_fraud_drivers(top_n=5)
        """
        if self.analyzer.shap_values is None:
            raise ValueError("SHAP values not calculated in analyzer.")

        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(self.analyzer.shap_values).mean(axis=0)

        self.fraud_drivers = pd.DataFrame(
            {
                "feature": self.analyzer.feature_names,
                "mean_abs_shap": mean_abs_shap,
                "importance_score": mean_abs_shap / mean_abs_shap.sum(),  # Normalized
            }
        )

        self.fraud_drivers = self.fraud_drivers.sort_values(
            "mean_abs_shap", ascending=False
        ).head(top_n)

        return self.fraud_drivers

    def generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate actionable business recommendations based on fraud drivers.

        Returns:
            List[Dict]: List of recommendations with feature, insight, and action

        Example:
            >>> recommendations = generator.generate_recommendations()
            >>> for rec in recommendations:
            ...     print(f"{rec['feature']}: {rec['action']}")
        """
        if self.fraud_drivers is None:
            self.identify_fraud_drivers(top_n=5)

        recommendations = []

        for _, row in self.fraud_drivers.iterrows():
            feature = row["feature"]
            importance = row["importance_score"]

            # Generate recommendation based on feature
            recommendation = self._generate_feature_recommendation(feature, importance)
            recommendations.append(recommendation)

        return recommendations

    def _generate_feature_recommendation(
        self, feature: str, importance: float
    ) -> Dict[str, str]:
        """Generate specific recommendation for a feature.

        Args:
            feature (str): Feature name
            importance (float): Importance score

        Returns:
            Dict: Recommendation with feature, insight, and action
        """
        # Feature-specific recommendations
        feature_recommendations = {
            "V14": {
                "insight": "V14 shows strong predictive power for fraud detection. This PCA feature captures critical transaction patterns.",
                "action": "Implement real-time monitoring of V14 values. Flag transactions with V14 values outside normal ranges for enhanced review.",
            },
            "V10": {
                "insight": "V10 is a key discriminator between legitimate and fraudulent transactions.",
                "action": "Develop V10-based risk scoring rules. Consider dynamic thresholds that adapt to user behavior patterns.",
            },
            "V17": {
                "insight": "V17 contributes significantly to fraud prediction, indicating specific transaction characteristics.",
                "action": "Integrate V17 into fraud prevention workflows. Create alerts for unusual V17 patterns in customer transactions.",
            },
            "V12": {
                "insight": "V12 reveals important fraud indicators in transaction data.",
                "action": "Use V12 as a primary feature in fraud risk models. Implement automated blocking for extreme V12 values.",
            },
            "V4": {
                "insight": "V4 shows consistent fraud signal strength across different transaction types.",
                "action": "Incorporate V4 into multi-factor authentication triggers. High-risk V4 values should prompt additional verification.",
            },
            "Amount": {
                "insight": "Transaction amount is a strong fraud indicator, with certain ranges showing higher risk.",
                "action": "Implement transaction amount-based velocity checks. Flag rapid sequences of similar amounts for review.",
            },
            "Time": {
                "insight": "Transaction timing patterns reveal fraud behavior, especially during off-peak hours.",
                "action": "Enhance fraud detection during high-risk time periods. Apply stricter verification for unusual timing patterns.",
            },
        }

        # Get specific recommendation or generate generic one
        if feature in feature_recommendations:
            rec = feature_recommendations[feature]
        else:
            rec = {
                "insight": f"{feature} is a significant fraud predictor (importance: {importance:.1%}).",
                "action": f"Monitor {feature} values closely. Establish baseline patterns and flag deviations for investigation.",
            }

        return {
            "feature": feature,
            "importance": importance,
            "insight": rec["insight"],
            "action": rec["action"],
        }

    def create_business_report(self, save_path: Optional[str] = None) -> str:
        """Create executive summary report of fraud detection insights.

        Args:
            save_path (str, optional): Path to save the report

        Returns:
            str: Formatted business report

        Example:
            >>> report = generator.create_business_report('reports/business_insights.md')
        """
        recommendations = self.generate_recommendations()

        report = "# Fraud Detection Model - Business Insights Report\n\n"
        report += "## Executive Summary\n\n"
        report += "Based on SHAP explainability analysis, we have identified the key drivers of "
        report += "fraudulent transactions and developed actionable recommendations to enhance "
        report += "fraud prevention capabilities.\n\n"

        report += "## Top Fraud Drivers\n\n"
        report += (
            "The following features have the strongest impact on fraud prediction:\n\n"
        )

        for i, rec in enumerate(recommendations, 1):
            report += (
                f"### {i}. {rec['feature']} (Importance: {rec['importance']:.1%})\n\n"
            )
            report += f"**Insight:** {rec['insight']}\n\n"
            report += f"**Recommended Action:** {rec['action']}\n\n"

        report += "## Strategic Recommendations\n\n"
        report += "1. **Enhanced Monitoring:** Implement real-time monitoring for all identified fraud drivers\n"
        report += "2. **Risk Scoring:** Develop multi-factor risk scores incorporating SHAP insights\n"
        report += "3. **Adaptive Thresholds:** Create dynamic thresholds that adapt to evolving fraud patterns\n"
        report += "4. **Investigation Prioritization:** Use feature importance to prioritize manual reviews\n"
        report += "5. **Customer Communication:** Develop educational content for customers about fraud indicators\n\n"

        report += "## Next Steps\n\n"
        report += "- Deploy model in production with continuous monitoring\n"
        report += "- Establish feedback loop to capture fraud analyst insights\n"
        report += "- Schedule quarterly model retraining and explainability analysis\n"
        report += (
            "- Develop customer-facing fraud prevention tools based on insights\n\n"
        )

        report += "---\n"
        report += (
            "*Report generated using SHAP (SHapley Additive exPlanations) analysis*\n"
        )

        if save_path:
            with open(save_path, "w") as f:
                f.write(report)
            print(f"✓ Business report saved to {save_path}")

        return report

    def plot_fraud_driver_summary(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create visualization of top fraud drivers with importance scores.

        Args:
            save_path (str, optional): Path to save the plot

        Returns:
            plt.Figure: Matplotlib figure object

        Example:
            >>> fig = generator.plot_fraud_driver_summary('reports/images/fraud_drivers.png')
        """
        if self.fraud_drivers is None:
            self.identify_fraud_drivers(top_n=5)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Create horizontal bar plot
        bars = ax.barh(
            range(len(self.fraud_drivers)),
            self.fraud_drivers["importance_score"],
            color="crimson",
            alpha=0.7,
            edgecolor="darkred",
            linewidth=1.5,
        )

        ax.set_yticks(range(len(self.fraud_drivers)))
        ax.set_yticklabels(self.fraud_drivers["feature"])
        ax.set_xlabel("Normalized Importance Score", fontsize=12, fontweight="bold")
        ax.set_ylabel("Features", fontsize=12, fontweight="bold")
        ax.set_title(
            "Top 5 Fraud Drivers by SHAP Importance",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )

        # Add value labels on bars
        for i, (bar, value) in enumerate(
            zip(bars, self.fraud_drivers["importance_score"])
        ):
            ax.text(
                value + 0.01,
                i,
                f"{value:.1%}",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

        # Invert y-axis
        ax.invert_yaxis()

        # Add grid
        ax.grid(axis="x", alpha=0.3, linestyle="--")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✓ Fraud driver summary saved to {save_path}")

        return fig
