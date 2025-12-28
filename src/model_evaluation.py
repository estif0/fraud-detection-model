"""Module for model evaluation and comparison.

This module contains classes for evaluating fraud detection models using
appropriate metrics for imbalanced classification:
- Precision, Recall, F1-Score
- ROC-AUC and Precision-Recall AUC
- Confusion Matrix
- Model comparison and selection
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
)
import warnings

warnings.filterwarnings("ignore")


class ModelEvaluator:
    """Class for comprehensive model evaluation.

    Provides methods to evaluate fraud detection models using metrics
    appropriate for imbalanced classification problems.

    Attributes:
        model_name (str): Name identifier for the model being evaluated.
    """

    def __init__(self, model_name: str = "Model"):
        """Initialize the ModelEvaluator.

        Args:
            model_name (str): Name to identify the model in reports.
        """
        self.model_name = model_name
        self.evaluation_results = {}

    def evaluate_model(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Comprehensive model evaluation.

        Calculates all relevant metrics for fraud detection including
        accuracy, precision, recall, F1, ROC-AUC, and PR-AUC.

        Args:
            y_true (pd.Series): True labels.
            y_pred (np.ndarray): Predicted labels.
            y_pred_proba (np.ndarray): Predicted probabilities for positive class.

        Returns:
            Dictionary containing all evaluation metrics.
        """
        results = {}

        # Basic metrics
        results["accuracy"] = accuracy_score(y_true, y_pred)
        results["precision"] = precision_score(y_true, y_pred, zero_division=0)
        results["recall"] = recall_score(y_true, y_pred, zero_division=0)
        results["f1_score"] = f1_score(y_true, y_pred, zero_division=0)

        # Probability-based metrics (if available)
        if y_pred_proba is not None:
            results["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
            results["pr_auc"] = average_precision_score(y_true, y_pred_proba)

        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        results["true_negatives"] = int(tn)
        results["false_positives"] = int(fp)
        results["false_negatives"] = int(fn)
        results["true_positives"] = int(tp)

        # Specificity
        results["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0

        self.evaluation_results = results
        return results

    def print_evaluation_report(self, results: Optional[Dict[str, Any]] = None) -> None:
        """Print formatted evaluation report.

        Args:
            results (Dict): Evaluation results dictionary. If None, uses stored results.
        """
        if results is None:
            results = self.evaluation_results

        if not results:
            print("No evaluation results available. Run evaluate_model() first.")
            return

        print(f"\n{'='*60}")
        print(f"Evaluation Report: {self.model_name}")
        print(f"{'='*60}\n")

        print("Classification Metrics:")
        print(f"  Accuracy:    {results['accuracy']:.4f}")
        print(f"  Precision:   {results['precision']:.4f}")
        print(f"  Recall:      {results['recall']:.4f}")
        print(f"  F1-Score:    {results['f1_score']:.4f}")
        print(f"  Specificity: {results['specificity']:.4f}")

        if "roc_auc" in results:
            print(f"\nProbability-based Metrics:")
            print(f"  ROC-AUC:     {results['roc_auc']:.4f}")
            print(f"  PR-AUC:      {results['pr_auc']:.4f}")

        print(f"\nConfusion Matrix Components:")
        print(f"  True Positives:  {results['true_positives']}")
        print(f"  True Negatives:  {results['true_negatives']}")
        print(f"  False Positives: {results['false_positives']}")
        print(f"  False Negatives: {results['false_negatives']}")

        print(f"{'='*60}\n")

    def plot_confusion_matrix(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6),
    ) -> plt.Figure:
        """Plot confusion matrix heatmap.

        Args:
            y_true (pd.Series): True labels.
            y_pred (np.ndarray): Predicted labels.
            save_path (str): Path to save figure.
            figsize (tuple): Figure size.

        Returns:
            Matplotlib figure object.
        """
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"],
            ax=ax,
        )

        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title(
            f"Confusion Matrix - {self.model_name}", fontsize=14, fontweight="bold"
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Confusion matrix saved to {save_path}")

        return fig

    def plot_roc_curve(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6),
    ) -> plt.Figure:
        """Plot ROC curve.

        Args:
            y_true (pd.Series): True labels.
            y_pred_proba (np.ndarray): Predicted probabilities for positive class.
            save_path (str): Path to save figure.
            figsize (tuple): Figure size.

        Returns:
            Matplotlib figure object.
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})"
        )
        ax.plot(
            [0, 1],
            [0, 1],
            color="navy",
            lw=2,
            linestyle="--",
            label="Random Classifier",
        )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(f"ROC Curve - {self.model_name}", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"ROC curve saved to {save_path}")

        return fig

    def plot_precision_recall_curve(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6),
    ) -> plt.Figure:
        """Plot Precision-Recall curve.

        PR curve is more informative than ROC for imbalanced datasets
        as it focuses on the performance on the minority (fraud) class.

        Args:
            y_true (pd.Series): True labels.
            y_pred_proba (np.ndarray): Predicted probabilities for positive class.
            save_path (str): Path to save figure.
            figsize (tuple): Figure size.

        Returns:
            Matplotlib figure object.
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(
            recall,
            precision,
            color="darkgreen",
            lw=2,
            label=f"PR curve (AUC = {pr_auc:.4f})",
        )

        # Baseline (random classifier performance)
        baseline = sum(y_true) / len(y_true)
        ax.axhline(
            y=baseline,
            color="navy",
            lw=2,
            linestyle="--",
            label=f"Random Classifier (baseline = {baseline:.4f})",
        )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title(
            f"Precision-Recall Curve - {self.model_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(loc="lower left", fontsize=10)
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"PR curve saved to {save_path}")

        return fig

    def find_optimal_threshold(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
        metric: str = "f1",
        beta: float = 1.0,
    ) -> Tuple[float, Dict[str, float]]:
        """Find optimal probability threshold for classification.

        For imbalanced fraud detection, the default 0.5 threshold often produces
        too many false positives. This method finds the threshold that maximizes
        a chosen metric.

        Args:
            y_true (pd.Series): True labels.
            y_pred_proba (np.ndarray): Predicted probabilities for positive class.
            metric (str): Metric to optimize ('f1', 'f2', 'precision', 'recall', 'balanced').
                         'f1' = Equal weight to precision and recall
                         'f2' = More weight to recall (catch more fraud, accept more FP)
                         'precision' = Minimize false positives
                         'recall' = Minimize false negatives
                         'balanced' = Minimize difference between precision and recall
            beta (float): Beta value for F-beta score (only used if metric='fbeta').

        Returns:
            Tuple of (optimal_threshold, metrics_at_threshold)

        Example:
            >>> evaluator = ModelEvaluator()
            >>> threshold, metrics = evaluator.find_optimal_threshold(y_test, y_proba, metric='f1')
            >>> print(f"Optimal threshold: {threshold:.3f}")
            >>> print(f"F1-Score: {metrics['f1']:.4f}")
        """
        thresholds = np.linspace(0.1, 0.9, 81)  # Test 81 thresholds from 0.1 to 0.9
        best_score = 0
        best_threshold = 0.5
        best_metrics = {}

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)

            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            # Calculate score based on chosen metric
            if metric == "f1":
                score = f1
            elif metric == "f2":
                # F2 score - weights recall higher than precision
                score = (
                    (5 * precision * recall) / (4 * precision + recall)
                    if (precision + recall) > 0
                    else 0
                )
            elif metric == "fbeta":
                beta_sq = beta**2
                score = (
                    ((1 + beta_sq) * precision * recall)
                    / (beta_sq * precision + recall)
                    if (precision + recall) > 0
                    else 0
                )
            elif metric == "precision":
                score = precision
            elif metric == "recall":
                score = recall
            elif metric == "balanced":
                # Minimize difference between precision and recall
                score = 1 - abs(precision - recall)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_metrics = {
                    "threshold": threshold,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "score": score,
                }

        return best_threshold, best_metrics

    def generate_classification_report(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None,
    ) -> str:
        """Generate detailed classification report.

        Args:
            y_true (pd.Series): True labels.
            y_pred (np.ndarray): Predicted labels.
            target_names (List[str]): Names for classes.

        Returns:
            Formatted classification report string.
        """
        if target_names is None:
            target_names = ["Legitimate", "Fraud"]

        report = classification_report(
            y_true, y_pred, target_names=target_names, zero_division=0
        )

        return report


class ModelComparator:
    """Class for comparing multiple models.

    Facilitates side-by-side comparison of different fraud detection models
    to support model selection decisions.

    Attributes:
        models_results (Dict): Dictionary storing results for each model.
    """

    def __init__(self):
        """Initialize the ModelComparator."""
        self.models_results = {}

    def add_model_results(self, model_name: str, results: Dict[str, Any]) -> None:
        """Add evaluation results for a model.

        Args:
            model_name (str): Name identifier for the model.
            results (Dict): Dictionary of evaluation metrics.
        """
        self.models_results[model_name] = results

    def create_comparison_table(
        self, metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Create comparison table of model performance.

        Args:
            metrics (List[str]): List of metrics to include. If None, includes all.

        Returns:
            DataFrame with models as rows and metrics as columns.
        """
        if not self.models_results:
            print("No model results available. Add results using add_model_results().")
            return pd.DataFrame()

        # If no specific metrics requested, use common ones
        if metrics is None:
            metrics = [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
                "pr_auc",
                "specificity",
            ]

        comparison_data = {}
        for model_name, results in self.models_results.items():
            comparison_data[model_name] = {
                metric: results.get(metric, np.nan) for metric in metrics
            }

        df = pd.DataFrame(comparison_data).T
        df.index.name = "Model"

        return df

    def plot_model_comparison(
        self,
        metrics: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6),
    ) -> plt.Figure:
        """Plot bar chart comparing models across metrics.

        Args:
            metrics (List[str]): Metrics to compare.
            save_path (str): Path to save figure.
            figsize (tuple): Figure size.

        Returns:
            Matplotlib figure object.
        """
        if metrics is None:
            metrics = ["precision", "recall", "f1_score", "roc_auc", "pr_auc"]

        df = self.create_comparison_table(metrics)

        if df.empty:
            print("No data to plot.")
            return None

        fig, ax = plt.subplots(figsize=figsize)

        df[metrics].plot(kind="bar", ax=ax, width=0.8)

        ax.set_xlabel("Model", fontsize=12, fontweight="bold")
        ax.set_ylabel("Score", fontsize=12, fontweight="bold")
        ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
        ax.set_ylim([0, 1.0])
        ax.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        ax.set_xticklabels(df.index, rotation=45, ha="right")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Comparison plot saved to {save_path}")

        return fig

    def select_best_model(
        self,
        primary_metric: str = "f1_score",
        secondary_metric: Optional[str] = "pr_auc",
    ) -> Tuple[str, Dict[str, Any]]:
        """Select best model based on metrics.

        Selects the best performing model based on primary metric,
        with secondary metric as tiebreaker.

        Args:
            primary_metric (str): Main metric for selection (default: 'f1_score').
            secondary_metric (str): Tiebreaker metric (default: 'pr_auc').

        Returns:
            Tuple of (best_model_name, best_model_results).
        """
        if not self.models_results:
            raise ValueError("No model results available for comparison.")

        best_model_name = None
        best_primary_score = -np.inf
        best_secondary_score = -np.inf

        for model_name, results in self.models_results.items():
            primary_score = results.get(primary_metric, -np.inf)
            secondary_score = results.get(secondary_metric, -np.inf)

            # Compare primary metric first
            if primary_score > best_primary_score:
                best_primary_score = primary_score
                best_secondary_score = secondary_score
                best_model_name = model_name
            # If tied on primary, use secondary
            elif (
                primary_score == best_primary_score
                and secondary_score > best_secondary_score
            ):
                best_secondary_score = secondary_score
                best_model_name = model_name

        return best_model_name, self.models_results[best_model_name]

    def print_comparison_report(self) -> None:
        """Print formatted comparison report."""
        if not self.models_results:
            print("No model results available.")
            return

        df = self.create_comparison_table()

        print(f"\n{'='*80}")
        print("Model Comparison Report")
        print(f"{'='*80}\n")
        print(df.to_string())
        print(f"\n{'='*80}")

        # Highlight best model
        best_model, best_results = self.select_best_model()
        print(f"\nBest Model: {best_model}")
        print(f"F1-Score: {best_results['f1_score']:.4f}")
        print(f"PR-AUC: {best_results.get('pr_auc', 'N/A')}")
        print(f"{'='*80}\n")

    def generate_model_selection_justification(
        self, best_model_name: str, interpretability_notes: Optional[str] = None
    ) -> str:
        """Generate justification for model selection.

        Creates a narrative explaining why a particular model was selected,
        considering both performance metrics and interpretability.

        Args:
            best_model_name (str): Name of the selected model.
            interpretability_notes (str): Notes on model interpretability.

        Returns:
            Formatted justification text.
        """
        if best_model_name not in self.models_results:
            raise ValueError(f"Model '{best_model_name}' not found in results.")

        best_results = self.models_results[best_model_name]
        df = self.create_comparison_table()

        justification = f"\n{'='*80}\n"
        justification += "MODEL SELECTION JUSTIFICATION\n"
        justification += f"{'='*80}\n\n"
        justification += f"Selected Model: {best_model_name}\n\n"

        justification += "Performance Metrics:\n"
        for metric, value in best_results.items():
            if isinstance(value, (int, float)) and metric not in [
                "true_positives",
                "true_negatives",
                "false_positives",
                "false_negatives",
            ]:
                justification += f"  {metric}: {value:.4f}\n"

        justification += "\nRationale:\n"
        justification += (
            f"The {best_model_name} was selected based on comprehensive evaluation "
        )
        justification += "across multiple metrics relevant for fraud detection:\n\n"

        # F1-Score rationale
        justification += (
            f"1. F1-Score ({best_results['f1_score']:.4f}): Provides balanced measure "
        )
        justification += (
            "of precision and recall, critical for handling class imbalance.\n\n"
        )

        # PR-AUC rationale
        if "pr_auc" in best_results:
            justification += f"2. PR-AUC ({best_results['pr_auc']:.4f}): More informative than ROC-AUC "
            justification += (
                "for imbalanced datasets, focusing on minority class performance.\n\n"
            )

        # Business context
        justification += "3. Business Impact:\n"
        justification += f"   - Precision ({best_results['precision']:.4f}): Minimizes false positives "
        justification += "(legitimate transactions incorrectly flagged as fraud).\n"
        justification += (
            f"   - Recall ({best_results['recall']:.4f}): Maximizes fraud detection "
        )
        justification += "(minimizes missed fraudulent transactions).\n\n"

        # Interpretability
        if interpretability_notes:
            justification += f"4. Interpretability: {interpretability_notes}\n\n"

        justification += f"{'='*80}\n"

        return justification
