"""Script for running SHAP explainability analysis on fraud detection model.

This script performs comprehensive model explainability analysis including:
- Feature importance extraction
- SHAP global analysis
- SHAP local analysis for sample predictions
- Business recommendations generation

Usage:
    python run_shap_analysis.py --model models/best_model_xgboost_tuned.pkl \\
                                --test-data data/processed/cc_test_scaled_full.csv \\
                                --output-dir reports/

Author: Fraud Detection Team
Date: 2025-12-30
"""

import sys
import os
import argparse
import logging
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from src.shap_analysis import ExplainabilityAnalyzer, RecommendationGenerator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("explainability_analysis.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run SHAP explainability analysis on fraud detection model"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="../models/best_model_xgboost_tuned.pkl",
        help="Path to trained model file (default: ../models/best_model_xgboost_tuned.pkl)",
    )

    parser.add_argument(
        "--test-data",
        type=str,
        default="../data/processed/cc_test_scaled_full.csv",
        help="Path to test data CSV (default: ../data/processed/cc_test_scaled_full.csv)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="../reports/",
        help="Directory for output reports and images (default: ../reports/)",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum samples for SHAP calculation (default: 1000, use -1 for all)",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top features to display (default: 10)",
    )

    parser.add_argument(
        "--analyze-cases",
        action="store_true",
        help="Analyze individual prediction cases (TP, FP, FN)",
    )

    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating plot visualizations (faster execution)",
    )

    return parser.parse_args()


def load_data(model_path: str, test_data_path: str):
    """Load model and test data.

    Args:
        model_path (str): Path to model file
        test_data_path (str): Path to test data

    Returns:
        tuple: (model, X_test, y_test)
    """
    logger.info(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    logger.info(f"✓ Loaded model: {type(model).__name__}")

    logger.info(f"Loading test data from {test_data_path}...")
    test_df = pd.read_csv(test_data_path)
    logger.info(f"✓ Loaded test data: {test_df.shape}")

    # Separate features and target
    X_test = test_df.drop("Class", axis=1)
    y_test = test_df["Class"]

    logger.info(f"Test set - Features: {X_test.shape}, Fraud rate: {y_test.mean():.2%}")

    return model, X_test, y_test


def run_feature_importance_analysis(
    analyzer: ExplainabilityAnalyzer,
    output_dir: Path,
    top_n: int,
    skip_plots: bool = False,
):
    """Run feature importance analysis.

    Args:
        analyzer (ExplainabilityAnalyzer): Analyzer instance
        output_dir (Path): Output directory
        top_n (int): Number of top features
        skip_plots (bool): Skip plotting
    """
    logger.info("=" * 60)
    logger.info("FEATURE IMPORTANCE ANALYSIS")
    logger.info("=" * 60)

    # Extract importance
    importance_df = analyzer.extract_feature_importance()

    logger.info(f"\nTop {top_n} Features by Built-in Importance:")
    logger.info("\n" + importance_df.head(top_n).to_string(index=False))

    # Save to CSV
    importance_path = output_dir / "feature_importance_builtin.csv"
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"\n✓ Feature importance saved to {importance_path}")

    # Plot
    if not skip_plots:
        image_dir = output_dir / "images"
        image_dir.mkdir(exist_ok=True)

        plot_path = image_dir / "feature_importance_builtin.png"
        analyzer.plot_feature_importance(top_n=top_n, save_path=str(plot_path))
        plt.close("all")


def run_shap_global_analysis(
    analyzer: ExplainabilityAnalyzer,
    output_dir: Path,
    top_n: int,
    max_samples: int,
    skip_plots: bool = False,
):
    """Run SHAP global analysis.

    Args:
        analyzer (ExplainabilityAnalyzer): Analyzer instance
        output_dir (Path): Output directory
        top_n (int): Number of top features
        max_samples (int): Maximum samples for SHAP
        skip_plots (bool): Skip plotting
    """
    logger.info("\n" + "=" * 60)
    logger.info("SHAP GLOBAL ANALYSIS")
    logger.info("=" * 60)

    # Initialize explainer
    analyzer.initialize_shap_explainer()

    # Calculate SHAP values
    max_samples_arg = None if max_samples == -1 else max_samples
    shap_values = analyzer.calculate_shap_values(max_samples=max_samples_arg)

    # Create plots
    if not skip_plots:
        image_dir = output_dir / "images"

        logger.info("\nGenerating SHAP summary plot...")
        analyzer.plot_shap_summary(
            max_display=top_n, save_path=str(image_dir / "shap_summary_plot.png")
        )
        plt.close("all")

        logger.info("Generating SHAP bar plot...")
        analyzer.plot_shap_bar(
            max_display=top_n, save_path=str(image_dir / "shap_bar_plot.png")
        )
        plt.close("all")


def run_importance_comparison(
    analyzer: ExplainabilityAnalyzer,
    output_dir: Path,
    top_n: int,
    skip_plots: bool = False,
):
    """Compare feature importance methods.

    Args:
        analyzer (ExplainabilityAnalyzer): Analyzer instance
        output_dir (Path): Output directory
        top_n (int): Number of top features
        skip_plots (bool): Skip plotting
    """
    logger.info("\n" + "=" * 60)
    logger.info("FEATURE IMPORTANCE COMPARISON")
    logger.info("=" * 60)

    # Compare rankings
    comparison_df = analyzer.compare_importance_rankings(top_n=top_n)

    logger.info(f"\nTop {top_n} Features - Ranking Comparison:")
    logger.info("\n" + comparison_df.to_string(index=False))

    # Save comparison
    comparison_path = output_dir / "feature_importance_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"\n✓ Comparison saved to {comparison_path}")

    # Identify discrepancies
    discrepancies = analyzer.identify_discrepancies(threshold=5)
    if len(discrepancies) > 0:
        logger.info("\nFeatures with Significant Ranking Discrepancies:")
        logger.info("\n" + discrepancies.to_string(index=False))
    else:
        logger.info("\nNo significant discrepancies found between importance methods.")

    # Plot comparison
    if not skip_plots:
        image_dir = output_dir / "images"
        plot_path = image_dir / "importance_comparison.png"

        logger.info("\nGenerating importance comparison plot...")
        analyzer.plot_importance_comparison(top_n=top_n, save_path=str(plot_path))
        plt.close("all")


def run_local_analysis(
    analyzer: ExplainabilityAnalyzer,
    output_dir: Path,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    skip_plots: bool = False,
):
    """Analyze individual prediction cases.

    Args:
        analyzer (ExplainabilityAnalyzer): Analyzer instance
        output_dir (Path): Output directory
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        skip_plots (bool): Skip plotting
    """
    logger.info("\n" + "=" * 60)
    logger.info("LOCAL PREDICTION ANALYSIS")
    logger.info("=" * 60)

    # Get predictions for sample
    n_samples = len(analyzer.shap_values)
    y_pred = analyzer.model.predict(X_test.iloc[:n_samples])
    y_true = y_test.iloc[:n_samples]

    # Find examples
    tp_indices = np.where((y_pred == 1) & (y_true == 1))[0]
    fp_indices = np.where((y_pred == 1) & (y_true == 0))[0]
    fn_indices = np.where((y_pred == 0) & (y_true == 1))[0]

    logger.info(f"\nAvailable cases:")
    logger.info(f"  True Positives: {len(tp_indices)}")
    logger.info(f"  False Positives: {len(fp_indices)}")
    logger.info(f"  False Negatives: {len(fn_indices)}")

    image_dir = output_dir / "images"

    # Analyze each case type
    cases = {}
    if len(tp_indices) > 0:
        cases["TP"] = tp_indices[0]
    if len(fp_indices) > 0:
        cases["FP"] = fp_indices[0]
    if len(fn_indices) > 0:
        cases["FN"] = fn_indices[0]

    for case_type, idx in cases.items():
        logger.info(f"\n{'-'*60}")
        logger.info(f"Analyzing {case_type} case (index: {idx})...")
        logger.info(f"{'-'*60}")

        # Analyze prediction
        analysis = analyzer.analyze_prediction(idx)

        logger.info(
            f"Prediction: {analysis['prediction']} "
            + f"(Probability: {analysis['prediction_proba']:.2%})"
        )
        logger.info(f"Actual Label: {analysis['actual_label']}")
        logger.info(f"Classification: {analysis['prediction_type']}")
        logger.info(f"\nTop 5 Contributing Features:")
        logger.info("\n" + analysis["top_5_contributors"].to_string(index=False))

        # Create plots
        if not skip_plots:
            # Waterfall plot
            waterfall_path = image_dir / f"shap_waterfall_{case_type.lower()}.png"
            analyzer.plot_shap_waterfall(idx, save_path=str(waterfall_path))
            plt.close("all")

            # Force plot (HTML)
            force_path = image_dir / f"shap_force_{case_type.lower()}.html"
            analyzer.plot_shap_force_plot(idx, save_path=str(force_path))


def run_business_recommendations(
    analyzer: ExplainabilityAnalyzer, output_dir: Path, skip_plots: bool = False
):
    """Generate business recommendations.

    Args:
        analyzer (ExplainabilityAnalyzer): Analyzer instance
        output_dir (Path): Output directory
        skip_plots (bool): Skip plotting
    """
    logger.info("\n" + "=" * 60)
    logger.info("BUSINESS RECOMMENDATIONS")
    logger.info("=" * 60)

    # Initialize generator
    generator = RecommendationGenerator(analyzer)

    # Identify fraud drivers
    fraud_drivers = generator.identify_fraud_drivers(top_n=5)

    logger.info("\nTop 5 Fraud Drivers:")
    logger.info("\n" + fraud_drivers.to_string(index=False))

    # Generate recommendations
    recommendations = generator.generate_recommendations()

    logger.info("\nACTIONABLE RECOMMENDATIONS:")
    logger.info("=" * 60)
    for i, rec in enumerate(recommendations, 1):
        logger.info(
            f"\n{i}. {rec['feature'].upper()} (Importance: {rec['importance']:.1%})"
        )
        logger.info(f"   Insight: {rec['insight']}")
        logger.info(f"   Action:  {rec['action']}")

    # Create business report
    report_path = output_dir / "BUSINESS_INSIGHTS_REPORT.md"
    report = generator.create_business_report(save_path=str(report_path))
    logger.info(f"\n✓ Business report saved to {report_path}")

    # Plot fraud drivers
    if not skip_plots:
        image_dir = output_dir / "images"
        plot_path = image_dir / "fraud_drivers_summary.png"

        logger.info("Generating fraud drivers visualization...")
        generator.plot_fraud_driver_summary(save_path=str(plot_path))
        plt.close("all")


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()

    logger.info("=" * 60)
    logger.info("FRAUD DETECTION MODEL - EXPLAINABILITY ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Test Data: {args.test_data}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Max Samples: {'All' if args.max_samples == -1 else args.max_samples}")
    logger.info(f"Top N Features: {args.top_n}")
    logger.info("=" * 60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    (output_dir / "images").mkdir(exist_ok=True)

    try:
        # Load data
        model, X_test, y_test = load_data(args.model, args.test_data)

        # Initialize analyzer
        logger.info("\nInitializing explainability analyzer...")
        analyzer = ExplainabilityAnalyzer(model, X_test, y_test)

        # Run analyses
        run_feature_importance_analysis(
            analyzer, output_dir, args.top_n, args.skip_plots
        )
        run_shap_global_analysis(
            analyzer, output_dir, args.top_n, args.max_samples, args.skip_plots
        )
        run_importance_comparison(analyzer, output_dir, args.top_n, args.skip_plots)

        # Optional: Local analysis
        if args.analyze_cases:
            run_local_analysis(analyzer, output_dir, X_test, y_test, args.skip_plots)

        # Business recommendations
        run_business_recommendations(analyzer, output_dir, args.skip_plots)

        logger.info("\n" + "=" * 60)
        logger.info("EXPLAINABILITY ANALYSIS COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"All results saved to: {output_dir}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
