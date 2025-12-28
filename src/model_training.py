"""Module for model training and building.

This module contains classes for training fraud detection models including:
- Data splitting with stratification
- Baseline Logistic Regression
- Ensemble models (Random Forest, XGBoost, LightGBM)
- Hyperparameter tuning
- Cross-validation
"""

from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
from datetime import datetime


class DataSplitter:
    """Class for handling data splitting with stratification.

    This class provides methods to split data into training and test sets
    while preserving class distribution, which is critical for imbalanced datasets.

    Attributes:
        test_size (float): Proportion of data to use for testing.
        random_state (int): Random seed for reproducibility.
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """Initialize the DataSplitter.

        Args:
            test_size (float): Proportion of data for test set (default: 0.2).
            random_state (int): Random seed for reproducibility (default: 42).
        """
        self.test_size = test_size
        self.random_state = random_state

    def stratified_split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Perform stratified train-test split.

        Uses stratified sampling to preserve the proportion of samples for each class
        in both training and test sets. This is essential for imbalanced fraud detection.

        Args:
            X (pd.DataFrame): Features dataframe.
            y (pd.Series): Target variable.

        Returns:
            Tuple containing X_train, X_test, y_train, y_test.

        Example:
            >>> splitter = DataSplitter(test_size=0.2)
            >>> X_train, X_test, y_train, y_test = splitter.stratified_split(X, y)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,  # Preserve class distribution
        )

        return X_train, X_test, y_train, y_test

    def validate_split(self, y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """Validate that the split preserved class distribution.

        Compares class proportions between training and test sets to ensure
        stratification worked correctly.

        Args:
            y_train (pd.Series): Training target variable.
            y_test (pd.Series): Test target variable.

        Returns:
            Dict containing class distribution statistics.
        """
        train_dist = y_train.value_counts(normalize=True).to_dict()
        test_dist = y_test.value_counts(normalize=True).to_dict()

        validation_report = {
            "train_size": len(y_train),
            "test_size": len(y_test),
            "train_class_distribution": train_dist,
            "test_class_distribution": test_dist,
            "split_is_valid": self._check_distribution_similarity(
                train_dist, test_dist
            ),
        }

        return validation_report

    def _check_distribution_similarity(
        self,
        train_dist: Dict[int, float],
        test_dist: Dict[int, float],
        tolerance: float = 0.05,
    ) -> bool:
        """Check if train and test distributions are similar within tolerance.

        Args:
            train_dist (Dict): Training set class distribution.
            test_dist (Dict): Test set class distribution.
            tolerance (float): Maximum allowed difference (default: 0.05).

        Returns:
            bool: True if distributions are similar enough.
        """
        for class_label in train_dist.keys():
            diff = abs(train_dist[class_label] - test_dist.get(class_label, 0))
            if diff > tolerance:
                return False
        return True


class BaselineModel:
    """Class for training baseline Logistic Regression model.

    Logistic Regression serves as an interpretable baseline for fraud detection.
    It provides a benchmark against which to compare more complex models.

    Attributes:
        model (LogisticRegression): The trained logistic regression model.
        random_state (int): Random seed for reproducibility.
    """

    def __init__(self, random_state: int = 42, max_iter: int = 1000):
        """Initialize the BaselineModel.

        Args:
            random_state (int): Random seed for reproducibility.
            max_iter (int): Maximum iterations for convergence.
        """
        self.random_state = random_state
        self.max_iter = max_iter
        self.model = None

    def train(
        self, X_train: pd.DataFrame, y_train: pd.Series, class_weight: str = "balanced"
    ) -> LogisticRegression:
        """Train Logistic Regression model.

        Uses balanced class weights to handle class imbalance by automatically
        adjusting weights inversely proportional to class frequencies.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            class_weight (str): Weight balancing strategy (default: 'balanced').

        Returns:
            Trained LogisticRegression model.
        """
        self.model = LogisticRegression(
            random_state=self.random_state,
            max_iter=self.max_iter,
            class_weight=class_weight,
            n_jobs=-1,  # Use all available cores
        )

        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions.

        Args:
            X (pd.DataFrame): Features to predict.

        Returns:
            Array of predicted class labels.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate probability predictions.

        Args:
            X (pd.DataFrame): Features to predict.

        Returns:
            Array of prediction probabilities for each class.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        return self.model.predict_proba(X)

    def save_model(self, filepath: str) -> None:
        """Save trained model to disk.

        Args:
            filepath (str): Path where model will be saved.
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> LogisticRegression:
        """Load trained model from disk.

        Args:
            filepath (str): Path to saved model.

        Returns:
            Loaded model.
        """
        self.model = joblib.load(filepath)
        return self.model


class EnsembleModel:
    """Class for training ensemble models.

    Implements Random Forest, XGBoost, and LightGBM classifiers with
    hyperparameter tuning capabilities for improved fraud detection.

    Attributes:
        model_type (str): Type of ensemble model ('rf', 'xgb', 'lgb').
        model: The trained ensemble model.
        random_state (int): Random seed for reproducibility.
    """

    def __init__(self, model_type: str = "rf", random_state: int = 42):
        """Initialize the EnsembleModel.

        Args:
            model_type (str): Model type - 'rf' (Random Forest),
                            'xgb' (XGBoost), or 'lgb' (LightGBM).
            random_state (int): Random seed for reproducibility.

        Raises:
            ValueError: If model_type is not supported.
        """
        if model_type not in ["rf", "xgb", "lgb"]:
            raise ValueError("model_type must be 'rf', 'xgb', or 'lgb'")

        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.best_params = None

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        """Train the ensemble model.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            **kwargs: Additional model-specific parameters.

        Returns:
            Trained model.
        """
        if self.model_type == "rf":
            self.model = self._train_random_forest(X_train, y_train, **kwargs)
        elif self.model_type == "xgb":
            self.model = self._train_xgboost(X_train, y_train, **kwargs)
        elif self.model_type == "lgb":
            self.model = self._train_lightgbm(X_train, y_train, **kwargs)

        return self.model

    def _train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        class_weight: str = "balanced",
    ) -> RandomForestClassifier:
        """Train Random Forest classifier.

        Random Forest is an ensemble of decision trees that reduces overfitting
        through bagging and random feature selection.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            n_estimators (int): Number of trees in forest.
            max_depth (int): Maximum depth of trees.
            min_samples_split (int): Minimum samples to split node.
            class_weight (str): Weight balancing strategy.

        Returns:
            Trained RandomForestClassifier.
        """
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight=class_weight,
            random_state=self.random_state,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)
        return model

    def _train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        scale_pos_weight: Optional[float] = None,
    ) -> xgb.XGBClassifier:
        """Train XGBoost classifier.

        XGBoost uses gradient boosting to sequentially train trees, each
        correcting errors of previous ones.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            n_estimators (int): Number of boosting rounds.
            max_depth (int): Maximum tree depth.
            learning_rate (float): Learning rate (eta).
            scale_pos_weight (float): Weight for positive class.
                                     **IMPORTANT**: Do NOT calculate automatically
                                     when using SMOTE-balanced data. Only provide
                                     explicitly if training on imbalanced data.

        Returns:
            Trained XGBClassifier.
        """
        # DO NOT auto-calculate scale_pos_weight when using SMOTE
        # Let it default to 1 for balanced data
        model_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "random_state": self.random_state,
            "n_jobs": -1,
            "eval_metric": "logloss",
        }

        # Only add scale_pos_weight if explicitly provided
        if scale_pos_weight is not None:
            model_params["scale_pos_weight"] = scale_pos_weight

        model = xgb.XGBClassifier(**model_params)
        model.fit(X_train, y_train)
        return model

    def _train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_estimators: int = 100,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
    ) -> lgb.LGBMClassifier:
        """Train LightGBM classifier.

        LightGBM uses histogram-based gradient boosting for faster training
        and lower memory usage than traditional GBDT.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            n_estimators (int): Number of boosting rounds.
            max_depth (int): Maximum tree depth (-1 for no limit).
            learning_rate (float): Learning rate.
            num_leaves (int): Maximum number of leaves per tree.

        Returns:
            Trained LGBMClassifier.
        """
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)
        return model

    def hyperparameter_tuning(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Optional[Dict[str, List]] = None,
        cv: int = 5,
        scoring: str = "f1",
        search_type: str = "grid",
    ):
        """Perform hyperparameter tuning.

        Uses GridSearchCV or RandomizedSearchCV with cross-validation to find
        optimal hyperparameters for the model.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            param_grid (Dict): Parameter grid to search.
            cv (int): Number of cross-validation folds.
            scoring (str): Metric to optimize ('f1', 'roc_auc', 'precision', etc.).
            search_type (str): 'grid' or 'random' search.

        Returns:
            Best model from hyperparameter search.
        """
        # Define default parameter grids if not provided
        if param_grid is None:
            param_grid = self._get_default_param_grid()

        # Get base estimator
        if self.model_type == "rf":
            base_estimator = RandomForestClassifier(
                random_state=self.random_state, class_weight="balanced", n_jobs=-1
            )
        elif self.model_type == "xgb":
            # DO NOT use scale_pos_weight for SMOTE-balanced data
            base_estimator = xgb.XGBClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric="logloss",
            )
        elif self.model_type == "lgb":
            base_estimator = lgb.LGBMClassifier(
                random_state=self.random_state, class_weight="balanced", n_jobs=-1
            )

        # Perform search
        if search_type == "grid":
            search = GridSearchCV(
                base_estimator,
                param_grid,
                cv=StratifiedKFold(
                    n_splits=cv, shuffle=True, random_state=self.random_state
                ),
                scoring=scoring,
                n_jobs=-1,
                verbose=1,
            )
        else:  # random search
            search = RandomizedSearchCV(
                base_estimator,
                param_grid,
                n_iter=20,
                cv=StratifiedKFold(
                    n_splits=cv, shuffle=True, random_state=self.random_state
                ),
                scoring=scoring,
                n_jobs=-1,
                verbose=1,
                random_state=self.random_state,
            )

        search.fit(X_train, y_train)

        self.model = search.best_estimator_
        self.best_params = search.best_params_

        return self.model

    def _get_default_param_grid(self) -> Dict[str, List]:
        """Get default parameter grid for hyperparameter tuning.

        Returns:
            Dictionary with parameter ranges to search.
        """
        if self.model_type == "rf":
            return {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 20, 30, None],
                "min_samples_split": [2, 5, 10],
            }
        elif self.model_type == "xgb":
            return {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.3],
            }
        elif self.model_type == "lgb":
            return {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 10, 15],
                "learning_rate": [0.01, 0.1, 0.3],
                "num_leaves": [31, 50, 100],
            }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions.

        Args:
            X (pd.DataFrame): Features to predict.

        Returns:
            Array of predicted class labels.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        return self.model.predict(X)

    def predict_with_threshold(
        self, X: pd.DataFrame, threshold: float = 0.5
    ) -> np.ndarray:
        """Generate predictions with custom probability threshold.

        This is useful for imbalanced datasets where you want to adjust
        the trade-off between precision and recall.

        Args:
            X (pd.DataFrame): Features to predict.
            threshold (float): Probability threshold for positive class (default: 0.5).
                             Lower threshold = more fraud predictions (higher recall, lower precision)
                             Higher threshold = fewer fraud predictions (lower recall, higher precision)

        Returns:
            Array of predicted class labels.

        Example:
            >>> # More conservative (fewer false positives)
            >>> y_pred = model.predict_with_threshold(X_test, threshold=0.7)
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        y_proba = self.predict_proba(X)[:, 1]
        return (y_proba >= threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate probability predictions.

        Args:
            X (pd.DataFrame): Features to predict.

        Returns:
            Array of prediction probabilities for each class.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        return self.model.predict_proba(X)

    def save_model(self, filepath: str) -> None:
        """Save trained model to disk.

        Args:
            filepath (str): Path where model will be saved.
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save model with metadata
        model_data = {
            "model": self.model,
            "model_type": self.model_type,
            "best_params": self.best_params,
            "timestamp": datetime.now().isoformat(),
        }

        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load trained model from disk.

        Args:
            filepath (str): Path to saved model.

        Returns:
            Loaded model.
        """
        model_data = joblib.load(filepath)
        self.model = model_data["model"]
        self.model_type = model_data.get("model_type", self.model_type)
        self.best_params = model_data.get("best_params")
        return self.model


class CrossValidator:
    """Class for performing cross-validation.

    Implements stratified k-fold cross-validation to assess model performance
    and generalization ability across multiple data splits.

    Attributes:
        n_splits (int): Number of cross-validation folds.
        random_state (int): Random seed for reproducibility.
    """

    def __init__(self, n_splits: int = 5, random_state: int = 42):
        """Initialize the CrossValidator.

        Args:
            n_splits (int): Number of folds for cross-validation (default: 5).
            random_state (int): Random seed for reproducibility.
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )

    def cross_validate_model(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        scoring_functions: Dict[str, callable],
    ) -> Dict[str, Any]:
        """Perform cross-validation with multiple metrics.

        Trains and evaluates model on k different train-test splits,
        providing robust performance estimates with mean and standard deviation.

        Args:
            model: Sklearn-compatible model to evaluate.
            X (pd.DataFrame): Features.
            y (pd.Series): Target variable.
            scoring_functions (Dict): Dictionary of metric name to function.

        Returns:
            Dictionary with mean and std for each metric across folds.

        Example:
            >>> cv = CrossValidator(n_splits=5)
            >>> scoring = {'f1': f1_score, 'precision': precision_score}
            >>> results = cv.cross_validate_model(model, X, y, scoring)
        """
        scores = {metric: [] for metric in scoring_functions.keys()}

        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X, y), 1):
            # Split data
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]

            # Train model
            model.fit(X_train_fold, y_train_fold)

            # Predict
            y_pred = model.predict(X_val_fold)
            y_pred_proba = model.predict_proba(X_val_fold)[:, 1]

            # Calculate metrics
            for metric_name, metric_func in scoring_functions.items():
                # Some metrics need probabilities, others need predictions
                if "roc" in metric_name.lower() or "auc" in metric_name.lower():
                    score = metric_func(y_val_fold, y_pred_proba)
                else:
                    score = metric_func(y_val_fold, y_pred)
                scores[metric_name].append(score)

        # Calculate mean and std
        cv_results = {}
        for metric_name, metric_scores in scores.items():
            cv_results[f"{metric_name}_mean"] = np.mean(metric_scores)
            cv_results[f"{metric_name}_std"] = np.std(metric_scores)
            cv_results[f"{metric_name}_scores"] = metric_scores

        return cv_results
