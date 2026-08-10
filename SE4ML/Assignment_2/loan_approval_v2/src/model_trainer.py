"""Model training module — trains and evaluates the Logistic Regression model."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    matthews_corrcoef,
    brier_score_loss,
)

from .logger import get_logger

logger = get_logger("model_trainer")


class ModelTrainer:
    """Trains a Logistic Regression classifier and exposes evaluation metrics."""

    def __init__(self, max_iter: int = 1000, random_state: int = 42) -> None:
        self.max_iter = max_iter
        self.random_state = random_state
        self.model: LogisticRegression | None = None
        logger.info(
            "ModelTrainer initialised — max_iter=%d, random_state=%d",
            max_iter,
            random_state,
        )

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
        """Fit the model and return it."""
        logger.info("Starting model training on %d samples", len(X_train))
        self.model = LogisticRegression(
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver="lbfgs",
        )
        self.model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, self.model.predict(X_train))
        logger.info("Training complete — train accuracy: %.4f", train_acc)
        return self.model

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Return a dict of model-quality metrics on the supplied test set."""
        if self.model is None:
            logger.error("Model not trained — call train() first")
            raise RuntimeError("Model not trained")

        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "f1_score": round(f1_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall": round(recall_score(y_test, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
            "mcc": round(float(matthews_corrcoef(y_test, y_pred)), 4),
            # Brier score measures calibration (lower = better)
            "brier_score": round(float(brier_score_loss(y_test, y_prob)), 4),
        }

        logger.info(
            "Evaluation — accuracy=%.4f  f1=%.4f  auc=%.4f  brier=%.4f",
            metrics["accuracy"],
            metrics["f1_score"],
            metrics["roc_auc"],
            metrics["brier_score"],
        )

        # Warn when calibration is poor
        if metrics["brier_score"] > 0.25:
            logger.warning(
                "Brier score %.4f > 0.25 — consider probability calibration",
                metrics["brier_score"],
            )

        return metrics

    def overfit_check(
        self, X_small: pd.DataFrame, y_small: pd.Series, threshold: float = 0.95
    ) -> bool:
        """Train on a tiny batch and verify the model can memorise it.

        Used in tests to confirm the learner is capable of fitting signal.
        Returns True when training accuracy meets the threshold.
        """
        logger.debug(
            "Overfit check on %d samples (threshold=%.2f)", len(X_small), threshold
        )
        tmp = LogisticRegression(max_iter=5000, random_state=0)
        tmp.fit(X_small, y_small)
        acc = accuracy_score(y_small, tmp.predict(X_small))
        logger.debug("Overfit check accuracy: %.4f", acc)
        return bool(acc >= threshold)
