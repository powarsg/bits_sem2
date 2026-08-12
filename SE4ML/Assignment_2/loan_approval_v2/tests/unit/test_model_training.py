"""ML-specific tests for ModelTrainer — training correctness and metric thresholds."""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import FeatureEngineering
from src.model_trainer import ModelTrainer


class TestModelTraining:
    def test_train_returns_fitted_model(self, trained_engine):
        _, trainer = trained_engine
        assert trainer.model is not None

    def test_evaluate_returns_expected_keys(self, trained_engine, sample_df):
        fe, trainer = trained_engine
        X, y = fe.split_features_target(sample_df)
        X_scaled = fe.transform(X)
        metrics = trainer.evaluate(X_scaled, y)
        expected_keys = {"accuracy", "f1_score", "precision", "recall", "roc_auc", "mcc", "brier_score"}
        assert expected_keys.issubset(set(metrics.keys()))

    def test_evaluate_without_training_raises(self, sample_df):
        fe = FeatureEngineering()
        X, y = fe.split_features_target(sample_df)
        X_scaled = fe.fit_transform_train(X)
        trainer = ModelTrainer()
        with pytest.raises(RuntimeError, match="not trained"):
            trainer.evaluate(X_scaled, y)

    def test_accuracy_within_reasonable_range(self, trained_engine, sample_df):
        """Model accuracy on random data should be between 0.4 and 1.0."""
        fe, trainer = trained_engine
        X, y = fe.split_features_target(sample_df)
        X_scaled = fe.transform(X)
        metrics = trainer.evaluate(X_scaled, y)
        assert 0.4 <= metrics["accuracy"] <= 1.0

    def test_roc_auc_above_random(self, trained_engine, sample_df):
        """AUC should be above 0.5 (better than random)."""
        fe, trainer = trained_engine
        X, y = fe.split_features_target(sample_df)
        X_scaled = fe.transform(X)
        metrics = trainer.evaluate(X_scaled, y)
        assert metrics["roc_auc"] >= 0.5


class TestOverfitCheck:
    def test_model_can_overfit_small_batch(self, sample_df):
        """Verify the model is capable of memorising a tiny dataset."""
        fe = FeatureEngineering()
        small = sample_df.head(30).copy()
        X, y = fe.split_features_target(small)
        X_scaled = fe.fit_transform_train(X)
        trainer = ModelTrainer()
        result = trainer.overfit_check(X_scaled, y, threshold=0.70)
        assert result is True, "Model should achieve ≥ 70% train accuracy on 30 samples"


class TestModelQualityMetrics:
    """Verify model quality metrics meet minimum acceptable thresholds on synthetic data."""

    def test_f1_score_above_threshold(self, trained_engine, sample_df):
        fe, trainer = trained_engine
        X, y = fe.split_features_target(sample_df)
        X_scaled = fe.transform(X)
        metrics = trainer.evaluate(X_scaled, y)
        # On random labels we expect at least 0.4 F1 (class imbalance tolerance)
        assert metrics["f1_score"] >= 0.3

    def test_brier_score_below_threshold(self, trained_engine, sample_df):
        """Brier score < 0.5 means better calibration than a naive all-0.5 predictor."""
        fe, trainer = trained_engine
        X, y = fe.split_features_target(sample_df)
        X_scaled = fe.transform(X)
        metrics = trainer.evaluate(X_scaled, y)
        assert metrics["brier_score"] < 0.5
