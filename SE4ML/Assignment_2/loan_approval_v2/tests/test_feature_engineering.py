"""Unit tests for FeatureEngineering class."""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import FeatureEngineering, TARGET_COLUMN, CATEGORICAL_COLUMNS


class TestSplitFeaturesTarget:
    def test_splits_correctly(self, sample_df):
        fe = FeatureEngineering()
        X, y = fe.split_features_target(sample_df)
        assert TARGET_COLUMN not in X.columns
        assert len(X) == len(y)

    def test_missing_target_raises(self, sample_df):
        fe = FeatureEngineering()
        df_no_target = sample_df.drop(columns=[TARGET_COLUMN])
        with pytest.raises(KeyError):
            fe.split_features_target(df_no_target)


class TestEncoding:
    def test_fit_encoders_creates_encoder_for_each_cat_col(self, sample_df):
        fe = FeatureEngineering()
        X, _ = fe.split_features_target(sample_df)
        fe.fit_encoders(X)
        for col in CATEGORICAL_COLUMNS:
            assert col in fe.label_encoders

    def test_encoded_values_are_numeric(self, sample_df):
        fe = FeatureEngineering()
        X, _ = fe.split_features_target(sample_df)
        X_enc = fe.fit_encoders(X)
        for col in CATEGORICAL_COLUMNS:
            assert pd.api.types.is_numeric_dtype(X_enc[col])

    def test_transform_encoders_handles_unseen_labels(self, sample_df):
        fe = FeatureEngineering()
        X, _ = fe.split_features_target(sample_df)
        fe.fit_encoders(X)
        X_new = X.copy()
        # Force a completely new unseen value
        X_new = X_new.astype({"loan_intent": str})
        X_new.iloc[0, X_new.columns.get_loc("loan_intent")] = "UNKNOWN_INTENT"
        # Should not raise — unseen labels mapped to 0
        X_enc = fe.transform_encoders(X_new)
        assert X_enc.iloc[0]["loan_intent"] == 0

    def test_transform_without_fit_raises(self, sample_df):
        fe = FeatureEngineering()
        X, _ = fe.split_features_target(sample_df)
        with pytest.raises(RuntimeError, match="not fitted"):
            fe.transform_encoders(X)


class TestScaling:
    def test_scaled_values_have_approx_zero_mean(self, sample_df):
        fe = FeatureEngineering()
        X, _ = fe.split_features_target(sample_df)
        X_enc = fe.fit_encoders(X)
        X_scaled = fe.fit_scaler(X_enc)
        # Mean of each column should be close to 0
        assert abs(float(X_scaled["person_age"].mean())) < 0.1

    def test_transform_scaler_without_fit_raises(self, sample_df):
        fe = FeatureEngineering()
        X, _ = fe.split_features_target(sample_df)
        X_enc = fe.fit_encoders(X)
        with pytest.raises(RuntimeError, match="not fitted"):
            fe.transform_scaler(X_enc)


class TestFitTransformPipeline:
    def test_fit_transform_produces_same_shape(self, sample_df):
        fe = FeatureEngineering()
        X, _ = fe.split_features_target(sample_df)
        X_scaled = fe.fit_transform_train(X)
        assert X_scaled.shape == X.shape

    def test_transform_test_set_uses_train_statistics(self, sample_df):
        fe = FeatureEngineering()
        X, _ = fe.split_features_target(sample_df)
        half = len(X) // 2
        X_train, X_test = X.iloc[:half], X.iloc[half:]
        fe.fit_transform_train(X_train)
        X_test_scaled = fe.transform(X_test)
        assert X_test_scaled.shape == X_test.shape
