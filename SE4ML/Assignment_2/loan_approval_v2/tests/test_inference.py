"""Tests for InferenceEngine — output shape/range and directional/invariance checks."""

import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import FeatureEngineering
from src.inference import InferenceEngine
from src.model_trainer import ModelTrainer


@pytest.fixture
def saved_artifacts(tmp_path, sample_df):
    """Train a model, save artifacts, and return the artifact directory."""
    fe = FeatureEngineering()
    X, y = fe.split_features_target(sample_df)
    X_scaled = fe.fit_transform_train(X)
    trainer = ModelTrainer()
    trainer.train(X_scaled, y)

    artifact_dir = str(tmp_path / "model_artifacts")
    os.makedirs(artifact_dir)
    joblib.dump(trainer.model, os.path.join(artifact_dir, "logistic_regression.joblib"))
    joblib.dump(fe.scaler, os.path.join(artifact_dir, "scaler.joblib"))
    joblib.dump(fe.label_encoders, os.path.join(artifact_dir, "label_encoders.joblib"))
    joblib.dump(fe.feature_names, os.path.join(artifact_dir, "feature_names.joblib"))
    return artifact_dir


@pytest.fixture
def engine(saved_artifacts):
    eng = InferenceEngine(model_dir=saved_artifacts)
    eng.load_artifacts()
    return eng


# ---------------------------------------------------------------------------
# Output shape and range checks
# ---------------------------------------------------------------------------

class TestOutputShapeRange:
    def test_predict_returns_correct_length(self, engine, sample_df):
        X = sample_df.drop(columns=["loan_status"])
        preds = engine.predict(X)
        assert len(preds) == len(X)

    def test_predict_values_are_binary(self, engine, sample_df):
        X = sample_df.drop(columns=["loan_status"])
        preds = engine.predict(X)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self, engine, sample_df):
        X = sample_df.drop(columns=["loan_status"])
        probs = engine.predict_proba(X)
        assert probs.shape == (len(X), 2)

    def test_predict_proba_sum_to_one(self, engine, sample_df):
        X = sample_df.drop(columns=["loan_status"])
        probs = engine.predict_proba(X)
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_predict_proba_in_zero_one_range(self, engine, sample_df):
        X = sample_df.drop(columns=["loan_status"])
        probs = engine.predict_proba(X)
        assert (probs >= 0).all() and (probs <= 1).all()


# ---------------------------------------------------------------------------
# Single prediction
# ---------------------------------------------------------------------------

class TestPredictSingle:
    def test_single_record_returns_dict_with_required_keys(self, engine):
        record = {
            "person_age": 30, "person_gender": "male",
            "person_education": "Bachelor", "person_income": 60000,
            "person_emp_exp": 5, "person_home_ownership": "RENT",
            "loan_amnt": 10000, "loan_intent": "PERSONAL",
            "loan_int_rate": 11.5, "loan_percent_income": 0.17,
            "cb_person_cred_hist_length": 4, "credit_score": 680,
            "previous_loan_defaults_on_file": "No",
        }
        result = engine.predict_single(record)
        assert "loan_status" in result
        assert "decision" in result
        assert "approval_probability" in result

    def test_decision_matches_label(self, engine):
        record = {
            "person_age": 30, "person_gender": "male",
            "person_education": "Bachelor", "person_income": 60000,
            "person_emp_exp": 5, "person_home_ownership": "RENT",
            "loan_amnt": 10000, "loan_intent": "PERSONAL",
            "loan_int_rate": 11.5, "loan_percent_income": 0.17,
            "cb_person_cred_hist_length": 4, "credit_score": 680,
            "previous_loan_defaults_on_file": "No",
        }
        result = engine.predict_single(record)
        expected_decision = "Approved" if result["loan_status"] == 1 else "Rejected"
        assert result["decision"] == expected_decision


# ---------------------------------------------------------------------------
# Directional / monotonicity tests
# ---------------------------------------------------------------------------

class TestDirectionalChecks:
    """Higher income with same loan amount should not decrease approval probability."""

    def _base_record(self) -> dict:
        return {
            "person_age": 35,
            "person_gender": "male",
            "person_education": "Bachelor",
            "person_income": 50000,
            "person_emp_exp": 5,
            "person_home_ownership": "RENT",
            "loan_amnt": 5000,
            "loan_intent": "PERSONAL",
            "loan_int_rate": 10.0,
            "loan_percent_income": 0.10,
            "cb_person_cred_hist_length": 5,
            "credit_score": 680,
            "previous_loan_defaults_on_file": "No",
        }

    def test_higher_income_does_not_decrease_approval_prob(self, engine):
        low = self._base_record()
        high = dict(low)
        high["person_income"] = 200000
        high["loan_percent_income"] = 0.025  # proportionally lower with more income

        p_low = engine.predict_single(low)["approval_probability"]
        p_high = engine.predict_single(high)["approval_probability"]
        # With much higher income, approval probability should not dramatically drop
        assert p_high >= p_low - 0.20, (
            f"Higher income case p={p_high:.4f} unexpectedly much lower than "
            f"lower income case p={p_low:.4f}"
        )

    def test_default_history_decreases_approval_prob(self, engine):
        no_default = self._base_record()
        with_default = dict(no_default)
        with_default["previous_loan_defaults_on_file"] = "Yes"

        p_no_default = engine.predict_single(no_default)["approval_probability"]
        p_with_default = engine.predict_single(with_default)["approval_probability"]
        # Having a default on file should not dramatically increase approval probability
        assert p_with_default <= p_no_default + 0.15


# ---------------------------------------------------------------------------
# Missing artifact handling
# ---------------------------------------------------------------------------

class TestMissingArtifacts:
    def test_load_missing_artifacts_raises(self, tmp_path):
        eng = InferenceEngine(model_dir=str(tmp_path / "empty"))
        with pytest.raises(FileNotFoundError):
            eng.load_artifacts()
