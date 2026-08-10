"""Shared pytest fixtures for the loan-approval test suite."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# Minimal synthetic dataset that mirrors the real loan dataset schema
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Return a small but valid DataFrame matching the loan dataset schema."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame(
        {
            "person_age": np.random.uniform(22, 60, n),
            "person_gender": np.random.choice(["male", "female"], n),
            "person_education": np.random.choice(["High School", "Bachelor", "Master"], n),
            "person_income": np.random.uniform(20000, 150000, n),
            "person_emp_exp": np.random.uniform(0, 20, n),
            "person_home_ownership": np.random.choice(["RENT", "OWN", "MORTGAGE"], n),
            "loan_amnt": np.random.uniform(1000, 30000, n),
            "loan_intent": np.random.choice(
                ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE"], n
            ),
            "loan_int_rate": np.random.uniform(5.0, 25.0, n),
            "loan_percent_income": np.random.uniform(0.01, 0.5, n),
            "cb_person_cred_hist_length": np.random.uniform(1, 20, n),
            "credit_score": np.random.uniform(400, 800, n),
            "previous_loan_defaults_on_file": np.random.choice(["Yes", "No"], n),
            "loan_status": np.random.randint(0, 2, n),
        }
    )


@pytest.fixture
def sample_df_no_target(sample_df: pd.DataFrame) -> pd.DataFrame:
    return sample_df.drop(columns=["loan_status"])


@pytest.fixture
def tmp_csv(sample_df: pd.DataFrame, tmp_path) -> str:
    path = str(tmp_path / "loan_data.csv")
    sample_df.to_csv(path, index=False)
    return path


@pytest.fixture
def trained_engine(sample_df):
    """Return a fitted FeatureEngineering + ModelTrainer pair."""
    from src.feature_engineering import FeatureEngineering
    from src.model_trainer import ModelTrainer

    fe = FeatureEngineering()
    X, y = fe.split_features_target(sample_df)
    X_scaled = fe.fit_transform_train(X)
    trainer = ModelTrainer()
    trainer.train(X_scaled, y)
    return fe, trainer
