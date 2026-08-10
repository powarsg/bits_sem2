"""Integration tests for the FastAPI endpoints using TestClient."""

import os
import json
import joblib
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def sample_df_module():
    """Module-scoped version of sample_df for API tests."""
    import numpy as np
    import pandas as pd

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
            "loan_intent": np.random.choice(["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE"], n),
            "loan_int_rate": np.random.uniform(5.0, 25.0, n),
            "loan_percent_income": np.random.uniform(0.01, 0.5, n),
            "cb_person_cred_hist_length": np.random.uniform(1, 20, n),
            "credit_score": np.random.uniform(400, 800, n),
            "previous_loan_defaults_on_file": np.random.choice(["Yes", "No"], n),
            "loan_status": np.random.randint(0, 2, n),
        }
    )


@pytest.fixture(scope="module")
def client_with_model(tmp_path_factory, sample_df_module):
    sample_df = sample_df_module
    """Start the API with a real trained model loaded from a temp directory."""
    import sys

    model_dir = str(tmp_path_factory.mktemp("model_artifacts"))

    from src.feature_engineering import FeatureEngineering
    from src.model_trainer import ModelTrainer

    fe = FeatureEngineering()
    X, y = fe.split_features_target(sample_df)
    X_scaled = fe.fit_transform_train(X)
    trainer = ModelTrainer()
    trainer.train(X_scaled, y)
    test_metrics = trainer.evaluate(X_scaled, y)

    joblib.dump(trainer.model, os.path.join(model_dir, "logistic_regression.joblib"))
    joblib.dump(fe.scaler, os.path.join(model_dir, "scaler.joblib"))
    joblib.dump(fe.label_encoders, os.path.join(model_dir, "label_encoders.joblib"))
    joblib.dump(fe.feature_names, os.path.join(model_dir, "feature_names.joblib"))
    with open(os.path.join(model_dir, "model_metrics.json"), "w") as f:
        json.dump(test_metrics, f)

    os.environ["MODEL_DIR"] = model_dir

    # Reload api.main so it picks up the new MODEL_DIR
    if "api.main" in sys.modules:
        del sys.modules["api.main"]

    from api.main import app

    with TestClient(app) as client:
        yield client

    del os.environ["MODEL_DIR"]


SAMPLE_RECORD = {
    "person_age": 30,
    "person_gender": "male",
    "person_education": "Bachelor",
    "person_income": 60000,
    "person_emp_exp": 5,
    "person_home_ownership": "RENT",
    "loan_amnt": 10000,
    "loan_intent": "PERSONAL",
    "loan_int_rate": 11.5,
    "loan_percent_income": 0.17,
    "cb_person_cred_hist_length": 4,
    "credit_score": 680,
    "previous_loan_defaults_on_file": "No",
}


class TestHealthEndpoint:
    def test_health_returns_200(self, client_with_model):
        response = client_with_model.get("/health")
        assert response.status_code == 200

    def test_health_model_loaded(self, client_with_model):
        response = client_with_model.get("/health")
        data = response.json()
        assert data["model_loaded"] is True
        assert data["status"] == "ok"


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client_with_model):
        response = client_with_model.get("/metrics")
        assert response.status_code == 200

    def test_metrics_contains_accuracy(self, client_with_model):
        data = client_with_model.get("/metrics").json()
        assert "accuracy" in data
        assert 0.0 <= data["accuracy"] <= 1.0


class TestPredictEndpoint:
    def test_predict_returns_200(self, client_with_model):
        response = client_with_model.post("/predict", json=SAMPLE_RECORD)
        assert response.status_code == 200

    def test_predict_response_schema(self, client_with_model):
        data = client_with_model.post("/predict", json=SAMPLE_RECORD).json()
        assert "loan_status" in data
        assert "decision" in data
        assert "approval_probability" in data

    def test_predict_loan_status_binary(self, client_with_model):
        data = client_with_model.post("/predict", json=SAMPLE_RECORD).json()
        assert data["loan_status"] in (0, 1)

    def test_predict_invalid_age_returns_422(self, client_with_model):
        bad_record = dict(SAMPLE_RECORD)
        bad_record["person_age"] = 10  # below minimum of 18
        response = client_with_model.post("/predict", json=bad_record)
        assert response.status_code == 422

    def test_predict_missing_field_returns_422(self, client_with_model):
        incomplete = {k: v for k, v in SAMPLE_RECORD.items() if k != "credit_score"}
        response = client_with_model.post("/predict", json=incomplete)
        assert response.status_code == 422


class TestBatchPredictEndpoint:
    def test_batch_predict_returns_list(self, client_with_model):
        batch = [SAMPLE_RECORD, SAMPLE_RECORD]
        data = client_with_model.post("/predict/batch", json=batch).json()
        assert isinstance(data, list)
        assert len(data) == 2

    def test_batch_predict_empty_list_returns_422(self, client_with_model):
        response = client_with_model.post("/predict/batch", json=[])
        assert response.status_code == 422
