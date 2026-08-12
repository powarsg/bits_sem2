# Loan Approval Prediction v2 — AIMLCZG546 Assignment II

**Group 216** | BITS WILP SE4ML

| # | BITS ID | Name |
|---|---------|------|
| 1 | 2025aa05444@wilp.bits-pilani.ac.in | PRASAD SHIVAJI KULKARNI |
| 2 | 2025aa05387@wilp.bits-pilani.ac.in | SHELAR SACHIN KRISHNA |
| 3 | 2025aa05421@wilp.bits-pilani.ac.in | POWAR SAGAR GANPATI |
| 4 | 2025aa05326@wilp.bits-pilani.ac.in | SUJEET KUMAR YADAV |

---

## Setup

```bash
cd Assignment_2/loan_approval_v2
pip install -e ".[dev]"          # installs src/ and api/ as proper packages
cp .env.example .env             # edit paths if needed
```

---

## Train the model

```bash
python3 scripts/train.py
```

Reads `data_path` from `.env` (default: `data/loan_data.csv` — bundled in the repo).  
Saves artifacts to `model_artifacts/` and metrics to `model_artifacts/model_metrics.json`.

### Docker

```bash
docker build -t loan-approval .
docker run -p 8000:8000 -v $(pwd)/model_artifacts:/app/model_artifacts loan-approval
```

---

## Start the API server

```bash
uvicorn api.main:app --reload --port 8000
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness + readiness probe |
| `/metrics` | GET | Model evaluation metrics |
| `/predict` | POST | Single applicant prediction |
| `/predict/batch` | POST | Batch predictions |
| `/docs` | GET | Interactive Swagger UI |

Example request:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"person_age":30,"person_gender":"male","person_education":"Bachelor",
       "person_income":60000,"person_emp_exp":5,"person_home_ownership":"RENT",
       "loan_amnt":10000,"loan_intent":"PERSONAL","loan_int_rate":11.5,
       "loan_percent_income":0.17,"cb_person_cred_hist_length":4,
       "credit_score":680,"previous_loan_defaults_on_file":"No"}'
```

---

## Run tests

```bash
# All tests
python3 -m pytest tests/ -v

# Unit tests only
python3 -m pytest tests/unit/ -v

# Integration tests only
python3 -m pytest tests/integration/ -v
```

| Test file | Suite | Type | Tests |
|-----------|-------|------|-------|
| `tests/unit/test_data_ingestion.py` | unit | Unit | 9 |
| `tests/unit/test_feature_engineering.py` | unit | Unit | 10 |
| `tests/unit/test_data_quality.py` | unit | Data validation | 12 |
| `tests/unit/test_model_training.py` | unit | ML-specific | 7 |
| `tests/unit/test_inference.py` | unit | ML-specific (directional) | 9 |
| `tests/integration/test_api.py` | integration | API / end-to-end | 11 |
| **Total** | | | **62** |

---

## Run linter

```bash
python3 -m flake8 src/ api/ scripts/ --max-line-length=120
```

---

## File locations

```
loan_approval_v2/
  pyproject.toml              # installable package — eliminates sys.path hacks
  Dockerfile                  # containerised deployment
  .env.example                # environment variable template
  .gitignore
  data/
    loan_data.csv             # 45K-row dataset 
  scripts/
    train.py                  # training entry point (separate from serving)
  src/
    config.py                 # centralised settings (pydantic-settings)
    logger.py                 # centralised logging factory
    data_ingestion.py         # DataIngestion class
    feature_engineering.py    # FeatureEngineering class
    model_trainer.py          # ModelTrainer class
    inference.py              # InferenceEngine class
    data_quality.py           # DataQualityChecker class
  api/
    main.py                   # FastAPI application
    schemas.py                # Pydantic request/response schemas
  tests/
    conftest.py               # shared fixtures (used by both suites)
    unit/
      test_data_ingestion.py
      test_feature_engineering.py
      test_data_quality.py
      test_model_training.py
      test_inference.py
    integration/
      test_api.py
  model_artifacts/            # saved model + preprocessors (generated; gitignored)
  logs/                       # rotating log files (generated; gitignored)
```

---

## Model results (45K rows, 80/10/10 split)

| Metric | Value |
|--------|-------|
| Accuracy | 0.8927 |
| F1 Score | 0.7564 |
| ROC-AUC | 0.9507 |
| Brier Score | 0.0748 |
