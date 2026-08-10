"""Pydantic schemas for the loan-approval FastAPI endpoints."""

from pydantic import BaseModel, Field


class LoanApplicationRequest(BaseModel):
    person_age: float = Field(..., ge=18, le=120, description="Applicant age in years")
    person_gender: str = Field(..., description="male | female")
    person_education: str = Field(..., description="High School | Associate | Bachelor | Master | Doctorate")
    person_income: float = Field(..., ge=0, description="Annual income in USD")
    person_emp_exp: float = Field(..., ge=0, le=60, description="Employment experience in years")
    person_home_ownership: str = Field(..., description="RENT | OWN | MORTGAGE | OTHER")
    loan_amnt: float = Field(..., ge=100, description="Requested loan amount in USD")
    loan_intent: str = Field(
        ...,
        description="PERSONAL | EDUCATION | MEDICAL | VENTURE | HOMEIMPROVEMENT | DEBTCONSOLIDATION",
    )
    loan_int_rate: float = Field(..., ge=0, le=100, description="Interest rate (%)")
    loan_percent_income: float = Field(..., ge=0, le=1, description="Loan amount as fraction of income")
    cb_person_cred_hist_length: float = Field(..., ge=0, description="Credit history length in years")
    credit_score: float = Field(..., ge=300, le=850, description="Credit score")
    previous_loan_defaults_on_file: str = Field(..., description="Yes | No")

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    loan_status: int = Field(..., description="0 = Rejected, 1 = Approved")
    decision: str = Field(..., description="Human-readable decision")
    approval_probability: float = Field(..., description="Model probability of approval")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class MetricsResponse(BaseModel):
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    roc_auc: float
    mcc: float
    brier_score: float
