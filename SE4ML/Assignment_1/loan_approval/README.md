# Loan Status Classification Model

## a. Problem Statement

**Objective:** Predict loan approval status based on applicant demographics and financial information.

**Classification Type:** Binary Classification (0 = Not Approved, 1 = Approved)

**Target Variable:** `loan_status`

**Business Context:** Automated loan decision-making system for financial institutions.

---

## b. Dataset Description

**Dataset:** 45,000 loan application records (45,000 → 44,995 after preprocessing)

**Features:** 13 features + 1 target variable

**Target Distribution:**
- Class 0 (Not Approved): 77.8% (35,000 records)
- Class 1 (Approved): 22.2% (10,000 records)
- Imbalanced dataset

**Features:**

| Category | Feature | Type | Details |
|----------|---------|------|---------|
| Demographic | person_age | Numeric | 21-120 years (outliers removed) |
| Demographic | person_gender | Categorical | male/female |
| Demographic | person_education | Categorical | High School, Associate, Bachelor, Master, Doctorate |
| Demographic | person_income | Numeric | Annual income (~$12K - $600K) |
| Demographic | person_emp_exp | Numeric | Employment experience (years) |
| Home | person_home_ownership | Categorical | RENT, OWN, MORTGAGE, OTHER |
| Loan Details | loan_amnt | Numeric | Requested loan amount ($1K-$35K) |
| Loan Details | loan_intent | Categorical | PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION |
| Loan Details | loan_int_rate | Numeric | Interest rate (%) |
| Financial | loan_percent_income | Numeric | Loan as % of income |
| Financial | cb_person_cred_hist_length | Numeric | Credit history length (years) |
| Financial | credit_score | Numeric | Credit score (500-789) |
| Financial | previous_loan_defaults_on_file | Categorical | Yes/No |

**Preprocessing Steps:**
- Removed age outliers (person_age > 120)
- Applied StandardScaler to numeric features
- Applied LabelEncoder to categorical features
- Train-Test Split: 80-10-10 with stratification (36,000 train, 4,495 val, 4,500 test)

---

## c. Model Used

**Logistic Regression** - Linear probabilistic baseline model

### Software Architecture: Pipe and Filter Pattern

The training script (`model/train_loan_model.py`) is designed using the **Pipe and Filter** architectural pattern:

- Each **Filter** is a single-responsibility function performing one transformation step.
- The **Pipe** is the return value (data payload) flowing from one filter into the next.

```
raw CSV path
    │── [FILTER 1:  load_data]             ──► raw DataFrame
    │── [FILTER 2:  drop_missing]          ──► NaN-free DataFrame
    │── [FILTER 3:  remove_outliers]       ──► outlier-free DataFrame
    │── [FILTER 4:  split_features_target] ──► (X, y, cat_cols, num_cols)
    │── [FILTER 5:  split_datasets]        ──► train / val / test splits
    │── [FILTER 6:  fit_label_encoders]    ──► (X_train_encoded, encoders)
    │── [FILTER 7:  apply_label_encoders]  ──► (X_val_encoded, X_test_encoded)
    │── [FILTER 8:  fit_scaler]            ──► (X_train_scaled, scaler)
    │── [FILTER 9:  apply_scaler]          ──► (X_val_scaled, X_test_scaled)
    │── [FILTER 10: save_test_data]        ──► test_data.csv
    │── [FILTER 11: train_model]           ──► fitted model
    │── [FILTER 12: evaluate_model]        ──► metrics dict
    │── [FILTER 13: save_artifacts]        ──► *.joblib files
    └── [FILTER 14: save_results]          ──► model_results.csv
```

**Benefits of this pattern:**
- Each filter is independently testable and replaceable.
- Data flow is explicit and traceable.
- New preprocessing or evaluation steps can be inserted without modifying existing filters.


### Evaluation Metrics Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8927 | 0.9507 | 0.7630 | 0.7500 | 0.7564 | 0.6876 |

**Metrics Definitions:**
- **Accuracy:** Overall correctness (0-1, higher is better)
- **AUC:** Discrimination ability between classes (0-1, higher is better)
- **Precision:** True positives / All positive predictions (reduces false approvals)
- **Recall:** True positives / All actual positives (captures valid approvals)
- **F1 Score:** Harmonic mean of precision and recall
- **MCC:** Matthews Correlation Coefficient (-1 to +1, works well with imbalanced data)

---

## d. Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Moderate baseline (89.27% accuracy). Interpretable with good AUC (0.9507), but linear assumptions limit non-linear pattern capture. Suitable for simple baseline comparisons. |


### Key Learning - Imbalanced Data:
- Accuracy alone is misleading (77.8% vs 22.2% class distribution)
- AUC, F1, and MCC are more informative metrics for imbalanced datasets
- Ensemble methods (Random Forest, XGBoost) naturally handle class imbalance better
- Trade-off between precision and recall critical in loan approval decisions

---

## Repository Structure

```
project-folder/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── model/
    ├── train_loan_model.py         # Training script (Pipe and Filter architecture)
    ├── loan_data.csv               # Original dataset
    ├── logistic_regression.joblib  # Trained model
    ├── scaler.joblib               # StandardScaler
    ├── label_encoders.joblib       # Categorical encoders
    ├── feature_names.joblib        # Feature names
    ├── test_data.csv               # Test dataset (original categorical values)
    └── model_results.csv           # Evaluation metrics
```

## How to Use

**Train Models (Optional):**
```bash
cd model/
python3 train_loan_model.py
```

**Run Streamlit App:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

**Access Live App:**
- Local: `http://localhost:8501`
- Cloud: Deployed on Streamlit Community Cloud
 `https://powarsg-ml-classification-models.streamlit.app/`

## Streamlit App Features

✅ Model selection dropdown (6 models)  
✅ CSV upload for test data  
✅ Performance metrics display  
✅ Confusion matrix visualization  
✅ Classification report  
✅ Model comparison charts  
✅ Sample predictions  

