"""
╔══════════════════════════════════════════════════════════════╗
║          CQRS PATTERN — COMMAND SIDE                         ║
║  Responsibility : WRITE / mutate the model store             ║
║  Triggers       : Run manually (python3 train_loan_model.py) ║
║  Output         : *.joblib artifacts + CSV reports           ║
╚══════════════════════════════════════════════════════════════╝

CQRS Separation
---------------
COMMAND (this file)          │  QUERY (app.py)
─────────────────────────────┼───────────────────────────────
Trains the model             │  Loads trained artifacts
Encodes & scales features    │  Encodes & scales user input
Writes *.joblib to disk      │  Reads *.joblib from disk
Writes CSV reports           │  Reads CSV reports
No HTTP / UI concerns        │  No training logic
Run once (or on demand)      │  Runs on every user request

Internally this file also demonstrates the Pipe and Filter pattern:
    Each function is a FILTER; its return value is the PIPE flowing into the next filter.

Pipeline flow:
    raw CSV path
        |-- [FILTER 1: load_data]             --> raw DataFrame
        |-- [FILTER 2: drop_missing]          --> cleaned DataFrame
        |-- [FILTER 3: remove_outliers]       --> outlier-free DataFrame
        |-- [FILTER 4: split_features_target] --> (X, y, cat_cols, num_cols)
        |-- [FILTER 5: split_datasets]        --> train / val / test splits
        |-- [FILTER 6: fit_label_encoders]    --> (X_train_encoded, encoders)
        |-- [FILTER 7: apply_label_encoders]  --> (X_val_encoded, X_test_encoded)
        |-- [FILTER 8: fit_scaler]            --> (X_train_scaled, scaler)
        |-- [FILTER 9: apply_scaler]          --> (X_val_scaled, X_test_scaled)
        |-- [FILTER 10: save_test_data]       --> test_data.csv  (side-effect)
        |-- [FILTER 11: train_model]          --> trained model
        |-- [FILTER 12: evaluate_model]       --> metrics dict
        |-- [FILTER 13: save_artifacts]       --> *.joblib files (side-effect)
        |-- [FILTER 14: save_results]         --> model_results.csv (side-effect)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)
import joblib
import warnings
warnings.filterwarnings('ignore')


# ===========================================================
# FILTER 1 — Load raw data from disk
# PIPE IN  : file path (str)
# PIPE OUT : raw DataFrame
# ===========================================================
def load_data(filepath: str) -> pd.DataFrame:
    print(f"[FILTER 1] Loading dataset from '{filepath}' ...")
    df = pd.read_csv(filepath)
    print(f"           Shape: {df.shape}")
    print(f"           Target distribution:\n{df['loan_status'].value_counts()}")
    return df


# ===========================================================
# FILTER 2 — Drop rows with missing values
# PIPE IN  : raw DataFrame
# PIPE OUT : DataFrame without NaN rows
# ===========================================================
def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[FILTER 2] Dropping missing values ...")
    print(f"           Missing counts before:\n{df.isnull().sum()}")
    df_clean = df.dropna()
    print(f"           Shape after drop: {df_clean.shape}")
    return df_clean


# ===========================================================
# FILTER 3 — Remove obvious outliers (age > 120)
# PIPE IN  : cleaned DataFrame
# PIPE OUT : outlier-free DataFrame
# ===========================================================
def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[FILTER 3] Removing age outliers (person_age > 120) ...")
    df_filtered = df[df['person_age'] <= 120]
    print(f"           Shape after outlier removal: {df_filtered.shape}")
    return df_filtered


# ===========================================================
# FILTER 4 — Separate features (X) from target (y)
# PIPE IN  : clean DataFrame
# PIPE OUT : (X, y, categorical_cols, numerical_cols)
# ===========================================================
def split_features_target(df: pd.DataFrame, target_col: str = 'loan_status'):
    print(f"\n[FILTER 4] Splitting features and target ('{target_col}') ...")
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols   = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    print(f"           Categorical columns : {categorical_cols}")
    print(f"           Numerical columns   : {numerical_cols}")
    return X, y, categorical_cols, numerical_cols


# ===========================================================
# FILTER 5 — Stratified train / validation / test split
# PIPE IN  : (X, y)
# PIPE OUT : (X_train_orig, X_val_orig, X_test_orig,
#             y_train, y_val, y_test)
# ===========================================================
def split_datasets(X: pd.DataFrame, y: pd.Series):
    print("\n[FILTER 5] Splitting into train / val / test sets ...")
    X_temp, X_test_orig, y_temp, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    X_train_orig, X_val_orig, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.111, random_state=42, stratify=y_temp
    )
    print(f"           Train : {X_train_orig.shape[0]} rows")
    print(f"           Val   : {X_val_orig.shape[0]} rows")
    print(f"           Test  : {X_test_orig.shape[0]} rows")
    return X_train_orig, X_val_orig, X_test_orig, y_train, y_val, y_test


# ===========================================================
# FILTER 6 — Fit LabelEncoders on TRAINING data only
# PIPE IN  : (X_train_orig, categorical_cols)
# PIPE OUT : (X_train_encoded DataFrame, label_encoders dict)
# ===========================================================
def fit_label_encoders(X_train_orig: pd.DataFrame, categorical_cols: list):
    print("\n[FILTER 6] Fitting LabelEncoders on training data only ...")
    label_encoders = {}
    X_train_encoded = X_train_orig.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        X_train_encoded[col] = le.fit_transform(X_train_orig[col].astype(str))
        label_encoders[col] = le
        print(f"           Encoded '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}")
    return X_train_encoded, label_encoders


# ===========================================================
# FILTER 7 — Apply fitted encoders to validation and test sets
# PIPE IN  : (X_val_orig, X_test_orig, label_encoders, categorical_cols)
# PIPE OUT : (X_val_encoded, X_test_encoded)
# ===========================================================
def apply_label_encoders(X_val_orig: pd.DataFrame, X_test_orig: pd.DataFrame,
                         label_encoders: dict, categorical_cols: list):
    print("\n[FILTER 7] Applying LabelEncoders to val and test sets ...")
    X_val_encoded  = X_val_orig.copy()
    X_test_encoded = X_test_orig.copy()
    for col in categorical_cols:
        X_val_encoded[col]  = label_encoders[col].transform(X_val_orig[col].astype(str))
        X_test_encoded[col] = label_encoders[col].transform(X_test_orig[col].astype(str))
    print("           Encoding applied.")
    return X_val_encoded, X_test_encoded


# ===========================================================
# FILTER 8 — Fit StandardScaler on TRAINING data only
# PIPE IN  : X_train_encoded DataFrame
# PIPE OUT : (X_train_scaled DataFrame, fitted scaler)
# ===========================================================
def fit_scaler(X_train_encoded: pd.DataFrame):
    print("\n[FILTER 8] Fitting StandardScaler on training data only ...")
    feature_names = list(X_train_encoded.columns)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_encoded),
        columns=feature_names
    )
    print("           Scaler fitted.")
    return X_train_scaled, scaler


# ===========================================================
# FILTER 9 — Apply fitted scaler to validation and test sets
# PIPE IN  : (X_val_encoded, X_test_encoded, scaler, feature_names)
# PIPE OUT : (X_val_scaled, X_test_scaled)  — both DataFrames
# ===========================================================
def apply_scaler(X_val_encoded: pd.DataFrame, X_test_encoded: pd.DataFrame,
                 scaler: StandardScaler, feature_names: list):
    print("\n[FILTER 9] Applying StandardScaler to val and test sets ...")
    X_val_scaled  = pd.DataFrame(scaler.transform(X_val_encoded),  columns=feature_names)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_encoded), columns=feature_names)
    print("           Scaling applied.")
    return X_val_scaled, X_test_scaled


# ===========================================================
# FILTER 10 — Persist test set with original categorical values
# PIPE IN  : (X_test_orig, y_test)
# PIPE OUT : path to saved CSV (side-effect: writes file)
# ===========================================================
def save_test_data(X_test_orig: pd.DataFrame, y_test: pd.Series,
                   output_path: str = 'test_data.csv') -> str:
    print(f"\n[FILTER 10] Saving original test data to '{output_path}' ...")
    test_data = X_test_orig.copy()
    test_data['loan_status'] = y_test.values
    test_data.to_csv(output_path, index=False)
    print(f"            Saved {len(test_data)} rows.")
    return output_path


# ===========================================================
# FILTER 11 — Train Logistic Regression model
# PIPE IN  : (X_train_scaled, y_train)
# PIPE OUT : fitted LogisticRegression model
# ===========================================================
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    print("\n[FILTER 11] Training Logistic Regression model ...")
    model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
    model.fit(X_train, y_train)
    print("            Model training complete.")
    return model


# ===========================================================
# FILTER 12 — Evaluate model on test set
# PIPE IN  : (model, X_test_scaled, y_test)
# PIPE OUT : metrics dict
# ===========================================================
def evaluate_model(model: LogisticRegression,
                   X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    print("\n[FILTER 12] Evaluating model on test set ...")
    y_pred = model.predict(X_test)
    metrics = {
        'Model'    : 'Logistic Regression',
        'Accuracy' : accuracy_score(y_test, y_pred),
        'AUC'      : roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
        'Precision': precision_score(y_test, y_pred),
        'Recall'   : recall_score(y_test, y_pred),
        'F1'       : f1_score(y_test, y_pred),
        'MCC'      : matthews_corrcoef(y_test, y_pred),
    }
    print(f"            Accuracy  : {metrics['Accuracy']:.4f}")
    print(f"            AUC       : {metrics['AUC']:.4f}")
    print(f"            Precision : {metrics['Precision']:.4f}")
    print(f"            Recall    : {metrics['Recall']:.4f}")
    print(f"            F1 Score  : {metrics['F1']:.4f}")
    print(f"            MCC       : {metrics['MCC']:.4f}")
    return metrics


# ===========================================================
# FILTER 13 — Persist model + preprocessor artifacts to disk
# PIPE IN  : (model, scaler, label_encoders, feature_names)
# PIPE OUT : list of saved file paths (side-effect: writes files)
# ===========================================================
def save_artifacts(model: LogisticRegression, scaler: StandardScaler,
                   label_encoders: dict, feature_names: list) -> list:
    print("\n[FILTER 13] Saving model and preprocessor artifacts ...")
    paths = {
        'logistic_regression.joblib': model,
        'scaler.joblib'             : scaler,
        'label_encoders.joblib'     : label_encoders,
        'feature_names.joblib'      : feature_names,
    }
    for filename, obj in paths.items():
        joblib.dump(obj, filename)
        print(f"            Saved: {filename}")
    return list(paths.keys())


# ===========================================================
# FILTER 14 — Save evaluation results to CSV
# PIPE IN  : list of metrics dicts
# PIPE OUT : path to saved results CSV (side-effect: writes file)
# ===========================================================
def save_results(results: list, output_path: str = 'model_results.csv') -> str:
    print(f"\n[FILTER 14] Saving evaluation results to '{output_path}' ...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print("\n" + "="*60)
    print("SUMMARY RESULTS")
    print("="*60)
    print(results_df.to_string(index=False))
    return output_path


# ===========================================================
# CQRS COMMAND HANDLER
# run_pipeline() is the single entry-point for the Command side.
# It orchestrates all filters and WRITES state to the model store
# (*.joblib files + CSV reports). The Query side (app.py) will
# later READ from that store — the two sides never share memory.
# ===========================================================
def run_pipeline(data_path: str = 'loan_data.csv'):
    print("="*60)
    print("LOAN APPROVAL — PIPE AND FILTER PIPELINE")
    print("="*60)

    # --- PIPE: file path -----------------------------------
    raw_df = load_data(data_path)                             # FILTER 1

    # --- PIPE: raw DataFrame -------------------------------
    clean_df = drop_missing(raw_df)                           # FILTER 2

    # --- PIPE: NaN-free DataFrame --------------------------
    filtered_df = remove_outliers(clean_df)                   # FILTER 3

    # --- PIPE: outlier-free DataFrame ----------------------
    X, y, cat_cols, num_cols = split_features_target(         # FILTER 4
        filtered_df, target_col='loan_status'
    )

    # --- PIPE: (X, y, col lists) ---------------------------
    X_train_orig, X_val_orig, X_test_orig, \
    y_train, y_val, y_test = split_datasets(X, y)             # FILTER 5

    # --- PIPE: original split DataFrames + Series ----------
    X_train_encoded, label_encoders = fit_label_encoders(     # FILTER 6
        X_train_orig, cat_cols
    )

    # --- PIPE: encoded train + fitted encoders -------------
    X_val_encoded, X_test_encoded = apply_label_encoders(     # FILTER 7
        X_val_orig, X_test_orig, label_encoders, cat_cols
    )

    # --- PIPE: encoded val & test DataFrames ---------------
    X_train_scaled, scaler = fit_scaler(X_train_encoded)      # FILTER 8

    # --- PIPE: scaled train DataFrame + fitted scaler ------
    feature_names = list(X.columns)
    X_val_scaled, X_test_scaled = apply_scaler(               # FILTER 9
        X_val_encoded, X_test_encoded, scaler, feature_names
    )

    # --- PIPE: scaled val & test DataFrames ----------------
    save_test_data(X_test_orig, y_test)                       # FILTER 10

    # --- PIPE: (X_train_scaled, y_train) -------------------
    model = train_model(X_train_scaled, y_train)              # FILTER 11

    # --- PIPE: trained model + (X_test_scaled, y_test) -----
    metrics = evaluate_model(model, X_test_scaled, y_test)    # FILTER 12

    # --- PIPE: metrics dict --------------------------------
    save_artifacts(model, scaler, label_encoders,             # FILTER 13
                   feature_names)

    # --- PIPE: list of metrics dicts -----------------------
    save_results([metrics])                                    # FILTER 14

    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*60)
    print("Model is ready for deployment!")


# ===========================================================
# Entry point
# ===========================================================
if __name__ == '__main__':
    run_pipeline(data_path='loan_data.csv')
