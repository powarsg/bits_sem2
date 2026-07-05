"""
Loan Status Classification Model
Trains ML model with proper preprocessing pipeline
Saves test_data.csv with ORIGINAL (non-encoded) categorical values for re-upload
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

# Load dataset
print("Loading loan_data.csv...")
df = pd.read_csv('loan_data.csv')

print(f"Dataset shape: {df.shape}")
print(f"Target variable distribution:\n{df['loan_status'].value_counts()}")

# Data Preprocessing
print("\n" + "="*60)
print("DATA PREPROCESSING")
print("="*60)

# Handle missing values
print(f"\nMissing values before handling:\n{df.isnull().sum()}")
df = df.dropna()
print(f"\nDataset shape after dropping NAs: {df.shape}")

# Remove age outliers
df = df[df['person_age'] <= 120]
print(f"Dataset shape after removing age outliers: {df.shape}")

# Separate features and target
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

print(f"\nCategorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# STEP 1: Train-Test-Val Split FIRST (before ANY preprocessing)
X_temp, X_test_orig, y_temp, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)
X_train_orig, X_val_orig, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.111, random_state=42, stratify=y_temp
)

print(f"\nTrain set size: {X_train_orig.shape[0]}")
print(f"Val set size: {X_val_orig.shape[0]}")
print(f"Test set size: {X_test_orig.shape[0]}")

# STEP 2: Fit encoders on TRAINING data ONLY
print("\nFitting LabelEncoders on TRAINING data only...")
label_encoders = {}
X_train_encoded = X_train_orig.copy()
for col in categorical_cols:
    le = LabelEncoder()
    X_train_encoded[col] = le.fit_transform(X_train_orig[col].astype(str))
    label_encoders[col] = le
    print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# STEP 3: Apply encoders to validation and test
X_val_encoded = X_val_orig.copy()
X_test_encoded = X_test_orig.copy()
for col in categorical_cols:
    X_val_encoded[col] = label_encoders[col].transform(X_val_orig[col].astype(str))
    X_test_encoded[col] = label_encoders[col].transform(X_test_orig[col].astype(str))

# STEP 4: Fit scaler on TRAINING data ONLY
print("\nFitting StandardScaler on TRAINING data only...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_val_scaled = scaler.transform(X_val_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

# Convert to DataFrames
X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
X_val = pd.DataFrame(X_val_scaled, columns=X.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

# STEP 5: Save test_data.csv with ORIGINAL categorical values
print("\nSaving test_data.csv with ORIGINAL categorical values...")
test_data = X_test_orig.copy()
test_data['loan_status'] = y_test.values
test_data.to_csv('test_data.csv', index=False)
print("✓ Test data saved with original categorical values for re-upload")

# ============================================================
# MODEL TRAINING AND EVALUATION
# ============================================================

results = []

# 1. Logistic Regression
print("\n" + "="*60)
print("1. LOGISTIC REGRESSION")
print("="*60)
lr = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
prec_lr = precision_score(y_test, y_pred_lr)
rec_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
mcc_lr = matthews_corrcoef(y_test, y_pred_lr)

print(f"Accuracy:  {acc_lr:.4f}")
print(f"AUC:       {auc_lr:.4f}")
print(f"Precision: {prec_lr:.4f}")
print(f"Recall:    {rec_lr:.4f}")
print(f"F1 Score:  {f1_lr:.4f}")
print(f"MCC:       {mcc_lr:.4f}")
results.append({'Model': 'Logistic Regression', 'Accuracy': acc_lr, 'AUC': auc_lr, 'Precision': prec_lr, 'Recall': rec_lr, 'F1': f1_lr, 'MCC': mcc_lr})
joblib.dump(lr, 'logistic_regression.joblib')
print("✓ Model saved")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('model_results.csv', index=False)
print("\n✓ Results saved to model_results.csv")

# Save preprocessors
feature_names = list(X.columns)
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(label_encoders, 'label_encoders.joblib')
joblib.dump(feature_names, 'feature_names.joblib')
print("✓ Scaler, label encoders, and feature names saved")

# Summary
print("\n" + "="*60)
print("SUMMARY RESULTS")
print("="*60)
print(results_df.to_string(index=False))

print("\n" + "="*60)
print("✓ TRAINING COMPLETED SUCCESSFULLY")
print("="*60)
print("Model is ready for deployment!")
