"""Production training script — runs the full training pipeline and saves artifacts."""

import json
import os
import sys

import joblib
from sklearn.model_selection import train_test_split

from src.config import settings
from src.data_ingestion import DataIngestion
from src.data_quality import DataQualityChecker
from src.feature_engineering import FeatureEngineering
from src.logger import get_logger
from src.model_trainer import ModelTrainer

logger = get_logger("train")

DATA_PATH = settings.data_path
MODEL_DIR = settings.model_dir
os.makedirs(MODEL_DIR, exist_ok=True)


def main() -> None:
    logger.info("=" * 60)
    logger.info("LOAN APPROVAL — ASSIGNMENT II — PRODUCTION TRAINING")
    logger.info("=" * 60)

    # 1. Ingest
    ingestion = DataIngestion(DATA_PATH)
    df = ingestion.ingest()

    # Data quality gate — warn on issues, abort only on missing/schema failures
    checker = DataQualityChecker()
    report = checker.full_report(df)
    schema_ok = report["missing_values"]["passed"] and report["schema"]["passed"]
    if not schema_ok:
        logger.error("Critical data quality check FAILED — aborting training")
        sys.exit(1)
    if not report["overall_passed"]:
        logger.warning("Non-critical data quality issues detected — continuing")

    # 3. Feature engineering
    fe = FeatureEngineering()
    X, y = fe.split_features_target(df)

    # Stratified splits: 80% train, 10% val, 10% test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.111, random_state=42, stratify=y_temp)
    logger.info("Split — train=%d  val=%d  test=%d", len(X_train), len(X_val), len(X_test))

    X_train_scaled = fe.fit_transform_train(X_train)
    X_val_scaled = fe.transform(X_val)
    X_test_scaled = fe.transform(X_test)

    # 4. Train
    trainer = ModelTrainer()
    trainer.train(X_train_scaled, y_train)

    # 5. Evaluate on val set (for monitoring) and test set (final numbers)
    val_metrics = trainer.evaluate(X_val_scaled, y_val)
    logger.info("Validation metrics: %s", val_metrics)
    test_metrics = trainer.evaluate(X_test_scaled, y_test)
    logger.info("Test metrics: %s", test_metrics)

    # 6. Persist artifacts
    joblib.dump(trainer.model, os.path.join(MODEL_DIR, "logistic_regression.joblib"))
    joblib.dump(fe.scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(fe.label_encoders, os.path.join(MODEL_DIR, "label_encoders.joblib"))
    joblib.dump(fe.feature_names, os.path.join(MODEL_DIR, "feature_names.joblib"))
    logger.info("Artifacts saved to '%s'", MODEL_DIR)

    # Save test set for API metrics endpoint
    test_df = X_test.copy()
    test_df["loan_status"] = y_test.values
    test_df.to_csv(os.path.join(MODEL_DIR, "test_data.csv"), index=False)

    # Save metrics JSON for API
    with open(os.path.join(MODEL_DIR, "model_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info("Test metrics saved to model_metrics.json")

    # 7. Drift baseline — save training distribution for future drift checks
    X_train.to_csv(os.path.join(MODEL_DIR, "train_reference.csv"), index=False)
    logger.info("Training reference data saved for drift detection")

    logger.info("Training pipeline complete.")


if __name__ == "__main__":
    main()
