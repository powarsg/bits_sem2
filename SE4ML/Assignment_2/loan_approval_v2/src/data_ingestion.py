"""Data ingestion module — loads and performs basic cleaning on raw CSV data."""

import pandas as pd

from .logger import get_logger

logger = get_logger("data_ingestion")

REQUIRED_COLUMNS = [
    "person_age",
    "person_gender",
    "person_education",
    "person_income",
    "person_emp_exp",
    "person_home_ownership",
    "loan_amnt",
    "loan_intent",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
    "previous_loan_defaults_on_file",
    "loan_status",
]


class DataIngestion:
    """Handles loading, schema validation, and basic cleaning of raw data."""

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        logger.info("DataIngestion initialised with filepath: %s", filepath)

    def load(self) -> pd.DataFrame:
        """Load CSV from disk and return a raw DataFrame."""
        logger.info("Loading dataset from '%s'", self.filepath)
        try:
            df = pd.read_csv(self.filepath)
        except FileNotFoundError:
            logger.error("File not found: %s", self.filepath)
            raise
        except Exception as exc:
            logger.error("Unexpected error reading '%s': %s", self.filepath, exc)
            raise
        logger.info("Loaded %d rows × %d columns", *df.shape)
        return df

    def validate_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Raise ValueError if required columns are missing."""
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            logger.error("Schema validation failed — missing columns: %s", missing)
            raise ValueError(f"Missing required columns: {missing}")
        logger.info("Schema validation passed — all required columns present")
        return df

    def drop_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows containing any NaN values and log statistics."""
        before = len(df)
        df_clean = df.dropna()
        dropped = before - len(df_clean)
        if dropped > 0:
            logger.warning("Dropped %d rows with missing values (%.1f%%)", dropped, 100 * dropped / before)
        else:
            logger.info("No missing values found")
        return df_clean

    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows where person_age > 120 (data-entry errors)."""
        before = len(df)
        df_filtered = df[df["person_age"] <= 120].copy()
        removed = before - len(df_filtered)
        if removed > 0:
            logger.warning("Removed %d outlier rows (person_age > 120)", removed)
        else:
            logger.info("No age outliers detected")
        return df_filtered

    def ingest(self) -> pd.DataFrame:
        """Full ingestion pipeline: load → validate → clean → remove outliers."""
        df = self.load()
        df = self.validate_schema(df)
        df = self.drop_missing(df)
        df = self.remove_outliers(df)
        logger.info("Ingestion complete — final shape: %s", df.shape)
        return df
