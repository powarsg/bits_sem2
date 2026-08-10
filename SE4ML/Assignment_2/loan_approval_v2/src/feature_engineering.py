"""Feature engineering module — encoding and scaling."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .logger import get_logger

logger = get_logger("feature_engineering")

TARGET_COLUMN = "loan_status"
CATEGORICAL_COLUMNS = [
    "person_gender",
    "person_education",
    "person_home_ownership",
    "loan_intent",
    "previous_loan_defaults_on_file",
]


class FeatureEngineering:
    """Encodes categorical features and scales numeric features.

    Follows the fit-on-train / transform-on-all convention to prevent leakage.
    """

    def __init__(self) -> None:
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler: StandardScaler | None = None
        self.feature_names: list[str] = []
        logger.info("FeatureEngineering initialised")

    # ------------------------------------------------------------------
    # Splitting helpers
    # ------------------------------------------------------------------

    def split_features_target(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Return (X, y) from a labelled DataFrame."""
        if TARGET_COLUMN not in df.columns:
            logger.error("Target column '%s' not found in DataFrame", TARGET_COLUMN)
            raise KeyError(f"Target column '{TARGET_COLUMN}' not found")
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]
        logger.info("Split features/target — X: %s, y: %s", X.shape, y.shape)
        return X, y

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def fit_encoders(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """Fit LabelEncoders on training data and return encoded DataFrame."""
        logger.info("Fitting LabelEncoders on %d training rows", len(X_train))
        X_enc = X_train.copy()
        for col in CATEGORICAL_COLUMNS:
            if col not in X_train.columns:
                logger.warning("Categorical column '%s' not found — skipping", col)
                continue
            le = LabelEncoder()
            X_enc[col] = le.fit_transform(X_train[col].astype(str))
            self.label_encoders[col] = le
            logger.debug("Encoded '%s': %s", col, dict(zip(le.classes_, le.transform(le.classes_))))
        self.feature_names = list(X_enc.columns)
        logger.info("Encoding complete — %d features", len(self.feature_names))
        return X_enc

    def transform_encoders(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted encoders to a DataFrame (val / test / inference)."""
        if not self.label_encoders:
            logger.error("Encoders not fitted — call fit_encoders() first")
            raise RuntimeError("Encoders not fitted")
        X_enc = X.copy()
        for col, le in self.label_encoders.items():
            if col not in X.columns:
                logger.warning("Column '%s' missing in input — skipping", col)
                continue
            # Handle unseen labels gracefully
            known = set(le.classes_)
            unseen = set(X[col].astype(str).unique()) - known
            if unseen:
                logger.warning("Unseen labels in '%s': %s — mapping to 0", col, unseen)
            X_enc[col] = X[col].astype(str).apply(
                lambda v: le.transform([v])[0] if v in known else 0
            )
        return X_enc

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------

    def fit_scaler(self, X_enc: pd.DataFrame) -> pd.DataFrame:
        """Fit StandardScaler on encoded training data."""
        logger.info("Fitting StandardScaler on %d rows", len(X_enc))
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(X_enc)
        logger.info("Scaler fitted — mean: min=%.4f max=%.4f",
                    float(np.min(self.scaler.mean_)), float(np.max(self.scaler.mean_)))
        return pd.DataFrame(scaled, columns=list(X_enc.columns))

    def transform_scaler(self, X_enc: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scaler to encoded data."""
        if self.scaler is None:
            logger.error("Scaler not fitted — call fit_scaler() first")
            raise RuntimeError("Scaler not fitted")
        scaled = self.scaler.transform(X_enc)
        return pd.DataFrame(scaled, columns=list(X_enc.columns))

    # ------------------------------------------------------------------
    # Convenience end-to-end helpers
    # ------------------------------------------------------------------

    def fit_transform_train(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """Fit encoders + scaler on train set and return scaled DataFrame."""
        X_enc = self.fit_encoders(X_train)
        return self.fit_scaler(X_enc)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform a new DataFrame using fitted encoders and scaler."""
        X_enc = self.transform_encoders(X)
        return self.transform_scaler(X_enc)
