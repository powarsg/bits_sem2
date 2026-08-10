"""Inference engine — loads serialised artifacts and serves predictions."""

import os
import joblib
import pandas as pd
import numpy as np

from .logger import get_logger

logger = get_logger("inference")

# Default artifact paths (relative to repo root)
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model_artifacts")


class InferenceEngine:
    """Loads trained artifacts and provides predict / predict_proba methods."""

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR) -> None:
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.label_encoders: dict = {}
        self.feature_names: list[str] = []
        logger.info("InferenceEngine initialised — model_dir: %s", model_dir)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_artifacts(self) -> None:
        """Load all serialised artifacts from model_dir."""
        artifacts = {
            "model": "logistic_regression.joblib",
            "scaler": "scaler.joblib",
            "label_encoders": "label_encoders.joblib",
            "feature_names": "feature_names.joblib",
        }
        for attr, filename in artifacts.items():
            path = os.path.join(self.model_dir, filename)
            if not os.path.exists(path):
                logger.error("Artifact not found: %s", path)
                raise FileNotFoundError(f"Artifact not found: {path}")
            setattr(self, attr, joblib.load(path))
            logger.info("Loaded artifact: %s", filename)

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply label encoders then scaler to raw input."""
        if not self.label_encoders or self.scaler is None:
            logger.error("Artifacts not loaded — call load_artifacts() first")
            raise RuntimeError("Artifacts not loaded")

        df_enc = df.copy()
        for col, le in self.label_encoders.items():
            if col not in df_enc.columns:
                logger.warning("Column '%s' missing in input — defaulting to 0", col)
                df_enc[col] = 0
                continue
            known = set(le.classes_)
            df_enc[col] = df_enc[col].astype(str).apply(
                lambda v: le.transform([v])[0] if v in known else 0
            )

        # Align columns to training order
        df_enc = df_enc.reindex(columns=self.feature_names, fill_value=0)
        scaled = self.scaler.transform(df_enc)
        return pd.DataFrame(scaled, columns=self.feature_names)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return binary class predictions for input DataFrame."""
        logger.info("Running inference on %d rows", len(df))
        X = self._preprocess(df)
        predictions = self.model.predict(X)
        approved = int(np.sum(predictions))
        logger.info(
            "Inference complete — approved=%d / %d (%.1f%%)",
            approved,
            len(predictions),
            100 * approved / len(predictions),
        )
        return predictions

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return probability estimates [P(reject), P(approve)] per row."""
        X = self._preprocess(df)
        probs = self.model.predict_proba(X)
        logger.debug("Probabilities — min=%.4f  max=%.4f", float(probs[:, 1].min()), float(probs[:, 1].max()))
        return probs

    def predict_single(self, record: dict) -> dict:
        """Run inference on a single record dict and return a result dict."""
        df = pd.DataFrame([record])
        probs = self.predict_proba(df)
        label = int(self.model.predict(self._preprocess(df))[0])
        result = {
            "loan_status": label,
            "decision": "Approved" if label == 1 else "Rejected",
            "approval_probability": round(float(probs[0][1]), 4),
        }
        logger.info("Single prediction: %s (p=%.4f)", result["decision"], result["approval_probability"])
        return result
