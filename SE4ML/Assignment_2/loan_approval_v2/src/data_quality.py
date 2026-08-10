"""Data quality module — schema validation, missing-value checks, drift detection."""

import pandas as pd
from scipy import stats

from .logger import get_logger

logger = get_logger("data_quality")

EXPECTED_SCHEMA: dict[str, str] = {
    "person_age": "numeric",
    "person_gender": "categorical",
    "person_education": "categorical",
    "person_income": "numeric",
    "person_emp_exp": "numeric",
    "person_home_ownership": "categorical",
    "loan_amnt": "numeric",
    "loan_intent": "categorical",
    "loan_int_rate": "numeric",
    "loan_percent_income": "numeric",
    "cb_person_cred_hist_length": "numeric",
    "credit_score": "numeric",
    "previous_loan_defaults_on_file": "categorical",
}

VALID_CATEGORIES: dict[str, set] = {
    "person_gender": {"male", "female"},
    "person_education": {"High School", "Associate", "Bachelor", "Master", "Doctorate"},
    "person_home_ownership": {"RENT", "OWN", "MORTGAGE", "OTHER"},
    "loan_intent": {"PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"},
    "previous_loan_defaults_on_file": {"Yes", "No"},
}

NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "person_age": (18.0, 120.0),
    "person_income": (0.0, 1e8),
    "person_emp_exp": (0.0, 130.0),
    "loan_amnt": (100.0, 1e7),
    "loan_int_rate": (0.0, 100.0),
    "loan_percent_income": (0.0, 1.0),
    "cb_person_cred_hist_length": (0.0, 60.0),
    "credit_score": (300.0, 850.0),
}


class DataQualityChecker:
    """Measures and reports data-quality metrics for loan application data."""

    # ------------------------------------------------------------------
    # Missing value checks
    # ------------------------------------------------------------------

    def check_missing(self, df: pd.DataFrame) -> dict:
        """Return per-column and overall missing-value statistics."""
        total_cells = df.shape[0] * df.shape[1]
        missing_per_col = df.isnull().sum()
        missing_pct_per_col = (missing_per_col / len(df) * 100).round(2)
        total_missing = int(missing_per_col.sum())
        overall_pct = round(100 * total_missing / total_cells, 2)

        if total_missing > 0:
            logger.warning(
                "Missing values detected: %d cells (%.2f%% of dataset)",
                total_missing,
                overall_pct,
            )
            for col in missing_per_col[missing_per_col > 0].index:
                logger.warning(
                    "  Column '%s': %d missing (%.1f%%)",
                    col, missing_per_col[col], missing_pct_per_col[col],
                )
        else:
            logger.info("Missing value check passed — no missing values")

        return {
            "total_missing": total_missing,
            "overall_missing_pct": overall_pct,
            "per_column": missing_pct_per_col.to_dict(),
            "passed": total_missing == 0,
        }

    # ------------------------------------------------------------------
    # Schema / type validation
    # ------------------------------------------------------------------

    def validate_schema(self, df: pd.DataFrame) -> dict:
        """Verify column types and presence against EXPECTED_SCHEMA."""
        issues = []
        for col, dtype_class in EXPECTED_SCHEMA.items():
            if col not in df.columns:
                issues.append(f"Column '{col}' is missing")
                logger.error("Schema issue: column '%s' missing", col)
                continue
            if dtype_class == "numeric" and not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(f"Column '{col}' expected numeric, got {df[col].dtype}")
                logger.warning("Schema issue: '%s' expected numeric, got %s", col, df[col].dtype)
            elif dtype_class == "categorical" and pd.api.types.is_numeric_dtype(df[col]):
                issues.append(f"Column '{col}' expected categorical, got {df[col].dtype}")
                logger.warning("Schema issue: '%s' expected categorical, got %s", col, df[col].dtype)

        if issues:
            logger.error("Schema validation FAILED — %d issue(s)", len(issues))
        else:
            logger.info("Schema validation passed")

        return {"passed": len(issues) == 0, "issues": issues}

    # ------------------------------------------------------------------
    # Categorical value validation
    # ------------------------------------------------------------------

    def check_categories(self, df: pd.DataFrame) -> dict:
        """Detect values outside the known category sets."""
        violations: dict[str, list] = {}
        for col, valid_set in VALID_CATEGORIES.items():
            if col not in df.columns:
                continue
            observed = set(df[col].dropna().astype(str).unique())
            unknown = observed - valid_set
            if unknown:
                violations[col] = sorted(unknown)
                logger.warning("Unknown categories in '%s': %s", col, unknown)
        if not violations:
            logger.info("Category check passed — all values within expected sets")
        return {"passed": len(violations) == 0, "violations": violations}

    # ------------------------------------------------------------------
    # Numeric range / bounds check
    # ------------------------------------------------------------------

    def check_numeric_bounds(self, df: pd.DataFrame) -> dict:
        """Flag rows where numeric columns fall outside expected bounds."""
        out_of_bounds: dict[str, int] = {}
        for col, (lo, hi) in NUMERIC_BOUNDS.items():
            if col not in df.columns:
                continue
            mask = (df[col] < lo) | (df[col] > hi)
            count = int(mask.sum())
            if count > 0:
                out_of_bounds[col] = count
                logger.warning(
                    "Column '%s' has %d values outside [%.1f, %.1f]", col, count, lo, hi
                )
        if not out_of_bounds:
            logger.info("Numeric bounds check passed")
        return {"passed": len(out_of_bounds) == 0, "out_of_bounds": out_of_bounds}

    # ------------------------------------------------------------------
    # Distribution drift detection (KS test)
    # ------------------------------------------------------------------

    def check_drift(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        numeric_cols: list[str] | None = None,
        alpha: float = 0.05,
    ) -> dict:
        """Run Kolmogorov-Smirnov test for distribution drift on numeric columns.

        Returns per-column KS statistics and whether drift was detected (p < alpha).
        """
        if numeric_cols is None:
            numeric_cols = [c for c in NUMERIC_BOUNDS if c in reference.columns and c in current.columns]

        drift_results: dict[str, dict] = {}
        drift_detected = False
        for col in numeric_cols:
            stat, p_value = stats.ks_2samp(
                reference[col].dropna().values, current[col].dropna().values
            )
            drifted = bool(p_value < alpha)
            drift_results[col] = {
                "ks_statistic": round(float(stat), 4),
                "p_value": round(float(p_value), 4),
                "drift_detected": drifted,
            }
            if drifted:
                drift_detected = True
                logger.warning(
                    "Drift detected in '%s' — KS=%.4f, p=%.4f (alpha=%.2f)",
                    col, stat, p_value, alpha,
                )
            else:
                logger.debug("No drift in '%s' — KS=%.4f, p=%.4f", col, stat, p_value)

        if not drift_detected:
            logger.info("Drift check passed — no significant drift detected")
        return {"overall_drift_detected": drift_detected, "columns": drift_results}

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------

    def full_report(self, df: pd.DataFrame) -> dict:
        """Run all quality checks and return a consolidated report."""
        logger.info("Running full data-quality report on %d rows", len(df))
        report = {
            "row_count": len(df),
            "missing_values": self.check_missing(df),
            "schema": self.validate_schema(df),
            "categories": self.check_categories(df),
            "numeric_bounds": self.check_numeric_bounds(df),
        }
        all_passed = all(v.get("passed", True) for v in report.values() if isinstance(v, dict))
        report["overall_passed"] = all_passed
        if all_passed:
            logger.info("Data quality report: ALL CHECKS PASSED")
        else:
            logger.warning("Data quality report: SOME CHECKS FAILED")
        return report
