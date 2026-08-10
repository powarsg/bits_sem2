"""Tests for DataQualityChecker — unit and data validation tests."""

import numpy as np
import pandas as pd
import pytest

from src.data_quality import DataQualityChecker


@pytest.fixture
def checker() -> DataQualityChecker:
    return DataQualityChecker()


class TestMissingValueCheck:
    def test_no_missing_passes(self, checker, sample_df):
        result = checker.check_missing(sample_df)
        assert result["passed"] is True
        assert result["total_missing"] == 0

    def test_with_missing_fails(self, checker, sample_df):
        sample_df.loc[0, "person_age"] = np.nan
        result = checker.check_missing(sample_df)
        assert result["passed"] is False
        assert result["total_missing"] == 1

    def test_missing_pct_computed_correctly(self, checker, sample_df):
        n = len(sample_df)
        # Inject 10 missing values in person_income
        sample_df.loc[:9, "person_income"] = np.nan
        result = checker.check_missing(sample_df)
        assert abs(result["per_column"]["person_income"] - (10 / n * 100)) < 0.1


class TestSchemaValidation:
    def test_valid_df_passes(self, checker, sample_df):
        result = checker.validate_schema(sample_df)
        assert result["passed"] is True
        assert len(result["issues"]) == 0

    def test_missing_column_fails(self, checker, sample_df):
        df = sample_df.drop(columns=["loan_intent"])
        result = checker.validate_schema(df)
        assert result["passed"] is False
        assert any("loan_intent" in issue for issue in result["issues"])

    def test_wrong_type_flagged(self, checker, sample_df):
        df = sample_df.copy()
        df["person_age"] = df["person_age"].astype(str)  # numeric → string
        result = checker.validate_schema(df)
        assert result["passed"] is False


class TestCategoryCheck:
    def test_valid_categories_pass(self, checker, sample_df):
        result = checker.check_categories(sample_df)
        assert result["passed"] is True

    def test_unknown_category_flagged(self, checker, sample_df):
        df = sample_df.copy()
        df.loc[0, "person_gender"] = "nonbinary"
        result = checker.check_categories(df)
        assert result["passed"] is False
        assert "person_gender" in result["violations"]


class TestNumericBounds:
    def test_normal_data_passes(self, checker, sample_df):
        result = checker.check_numeric_bounds(sample_df)
        assert result["passed"] is True

    def test_out_of_range_value_flagged(self, checker, sample_df):
        df = sample_df.copy()
        df.loc[0, "person_age"] = 150
        result = checker.check_numeric_bounds(df)
        assert result["passed"] is False
        assert "person_age" in result["out_of_bounds"]


class TestDriftDetection:
    def test_identical_distributions_no_drift(self, checker, sample_df):
        result = checker.check_drift(sample_df, sample_df)
        assert result["overall_drift_detected"] is False

    def test_very_different_distributions_drift_detected(self, checker, sample_df):
        shifted = sample_df.copy()
        # Shift income by 10x to simulate severe drift
        shifted["person_income"] = shifted["person_income"] * 10
        result = checker.check_drift(
            sample_df, shifted, numeric_cols=["person_income"]
        )
        assert result["overall_drift_detected"] is True
        assert result["columns"]["person_income"]["drift_detected"] is True


class TestFullReport:
    def test_full_report_clean_data(self, checker, sample_df):
        report = checker.full_report(sample_df)
        assert report["overall_passed"] is True
        assert report["row_count"] == len(sample_df)

    def test_full_report_contains_expected_sections(self, checker, sample_df):
        report = checker.full_report(sample_df)
        assert "missing_values" in report
        assert "schema" in report
        assert "categories" in report
        assert "numeric_bounds" in report
