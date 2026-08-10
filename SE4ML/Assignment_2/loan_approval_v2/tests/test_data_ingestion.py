"""Unit tests for DataIngestion class."""

import os

import pandas as pd
import pytest

from src.data_ingestion import DataIngestion, REQUIRED_COLUMNS


class TestDataIngestionLoad:
    def test_load_returns_dataframe(self, tmp_csv):
        di = DataIngestion(tmp_csv)
        df = di.load()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_missing_file_raises(self, tmp_path):
        di = DataIngestion(str(tmp_path / "nonexistent.csv"))
        with pytest.raises(FileNotFoundError):
            di.load()


class TestSchemaValidation:
    def test_valid_schema_passes(self, sample_df):
        di = DataIngestion("dummy.csv")
        result = di.validate_schema(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_missing_column_raises(self, sample_df):
        di = DataIngestion("dummy.csv")
        df_incomplete = sample_df.drop(columns=["loan_status"])
        with pytest.raises(ValueError, match="Missing required columns"):
            di.validate_schema(df_incomplete)


class TestDropMissing:
    def test_no_missing_unchanged(self, sample_df):
        di = DataIngestion("dummy.csv")
        result = di.drop_missing(sample_df)
        assert len(result) == len(sample_df)

    def test_rows_with_nan_dropped(self, sample_df):
        di = DataIngestion("dummy.csv")
        sample_df.loc[0, "person_age"] = float("nan")
        result = di.drop_missing(sample_df)
        assert len(result) == len(sample_df) - 1


class TestRemoveOutliers:
    def test_normal_ages_unchanged(self, sample_df):
        di = DataIngestion("dummy.csv")
        result = di.remove_outliers(sample_df)
        assert len(result) == len(sample_df)

    def test_outlier_age_removed(self, sample_df):
        di = DataIngestion("dummy.csv")
        sample_df.loc[0, "person_age"] = 150
        result = di.remove_outliers(sample_df)
        assert 150 not in result["person_age"].values
        assert len(result) == len(sample_df) - 1


class TestIngestPipeline:
    def test_full_ingest_returns_clean_df(self, tmp_csv):
        di = DataIngestion(tmp_csv)
        df = di.ingest()
        assert isinstance(df, pd.DataFrame)
        assert df.isnull().sum().sum() == 0
        assert (df["person_age"] <= 120).all()
