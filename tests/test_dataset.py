"""Tests for core.dataset: CSV/Parquet ingestion and validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from core.dataset import extract_snapshot, load_subject_dataset, required_columns
from core.poset import KpiSpec, PosetEngine

KPI_SPECS = [
    KpiSpec(name="market_share_pct", higher_is_better=True),
    KpiSpec(name="unit_production_cost", higher_is_better=False),
]

VALID_ROWS = {
    "SUBJECT": ["A", "B", "A", "B"],
    "TIME": [0, 0, 1, 1],
    "MARKET_SHARE_PCT": [10.0, 20.0, 11.0, 19.0],
    "UNIT_PRODUCTION_COST": [90.0, 80.0, 88.0, 81.0],
}


def _write_csv(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "subjects.csv"
    pl.DataFrame(data).write_csv(path)
    return path


class TestLoadSubjectDataset:
    def test_loads_valid_csv(self, tmp_path: Path) -> None:
        df = load_subject_dataset(_write_csv(tmp_path, VALID_ROWS), KPI_SPECS)
        assert df.columns == required_columns(KPI_SPECS)
        assert df.height == 4

    def test_loads_parquet(self, tmp_path: Path) -> None:
        path = tmp_path / "subjects.parquet"
        pl.DataFrame(VALID_ROWS).write_parquet(path)
        df = load_subject_dataset(path, KPI_SPECS)
        assert df.height == 4

    def test_extra_columns_are_dropped(self, tmp_path: Path) -> None:
        data = dict(VALID_ROWS, REGION=["EU", "EU", "US", "US"])
        df = load_subject_dataset(_write_csv(tmp_path, data), KPI_SPECS)
        assert "REGION" not in df.columns

    def test_missing_kpi_column_raises(self, tmp_path: Path) -> None:
        data = {k: v for k, v in VALID_ROWS.items() if k != "UNIT_PRODUCTION_COST"}
        with pytest.raises(ValueError, match="UNIT_PRODUCTION_COST"):
            load_subject_dataset(_write_csv(tmp_path, data), KPI_SPECS)

    def test_missing_subject_column_raises(self, tmp_path: Path) -> None:
        data = {k: v for k, v in VALID_ROWS.items() if k != "SUBJECT"}
        with pytest.raises(ValueError, match="SUBJECT"):
            load_subject_dataset(_write_csv(tmp_path, data), KPI_SPECS)

    def test_lowercase_columns_rejected(self, tmp_path: Path) -> None:
        data = {k.lower(): v for k, v in VALID_ROWS.items()}
        with pytest.raises(ValueError, match="missing required column"):
            load_subject_dataset(_write_csv(tmp_path, data), KPI_SPECS)

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "subjects.json"
        path.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_subject_dataset(path, KPI_SPECS)


class TestForwardFill:
    def test_null_kpi_value_forward_filled(self, tmp_path: Path) -> None:
        data = dict(VALID_ROWS, MARKET_SHARE_PCT=[10.0, 20.0, None, 19.0])
        df = load_subject_dataset(_write_csv(tmp_path, data), KPI_SPECS)
        a1 = df.filter((pl.col("SUBJECT") == "A") & (pl.col("TIME") == 1))
        assert a1["MARKET_SHARE_PCT"].item() == 10.0  # carried from t=0

    def test_missing_year_row_completed(self, tmp_path: Path) -> None:
        data = {
            "SUBJECT": ["A", "B", "A"],  # B has no row at t=1
            "TIME": [0, 0, 1],
            "MARKET_SHARE_PCT": [10.0, 20.0, 11.0],
            "UNIT_PRODUCTION_COST": [90.0, 80.0, 88.0],
        }
        df = load_subject_dataset(_write_csv(tmp_path, data), KPI_SPECS)
        assert df.height == 4  # full subject x time grid
        b1 = df.filter((pl.col("SUBJECT") == "B") & (pl.col("TIME") == 1))
        assert b1["MARKET_SHARE_PCT"].item() == 20.0
        assert b1["UNIT_PRODUCTION_COST"].item() == 80.0

    def test_gap_before_first_observation_raises(self, tmp_path: Path) -> None:
        data = {
            "SUBJECT": ["A", "B"],  # B first appears at t=1: no value at t=0
            "TIME": [0, 1],
            "MARKET_SHARE_PCT": [10.0, 20.0],
            "UNIT_PRODUCTION_COST": [90.0, 80.0],
        }
        with pytest.raises(ValueError, match="B"):
            load_subject_dataset(_write_csv(tmp_path, data), KPI_SPECS)


class TestExtractSnapshot:
    def test_latest_time_by_default(self, tmp_path: Path) -> None:
        df = load_subject_dataset(_write_csv(tmp_path, VALID_ROWS), KPI_SPECS)
        snapshot = extract_snapshot(df, KPI_SPECS)
        np.testing.assert_allclose(snapshot["A"], [11.0, 88.0])
        np.testing.assert_allclose(snapshot["B"], [19.0, 81.0])

    def test_explicit_time(self, tmp_path: Path) -> None:
        df = load_subject_dataset(_write_csv(tmp_path, VALID_ROWS), KPI_SPECS)
        snapshot = extract_snapshot(df, KPI_SPECS, time=0)
        np.testing.assert_allclose(snapshot["A"], [10.0, 90.0])

    def test_missing_time_raises(self, tmp_path: Path) -> None:
        df = load_subject_dataset(_write_csv(tmp_path, VALID_ROWS), KPI_SPECS)
        with pytest.raises(ValueError, match="TIME == 5"):
            extract_snapshot(df, KPI_SPECS, time=5)

    def test_snapshot_feeds_poset_engine(self, tmp_path: Path) -> None:
        df = load_subject_dataset(_write_csv(tmp_path, VALID_ROWS), KPI_SPECS)
        engine = PosetEngine(kpi_specs=KPI_SPECS, subjects=extract_snapshot(df, KPI_SPECS))
        # B has higher share and lower cost at t=1: B dominates A.
        assert engine.get_maximal_elements() == ["B"]
