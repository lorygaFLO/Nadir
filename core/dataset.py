"""Dataset ingestion for user-provided subject histories.

Loads a wide-format table from CSV or Parquet with the schema:

    SUBJECT | TIME | <KPI_1> | <KPI_2> | ... | <KPI_n>

Column names must be UPPERCASE. The KPI columns must match (uppercased)
the ``kpi_definitions`` names from the YAML config. Extra columns are
ignored; missing required columns raise a ``ValueError``.

Missing data policy: null KPI values and entirely absent (SUBJECT, TIME)
rows are forward-filled with the subject's last observed value. Gaps
before a subject's first observation cannot be filled and raise a
``ValueError``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl

from core.poset import KpiSpec

SUBJECT_COL = "SUBJECT"
TIME_COL = "TIME"


def required_columns(kpi_specs: Sequence[KpiSpec]) -> list[str]:
    """Uppercase column set expected in a user dataset, in canonical order."""
    return [SUBJECT_COL, TIME_COL] + [spec.name.upper() for spec in kpi_specs]


def load_subject_dataset(
    path: str | Path, kpi_specs: Sequence[KpiSpec]
) -> pl.DataFrame:
    """Read a CSV or Parquet file and validate it against the KPI specs.

    Returns a DataFrame restricted to the required columns (extras are
    dropped), with KPI columns cast to Float64, sorted by (TIME, SUBJECT).
    Null KPI values and missing (SUBJECT, TIME) rows are forward-filled
    with the last observed value per subject.

    Raises
    ------
    ValueError
        If the file extension is unsupported, any required column is
        missing, or a gap cannot be forward-filled (no prior observation).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pl.read_csv(path)
    elif suffix in (".parquet", ".pq"):
        df = pl.read_parquet(path)
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}': expected .csv or .parquet."
        )

    required = required_columns(kpi_specs)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset '{path.name}' is missing required column(s): {missing}. "
            f"Expected columns (uppercase): {required}."
        )

    kpi_cols = required[2:]
    df = df.select(required).with_columns(  # select drops any extra columns
        pl.col(SUBJECT_COL).cast(pl.Utf8),
        pl.col(TIME_COL).cast(pl.Int64),
        *[pl.col(c).cast(pl.Float64) for c in kpi_cols],
    )
    return _forward_fill_gaps(df, kpi_cols).sort([TIME_COL, SUBJECT_COL])


def _forward_fill_gaps(df: pl.DataFrame, kpi_cols: Sequence[str]) -> pl.DataFrame:
    """Complete missing (SUBJECT, TIME) rows and null KPI values.

    Builds the full subject x time grid and propagates each subject's
    last observed value forward. Raises ``ValueError`` if a gap precedes
    the subject's first observation (nothing to propagate).
    """
    grid = df.select(pl.col(SUBJECT_COL).unique()).join(
        df.select(pl.col(TIME_COL).unique()), how="cross"
    )
    filled = (
        grid.join(df, on=[SUBJECT_COL, TIME_COL], how="left")
        .sort([SUBJECT_COL, TIME_COL])
        .with_columns(pl.col(kpi_cols).forward_fill().over(SUBJECT_COL))
    )

    unfillable = filled.filter(
        pl.any_horizontal(pl.col(c).is_null() for c in kpi_cols)
    )
    if unfillable.height:
        offenders = sorted(unfillable[SUBJECT_COL].unique().to_list())
        raise ValueError(
            f"Subject(s) {offenders} have missing KPI values with no prior "
            "observation to forward-fill."
        )
    return filled


def extract_snapshot(
    df: pl.DataFrame,
    kpi_specs: Sequence[KpiSpec],
    time: int | None = None,
) -> dict[str, np.ndarray]:
    """Subject -> KPI vector mapping at one period, ready for ``PosetEngine``.

    Uses the latest available ``TIME`` when ``time`` is omitted. Vector
    component order follows ``kpi_specs`` order.
    """
    t = int(df[TIME_COL].max()) if time is None else int(time)
    snap = df.filter(pl.col(TIME_COL) == t)
    if snap.is_empty():
        raise ValueError(f"No rows found for TIME == {t}.")

    kpi_cols = [spec.name.upper() for spec in kpi_specs]
    return {
        row[SUBJECT_COL]: np.array([row[c] for c in kpi_cols], dtype=float)
        for row in snap.iter_rows(named=True)
    }
