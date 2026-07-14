---
tags: [technical, nadir, data]
---

# Data Format

How to prepare a dataset for Nadir. Handled by [`core/dataset.py`](../../core/dataset.py) → [[Core Modules#dataset py]].

## Schema (wide format)

```
SUBJECT | TIME | <KPI_1> | <KPI_2> | ... | <KPI_n>
```

- **File types**: `.csv`, `.parquet` (or `.pq`). Anything else raises `ValueError`.
- **Column names must be UPPERCASE.**
- KPI columns must match the (uppercased) keys of `kpi_definitions` in [[Configuration]].
- **Extra columns are silently dropped** (e.g. a `REGION` column is fine); **missing required columns raise** `ValueError`.

## Types & ordering

On load, columns are cast to: `SUBJECT` → string, `TIME` → int, KPIs → float. Rows are sorted by `(TIME, SUBJECT)`. `TIME` is a discrete period index (0, 1, 2, …).

## Missing data policy

- Null KPI values **and** entirely absent `(SUBJECT, TIME)` rows are **forward-filled** with the subject's last observed value.
- Gaps *before* a subject's first observation cannot be filled → `ValueError`.

## Example

See [data/example_subjects.csv](../../data/example_subjects.csv):

```csv
SUBJECT,TIME,MARKET_SHARE_PCT,UNIT_PRODUCTION_COST,REGION
Us,0,12.5,84.0,EMEA
CompetitorA,0,18.2,71.5,EMEA
CompetitorB,0,15.9,79.3,APAC
...
```

`REGION` is an extra column → ignored. `Us` is the focus subject by convention in the notebooks; the engine itself takes the focus name as a parameter.

## From file to engine

```python
from core.dataset import load_subject_dataset, extract_snapshot

df = load_subject_dataset("data/example_subjects.csv", kpi_specs)
snapshot = extract_snapshot(df, t=0, kpi_specs=kpi_specs)  # {subject: values}
# feed snapshot into PosetEngine via set_subject(...)
```
