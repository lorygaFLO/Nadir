---
tags: [recipes, howto, nadir]
aliases: [Cookbook, Cheatsheet]
---

# 🍳 Recipes

Ready-made commands and tips for working on Nadir. Commands assume **Windows PowerShell** from the repo root; Linux/macOS variants noted where they differ.

## Create a virtual environment

### With conda (recommended for this repo)

```powershell
conda create -n nadir python=3.12 -y
conda activate nadir
pip install -r requirements.txt
```

> [!important] Always activate first
> The project convention is to run everything inside the `nadir` env — `conda activate nadir` **before** any test/script/notebook. The base env will miss dependencies.

Useful extras:

```powershell
conda env list                 # see all envs
conda deactivate               # leave the env
conda remove -n nadir --all    # nuke and recreate if broken
conda env export --from-history > environment.yml   # snapshot the env
```

### Without conda (vanilla venv)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # PowerShell
# .venv\Scripts\activate.bat        # cmd.exe
# source .venv/bin/activate         # Linux/macOS
pip install -r requirements.txt
```

> [!tip] PowerShell blocks activation?
> If you get *"running scripts is disabled"*: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`, then retry.

Deactivate with `deactivate`. Delete the env by just removing the `.venv` folder.

### Pick the env in VS Code

`Ctrl+Shift+P` → **Python: Select Interpreter** → choose `nadir` (conda) or `.venv`. This is what the test runner and notebooks will use.

## Run a marimo notebook

Notebooks live in `notebooks/` and are **marimo** notebooks (plain `.py` files, reactive, git-friendly):

```powershell
conda activate nadir
marimo edit notebooks/master_notebook.py        # interactive editor in the browser
```

Other modes:

```powershell
marimo run notebooks/master_notebook.py         # read-only app mode (no code editing)
marimo edit                                     # browse/create notebooks from a file picker
```

- `master_notebook.py` — full end-to-end pipeline demo.
- `phase1_sandbox.py` — poset/dominance experiments.
- `cost_functions_sandbox.py` — cost curve exploration.

> [!note] marimo ≠ Jupyter
> There is no `.ipynb` here. Cells re-run automatically when their dependencies change; the file on disk is pure Python.

## Run the test suite

```powershell
conda activate nadir
python -m pytest -q                       # all tests, quiet
python -m pytest tests/test_poset.py -q   # one file
python -m pytest -q -k "frontier"         # only tests matching a keyword
python -m pytest -q -x                    # stop at first failure
python -m pytest --lf -q                  # rerun only last failures
```

## Install / update dependencies

```powershell
pip install -r requirements.txt          # install everything
pip install <package>                    # add one (then add it to requirements.txt!)
pip list --outdated                      # what could be updated
```

## Quick sanity-check the pipeline

Smoke-test imports and a snapshot without opening a notebook:

```powershell
python -c "from core.poset import PosetEngine; print('core imports OK')"
```

## Regenerate/inspect the example data

The toy dataset is [data/example_subjects.csv](../../data/example_subjects.csv) (see [[Data Format]]). To eyeball it fast:

```powershell
python -c "import polars as pl; print(pl.read_csv('data/example_subjects.csv'))"
```

## Common gotchas

- **`ModuleNotFoundError: core`** → run from the repo root (imports are `from core.x import ...`), and make sure the right env is active.
- **`ValueError: missing required column(s)`** → dataset columns must be UPPERCASE and match `kpi_definitions` → [[Data Format]].
- **Weird optimizer behavior after config edits** → check `alpha` scales in [[Configuration]]; a too-small α makes a KPI "free" and it soaks up the whole budget.
- **Non-reproducible runs** → drift is stochastic; fix `random_seed` in [[Configuration]].
- **Don't search for `positive_cap`** → it was removed; the metric now uses projected gaps ([[03 - Signed Distance Metric]]).

## Open this vault in Obsidian

Obsidian → **Open folder as vault** → select `docs/docs`. Wikilinks (`[[...]]`), callouts and Mermaid diagrams all render natively. Start from [[Home]].
